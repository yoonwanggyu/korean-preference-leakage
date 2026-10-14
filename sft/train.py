"""교사 한 명의 답변 1,000개로 Qwen3-0.6B-Base를 SFT(full fine-tuning)해 학생 모델을 만든다.

사용법: python sft/train.py --teacher gpt --seed 42   (teacher: gpt | claude | gemini)
        이미 끝난 (teacher, seed)는 건너뛴다. 다시 학습하려면 --force.

세 학생은 교사 답변만 다르고 나머지(질문 id, 프롬프트, 하이퍼파라미터)는 모두 같다. seed는 학습 무작위성
(데이터 순서, dropout)을 바꿔 반복 측정하는 데 쓴다.
손실은 답변 토큰(+ <|im_end|>)에만 걸고, 시스템 프롬프트와 질문 토큰은 -100으로 가려 학습하지 않는다.
학습 중에는 같은 교사의 검증 답변(학습에 안 쓴 199문항)으로 val loss를 주기적으로 잰다. 최종 모델은
val로 고르지 않고 세 학생 모두 마지막 step의 모델을 쓴다.

출력
  sft/checkpoints/{teacher}/seed{seed}/   최종 학생 모델(bf16)과 토크나이저  ← Drive에 두고 내려받지 않아도 됨
  sft/runs/{teacher}_seed{seed}/
    log_history.json   Trainer 전체 기록 (step마다 train loss, 학습률, 기울기 크기 / EVAL_STEPS마다 val loss)
    train_loss.csv     step, epoch, loss, learning_rate, grad_norm
    val_loss.csv       step, epoch, eval_loss  (step 0 = 학습 전)
    run_info.json      하이퍼파라미터, 데이터 통계, 학습 시간, GPU, 라이브러리 버전
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
from prompt import END_TOKEN, SYSTEM_PROMPT, build_prompt  # noqa: E402

load_dotenv(ROOT / ".env")

MODEL_NAME = os.environ.get("STUDENT_TOKENIZER", "Qwen/Qwen3-0.6B-Base")
MAX_LEN = int(os.environ.get("STUDENT_MAX_LEN", 1024))
SFT_DIR = Path(__file__).parent
MODEL_DIR = SFT_DIR / "models" / MODEL_NAME.split("/")[-1]
DATA_DIR = SFT_DIR / "data"
# 학생 모델 저장 위치. Colab에서 Drive 용량이 부족하면 SFT_CKPT_DIR=/content/checkpoints 처럼 로컬 디스크로 바꾼다.
CKPT_DIR = Path(os.environ.get("SFT_CKPT_DIR", SFT_DIR / "checkpoints"))
RUNS_DIR = SFT_DIR / "runs"

# 하이퍼파라미터: 세 교사, 모든 seed에서 같은 값을 쓴다.
EPOCHS = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 8           # GPU 한 번에 넣는 예제 수. 메모리가 부족하면 줄이고 GRAD_ACCUM을 늘린다.
GRAD_ACCUM = 2           # 실효 배치 = BATCH_SIZE * GRAD_ACCUM = 16
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
LR_SCHEDULER = "cosine"
EVAL_STEPS = 10          # 이 간격(업데이트 step)마다 val loss를 잰다
LOGGING_STEPS = 1        # train loss는 매 step 기록한다


def tokenize(example, tok):
    prompt_ids = tok(build_prompt(example["instruction"]), add_special_tokens=False)["input_ids"]
    answer_ids = tok(example["output"] + END_TOKEN, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > MAX_LEN:
        # build_data.py에서 이미 걸렀으므로 여기 오면 데이터가 잘못된 것이다.
        raise ValueError(f"id={example['id']} 길이 {len(input_ids)} > {MAX_LEN}")
    return {"input_ids": input_ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def collate(batch, pad_id):
    """배치 안에서 가장 긴 길이에 맞춰 오른쪽을 패딩한다. 패딩 위치는 attention 0, label -100."""
    width = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention = [], [], []
    for x in batch:
        pad = width - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attention.append([1] * len(x["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


def load_split(teacher, split, tok):
    name = f"{teacher}_train_1000.json" if split == "train" else f"{teacher}_val.json"
    path = DATA_DIR / name
    rows = json.loads(path.read_text(encoding="utf-8"))
    ds = Dataset.from_list(rows).map(lambda x: tokenize(x, tok), remove_columns=list(rows[0].keys()))
    stats = {
        "file": name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_examples": len(rows),
        "ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest(),
        "answer_tokens": sum(sum(1 for l in x if l != -100) for x in ds["labels"]),
        "total_tokens": sum(len(x) for x in ds["input_ids"]),
        "max_tokens": max(len(x) for x in ds["input_ids"]),
    }
    return ds, stats


def write_csvs(log, run_dir):
    train = ["step,epoch,loss,learning_rate,grad_norm"]
    val = ["step,epoch,eval_loss"]
    for x in log:
        if "loss" in x:
            train.append(f"{x['step']},{x.get('epoch', '')},{x['loss']},{x.get('learning_rate', '')},{x.get('grad_norm', '')}")
        if "eval_loss" in x:
            val.append(f"{x['step']},{x.get('epoch', '')},{x['eval_loss']}")
    (run_dir / "train_loss.csv").write_text("\n".join(train) + "\n", encoding="utf-8")
    (run_dir / "val_loss.csv").write_text("\n".join(val) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, choices=["gpt", "claude", "gemini"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="이미 끝난 run도 다시 학습한다")
    parser.add_argument("--smoke", action="store_true", help="코드 점검용: 16개로 1 epoch만 돌리고 smoke 폴더에 저장한다")
    cli = parser.parse_args()
    teacher, seed = cli.teacher, cli.seed

    run_dir = RUNS_DIR / (f"smoke_{teacher}" if cli.smoke else f"{teacher}_seed{seed}")
    final_dir = CKPT_DIR / ("smoke" if cli.smoke else teacher) / f"seed{seed}"
    if (run_dir / "run_info.json").exists() and (final_dir / "config.json").exists() and not cli.force:
        print(f"[건너뜀] {teacher} seed {seed}: 이미 학습됨 ({final_dir})")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    if not (MODEL_DIR / "config.json").exists():
        print(f"모델 다운로드: {MODEL_NAME} -> {MODEL_DIR}")
        snapshot_download(MODEL_NAME, local_dir=MODEL_DIR)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    train_ds, train_stats = load_split(teacher, "train", tok)
    val_ds, val_stats = load_split(teacher, "val", tok)
    if cli.smoke:
        train_ds, val_ds = train_ds.select(range(16)), val_ds.select(range(8))
    print(f"교사 {teacher}, seed {seed}: train {train_stats['n_examples']}개 (답변 토큰 {train_stats['answer_tokens']:,}), "
          f"val {val_stats['n_examples']}개")

    # GPU가 bf16을 지원하면(L4, A100 등) bf16 혼합정밀도, 아니면 fp32로 학습한다. 가중치 원본은 fp32로 둔다.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.config.use_cache = False  # gradient checkpointing과 함께 쓸 수 없다

    args = TrainingArguments(
        output_dir=str(run_dir / "tmp"),
        num_train_epochs=1 if cli.smoke else EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=1 if cli.smoke else BATCH_SIZE,  # 점검 모드는 Mac 메모리에 맞춰 1
        per_device_eval_batch_size=1 if cli.smoke else BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_RATIO,          # 0~1 사이 float는 전체 step 대비 비율
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=use_bf16,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=1 if cli.smoke else EVAL_STEPS,
        eval_on_start=True,                 # 학습 전(step 0) val loss도 기록한다
        prediction_loss_only=True,          # 평가 때 logits를 모으지 않아 메모리를 아낀다
        save_strategy="no",                 # 중간 체크포인트는 저장하지 않고 마지막 모델만 저장한다
        seed=seed,
        data_seed=seed,
        report_to="none",
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    collator = lambda b: collate(b, tok.pad_token_id)  # noqa: E731
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator, processing_class=tok)

    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    final_eval = trainer.evaluate()

    # 최종 모델은 bf16으로 저장해 용량을 절반으로 줄인다(추론에는 bf16이면 충분하다).
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.to(torch.bfloat16).save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))

    shutil.rmtree(run_dir / "tmp", ignore_errors=True)  # Trainer가 만든 빈 임시 폴더
    log = trainer.state.log_history
    (run_dir / "log_history.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    write_csvs(log, run_dir)
    info = {
        "teacher": teacher,
        "seed": seed,
        "base_model": MODEL_NAME,
        "hyperparameters": {
            "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "per_device_batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM, "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "lr_scheduler": LR_SCHEDULER, "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM, "max_len": MAX_LEN, "optimizer": args.optim,
            "bf16": use_bf16, "gradient_checkpointing": True, "eval_steps": EVAL_STEPS,
            "loss": "token-mean cross-entropy on answer tokens + <|im_end|> (prompt masked with -100)",
        },
        "prompt_format": build_prompt("{instruction}") + "{answer}" + END_TOKEN,
        "system_prompt": SYSTEM_PROMPT,
        "data": {"train": train_stats, "val": val_stats},
        "total_steps": trainer.state.global_step,
        "train_runtime_sec": round(elapsed, 1),
        "train_loss_mean": result.training_loss,
        "final_val_loss": final_eval["eval_loss"],
        "environment": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
        },
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {teacher} seed {seed}, {elapsed / 60:.1f}분, 최종 val loss {final_eval['eval_loss']:.4f}")
    print(f"모델: {final_dir}\n기록: {run_dir}")


if __name__ == "__main__":
    main()
