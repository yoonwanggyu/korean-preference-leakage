"""KLUE-RoBERTa를 미세조정해 답변의 생성 모델(GPT/Claude/Gemini)을 분류한다.

사용법: python classify/roberta.py --cond raw   (raw | norm)

logistic.py와 같은 train/val/test 분할(common.load_dataset)을 쓴다.
- train으로 학습하고, EVAL_STEPS마다 val loss·정확도를 재서 val 정확도가 가장 높은 체크포인트를 고른다.
- 고른 모델로 test를 한 번만 평가한다.

출력(classify/output/roberta/):
  metrics_{cond}.json     하이퍼파라미터, 학습 기록, best step, test 정확도·95% CI, 클래스별 P/R/F1
  training_{cond}.png     train/val loss 수렴 곡선과 val 정확도 곡선
  confusion_{cond}.png    test 혼동행렬
  confidence_{cond}.png   test 예측 확신도(정답 클래스 확률) 분포
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

from common import (COLORS, GRID, INK, INK_2, MODELS, OUT_DIR, SEED, SURFACE, bootstrap_ci, get_split, save_predictions,
                    load_dataset, plot_confusion, setup_plot, style_axes)

MODEL_NAME = "klue/roberta-base"
MAX_LEN = 512            # RoBERTa 최대 입력 길이. 넘는 답변(2개)은 뒷부분을 자른다.
EPOCHS = 4
LEARNING_RATE = 2e-5
BATCH_SIZE = 8             # 한 번에 넣는 배치. Mac 메모리 때문에 8로 두고
GRAD_ACCUM = 2            # 2번 누적해 한 번 업데이트한다 (실효 배치 16)
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EVAL_STEPS = 25
LOGGING_STEPS = 5
PAD_MULTIPLE = 128       # 입력 길이를 128의 배수로만 패딩해 MPS가 새 모양마다 메모리를 잡는 것을 막는다
OUT = OUT_DIR / "roberta"
CKPT_DIR = Path(__file__).parent / "checkpoints"
COND_NAME = {"raw": "원문", "norm": "정규화"}


def to_dataset(texts, labels, tok):
    ds = Dataset.from_dict({"text": texts, "label": labels.tolist()})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True,
                  remove_columns=["text"])


class EmptyMpsCache(TrainerCallback):
    """평가가 끝날 때마다 MPS 캐시를 비워 Mac 메모리가 쌓이지 않게 한다."""

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def compute_metrics(p):
    return {"accuracy": float((p.predictions.argmax(-1) == p.label_ids).mean())}


def plot_training(log, best_step, hp, cond, path):
    """왼쪽: train/val loss, 오른쪽: val 정확도. 세로 점선은 선택된(best) 체크포인트."""
    # Trainer 로그의 train loss는 gradient accumulation 횟수만큼 합산돼 기록되므로 평균으로 되돌린다.
    # (확인: 학습 시작 시 로그값 2.2 = 2 × ln3, val loss는 1.06에서 시작)
    train = [(x["step"], x["loss"] / hp.get("grad_accum", 1)) for x in log if "loss" in x]
    val = [(x["step"], x["eval_loss"], x["eval_accuracy"]) for x in log if "eval_loss" in x]
    steps_per_epoch = hp["steps_per_epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax1.plot(*zip(*train), color=INK_2, lw=1.2, label="train loss")
    ax1.plot([v[0] for v in val], [v[1] for v in val], color=COLORS["GPT"], lw=2, marker="o", ms=3.5,
             label="val loss")
    ax1.set_ylabel("cross-entropy loss", fontsize=8.5, color=INK_2)
    ax1.legend(frameon=False, fontsize=8)
    ax2.plot([v[0] for v in val], [v[2] * 100 for v in val], color=COLORS["GPT"], lw=2, marker="o", ms=3.5)
    ax2.set_ylabel("val 정확도 (%)", fontsize=8.5, color=INK_2)
    for ax in (ax1, ax2):
        style_axes(ax)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("학습 step", fontsize=8.5, color=INK_2)
        ax.axvline(best_step, color=INK, lw=1, ls=":")
        for e in range(1, hp["epochs"]):
            ax.axvline(e * steps_per_epoch, color=GRID, lw=0.8)
    best_acc = next(v[2] for v in val if v[0] == best_step)
    ax2.annotate(f"best step {best_step}\nval {best_acc:.1%}", (best_step, best_acc * 100), xytext=(8, -28),
                 textcoords="offset points", fontsize=8, color=INK)
    fig.suptitle(f"KLUE-RoBERTa 미세조정 · {COND_NAME[cond]}  |  epoch {hp['epochs']}, lr {hp['learning_rate']}, "
                 f"batch {hp['batch_size']}, warmup {hp['warmup_ratio']}, weight decay {hp['weight_decay']}",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_confidence(probs, y, cond, path):
    """정답 클래스에 준 확률의 분포를 실제 생성 모델별로 겹쳐 그린다."""
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    bins = np.linspace(0, 1, 21)
    for k, m in enumerate(MODELS):
        ax.hist(probs[y == k, k], bins=bins, histtype="step", lw=2, color=COLORS[m], label=m)
    ax.set_xlabel("정답 모델에 준 확률", fontsize=8.5, color=INK_2)
    ax.set_ylabel("test 답변 수", fontsize=8.5, color=INK_2)
    ax.set_title(f"예측 확신도 분포 · {COND_NAME[cond]} (1에 몰릴수록 확신)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(ax)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def save_all_predictions(trainer, ds, splits, cond):
    """val/test 답변별 예측 확률을 CSV로 저장한다."""
    for s in ("val", "test"):
        probs = torch.softmax(torch.tensor(trainer.predict(ds[s]).predictions), -1).numpy()
        _, y, ids = splits[s]
        save_predictions(ids, y, probs, OUT / f"predictions_{cond}_{s}.csv")
        print(f"[{cond}] {s} 정확도 {(probs.argmax(-1) == y).mean():.3f}, 예측 저장: predictions_{cond}_{s}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", default="raw", choices=["raw", "norm"])
    parser.add_argument("--plot-only", action="store_true", help="저장된 metrics로 학습 곡선만 다시 그린다")
    parser.add_argument("--predict-only", action="store_true", help="저장된 best 모델로 val/test 예측만 다시 저장한다")
    cli = parser.parse_args()
    cond = cli.cond
    setup_plot()
    if cli.plot_only:
        m = json.loads((OUT / f"metrics_{cond}.json").read_text(encoding="utf-8"))
        plot_training(m["log_history"], m["best_step"], m["hyperparameters"], cond, OUT / f"training_{cond}.png")
        print(f"다시 그림: {OUT / f'training_{cond}.png'}")
        return
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load_dataset()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    splits = {s: get_split(rows, s, cond) for s in ("train", "val", "test")}
    ds = {s: to_dataset(x, y, tok) for s, (x, y, _) in splits.items()}
    run_dir = CKPT_DIR / f"roberta_{cond}"
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=PAD_MULTIPLE)

    if cli.predict_only:
        model = AutoModelForSequenceClassification.from_pretrained(run_dir / "best")
        trainer = Trainer(model=model, data_collator=collator, processing_class=tok,
                          args=TrainingArguments(output_dir=str(run_dir / "predict"), report_to="none",
                                                 per_device_eval_batch_size=BATCH_SIZE * 2,
                                                 dataloader_pin_memory=torch.cuda.is_available()))
        save_all_predictions(trainer, ds, splits, cond)
        return
    n_trunc = sum(len(tok(t)["input_ids"]) > MAX_LEN for s in splits.values() for t in s[0])
    print(f"[{cond}] train {len(ds['train'])}, val {len(ds['val'])}, test {len(ds['test'])}, "
          f"{MAX_LEN}토큰 초과로 잘린 답변 {n_trunc}개")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(MODELS),
        id2label=dict(enumerate(MODELS)), label2id={m: i for i, m in enumerate(MODELS)})
    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,     # 중간 계산값을 저장하지 않고 다시 계산해 메모리를 아낀다
        warmup_steps=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=1,
        load_best_model_at_end=True,     # val 정확도가 가장 높은 체크포인트로 test를 본다
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=collator,
                      processing_class=tok, compute_metrics=compute_metrics, callbacks=[EmptyMpsCache()])
    trainer.train()

    best_step = int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1])
    out = trainer.predict(ds["test"])
    probs = torch.softmax(torch.tensor(out.predictions), -1).numpy()
    pred = probs.argmax(-1)
    _, y_te, g_te = splits["test"]
    acc = float((pred == y_te).mean())
    lo, hi = bootstrap_ci(y_te, pred, g_te)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=range(len(MODELS)), zero_division=0)
    print(f"[{cond}] best step {best_step}, test {acc:.3f} (95% CI {lo:.3f}-{hi:.3f})")

    hp = {"model": MODEL_NAME, "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE * GRAD_ACCUM, "grad_accum": GRAD_ACCUM,
          "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY, "max_len": MAX_LEN,
          "scheduler": "linear", "eval_steps": EVAL_STEPS, "seed": SEED,
          "steps_per_epoch": int(np.ceil(len(ds["train"]) / (BATCH_SIZE * GRAD_ACCUM))),
          "total_steps": trainer.state.global_step}
    log = trainer.state.log_history
    plot_training(log, best_step, hp, cond, OUT / f"training_{cond}.png")
    plot_confusion(y_te, pred, f"KLUE-RoBERTa · {COND_NAME[cond]}: test 정확도 {acc:.1%}", OUT / f"confusion_{cond}.png")
    plot_confidence(probs, y_te, cond, OUT / f"confidence_{cond}.png")

    metrics = {
        "hyperparameters": hp, "n_truncated": int(n_trunc), "best_step": best_step,
        "best_val_acc": float(trainer.state.best_metric),
        "test_acc": acc, "test_ci95": [lo, hi],
        "per_class": {m: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
                      for i, m in enumerate(MODELS)},
        "log_history": log,
    }
    (OUT / f"metrics_{cond}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    save_all_predictions(trainer, ds, splits, cond)

    # 가장 좋은 모델만 남기고 중간 체크포인트는 지운다.
    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    for d in run_dir.glob("checkpoint-*"):
        shutil.rmtree(d)
    print(f"저장: {OUT}, 모델: {best_dir}")


if __name__ == "__main__":
    main()
