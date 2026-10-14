"""학생(교사 × seed)과 학습 전 원본 모델이 세 교사의 검증 답변에 대해 내는 loss를 잰다.

학생이 자기 교사의 스타일을 흡수했다면, 자기 교사의 검증 답변에서 loss가 가장 낮아야 한다
(예: GPT 학생은 GPT 답변 < Claude·Gemini 답변). 검증 답변 199문항은 학습에 쓰지 않은 질문이다.

사용법
  python sft/cross_loss.py --teacher gpt --seed 42   학생 한 명 → sft/runs/gpt_seed42/cross_loss.csv
  python sft/cross_loss.py --base                    원본 모델 → sft/runs/base/cross_loss.csv
  python sft/cross_loss.py --merge                   위 파일들을 합쳐 sft/runs/cross_loss.csv
  이미 계산한 파일은 건너뛴다. 다시 계산하려면 --force.

CSV 열: student_teacher, seed, eval_teacher, loss(답변 토큰 평균 교차 엔트로피), perplexity, n_answer_tokens
student_teacher=base는 학습 전 원본 Qwen3-0.6B-Base다.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from huggingface_hub import snapshot_download  # noqa: E402
from train import CKPT_DIR, DATA_DIR, MODEL_DIR, MODEL_NAME, RUNS_DIR, collate, tokenize  # noqa: E402

TEACHERS = ["gpt", "claude", "gemini"]
BATCH_SIZE = 8
HEADER = "student_teacher,seed,eval_teacher,loss,perplexity,n_answer_tokens"


def answer_loss(model, tok, rows):
    """답변 토큰 전체에 대한 평균 교차 엔트로피(토큰 가중 평균)."""
    items = [tokenize(r, tok) for r in rows]
    total, count = 0.0, 0
    for start in range(0, len(items), BATCH_SIZE):
        batch = collate(items[start:start + BATCH_SIZE], tok.pad_token_id)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
        labels = batch["labels"][:, 1:]
        losses = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels, ignore_index=-100,
                                                   reduction="sum")
        total += losses.item()
        count += (labels != -100).sum().item()
    return total / count, count


def compute(name, seed, model_path, out_path, limit):
    val = {t: json.loads((DATA_DIR / f"{t}_val.json").read_text(encoding="utf-8"))[:limit] for t in TEACHERS}
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
    lines = [HEADER]
    for t in TEACHERS:
        loss, n = answer_loss(model, tok, val[t])
        lines.append(f"{name},{seed},{t},{loss:.6f},{math.exp(loss):.4f},{n}")
        print(f"{name} seed {seed} → {t} 검증 답변: loss {loss:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", help="gpt | claude | gemini (코드 점검 때는 smoke)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base", action="store_true", help="학습 전 원본 모델을 계산한다")
    parser.add_argument("--merge", action="store_true", help="학생별 결과를 sft/runs/cross_loss.csv로 합친다")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, help="코드 점검용: 검증 답변 앞의 N개만 쓴다")
    cli = parser.parse_args()

    if cli.merge:
        files = sorted(RUNS_DIR.glob("*/cross_loss.csv"))
        rows = [line for f in files for line in f.read_text(encoding="utf-8").splitlines()[1:] if line]
        (RUNS_DIR / "cross_loss.csv").write_text("\n".join([HEADER] + rows) + "\n", encoding="utf-8")
        print(f"합침: {len(files)}개 파일 → {RUNS_DIR / 'cross_loss.csv'} ({len(rows)}행)")
        return

    if cli.base:
        name, seed, model_path, out_path = "base", 0, MODEL_DIR, RUNS_DIR / "base" / "cross_loss.csv"
        if not (MODEL_DIR / "config.json").exists():   # 첫 학습 전에 실행해도 되도록 원본 모델을 먼저 받는다
            print(f"원본 모델 다운로드: {MODEL_NAME} -> {MODEL_DIR}")
            snapshot_download(MODEL_NAME, local_dir=MODEL_DIR)
    else:
        if not cli.teacher or cli.seed is None:
            raise SystemExit("--teacher와 --seed를 함께 주거나 --base / --merge를 쓰세요.")
        name, seed = cli.teacher, cli.seed
        model_path = CKPT_DIR / ("smoke" if name == "smoke" else name) / f"seed{seed}"
        out_path = RUNS_DIR / (f"smoke_{name}" if name == "smoke" else f"{name}_seed{seed}") / "cross_loss.csv"
    if out_path.exists() and not cli.force:
        print(f"[건너뜀] {out_path}: 이미 계산됨")
        return
    if not (Path(model_path) / "config.json").exists():
        raise SystemExit(f"모델이 없습니다: {model_path}")
    compute(name, seed, model_path, out_path, cli.limit)


if __name__ == "__main__":
    main()
