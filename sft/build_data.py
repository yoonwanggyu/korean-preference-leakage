"""세 교사(GPT/Claude/Gemini)의 생성 결과에서 공통 학습 데이터 1,000개를 만든다.

1. 세 파일을 id로 맞추고, 한 모델이라도 답이 null인 질문을 뺀다.
2. 학습 형식(프롬프트 + 답변 + <|im_end|>)을 Qwen 토크나이저로 세어, 한 모델이라도 STUDENT_MAX_LEN을 넘으면 뺀다.
3. 남은 질문에서 SEED로 1,000개 id를 뽑아 세 교사 모두 같은 id로 학습 파일을 저장한다.
4. 뽑히지 않은 나머지 질문은 검증 세트로 저장한다. 학습 중 val loss 기록과 교차 loss 행렬(cross_loss.py)에 쓴다.
5. 뺀 id와 이유를 excluded.json에, 개수와 토큰 통계를 data_summary.json에 남긴다.
"""

import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
from prompt import END_TOKEN, build_prompt  # noqa: E402

load_dotenv(ROOT / ".env")

TOKENIZER = os.environ.get("STUDENT_TOKENIZER", "Qwen/Qwen3-0.6B-Base")
MAX_LEN = int(os.environ.get("STUDENT_MAX_LEN", 1024))
TEACHERS = ["gpt", "claude", "gemini"]
N_TRAIN = 1000
SEED = 42
GEN_DIR = ROOT / "inference" / "output"
OUT_DIR = Path(__file__).parent / "data"


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    data = {}
    for t in TEACHERS:
        items = json.loads((GEN_DIR / f"{t}_train_1200.json").read_text(encoding="utf-8"))
        data[t] = {x["id"]: x for x in items}

    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    excluded = []
    eligible = []
    n_tokens = {t: {} for t in TEACHERS}
    for i in ids:
        reasons = []
        for t in TEACHERS:
            answer = data[t][i][f"{t}_output"]
            if not answer:
                reasons.append(f"null_{t}")
                continue
            text = build_prompt(data[t][i]["instruction"]) + answer + END_TOKEN
            n = len(tok(text)["input_ids"])
            n_tokens[t][i] = n
            if n > MAX_LEN:
                reasons.append(f"too_long_{t}({n})")
        if reasons:
            excluded.append({"id": i, "reasons": reasons})
        else:
            eligible.append(i)
    print(f"공통 id {len(ids)}개 중 제외 {len(excluded)}개, 사용 가능 {len(eligible)}개")
    if len(eligible) < N_TRAIN:
        raise SystemExit(f"사용 가능한 질문이 {N_TRAIN}개보다 적습니다.")

    random.seed(SEED)
    train_ids = sorted(random.sample(eligible, N_TRAIN))
    val_ids = sorted(set(eligible) - set(train_ids))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"n_common": len(ids), "n_excluded": len(excluded), "n_eligible": len(eligible),
               "n_train": len(train_ids), "n_val": len(val_ids), "seed": SEED, "max_len": MAX_LEN,
               "tokenizer": TOKENIZER, "tokens": {}}
    for t in TEACHERS:
        for split, split_ids in (("train", train_ids), ("val", val_ids)):
            rows = [{
                "id": i,
                "category": data[t][i]["category"],
                "instruction": data[t][i]["instruction"],
                "output": data[t][i][f"{t}_output"],
            } for i in split_ids]
            name = f"{t}_train_{N_TRAIN}.json" if split == "train" else f"{t}_val.json"
            (OUT_DIR / name).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"저장: {OUT_DIR / name} ({len(rows)}개)")
        lens = sorted(n_tokens[t][i] for i in train_ids)
        summary["tokens"][t] = {"train_total": sum(lens), "train_median": lens[len(lens) // 2], "train_max": lens[-1]}
    (OUT_DIR / "train_ids.json").write_text(json.dumps(train_ids), encoding="utf-8")
    (OUT_DIR / "val_ids.json").write_text(json.dumps(val_ids), encoding="utf-8")
    (OUT_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "excluded.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    for e in excluded:
        print(f"[제외] id={e['id']}: {', '.join(e['reasons'])}")


if __name__ == "__main__":
    main()
