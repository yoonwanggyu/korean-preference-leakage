"""KoAlpaca-v1.1a에서 학습용 1,200개와 겹치지 않는 평가 질문 400개를 뽑아 JSON으로 저장한다.

학습용 instruction과 id·텍스트가 모두 겹치지 않는 행만 후보로 쓴다.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from load_koalpaca import DATASET_NAME, to_record

N_SAMPLES = 400
SEED = 42
TRAIN_PATH = Path(__file__).parent / "koalpaca_instructions_1200.json"
OUT_PATH = Path(__file__).parent / "koalpaca_eval_400.json"


def main():
    train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    train_ids = {x["id"] for x in train}
    seen = {x["instruction"] for x in train}

    ds = load_dataset(DATASET_NAME, split="train")
    candidates = []
    for i, row in enumerate(ds):
        text = (row["instruction"] or "").strip()
        if i in train_ids or not text or text in seen:
            continue
        seen.add(text)
        candidates.append(to_record(i, row))
    print(f"후보 크기: {len(candidates)}")

    random.seed(SEED)
    samples = random.sample(candidates, N_SAMPLES)
    samples.sort(key=lambda x: x["id"])

    OUT_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(samples)}개 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
