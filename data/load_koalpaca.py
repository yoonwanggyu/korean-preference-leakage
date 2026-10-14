"""KoAlpaca-v1.1a에서 instruction 1,200개를 뽑아 JSON으로 저장한다.

저장된 instruction은 이후 GPT/Claude/Gemini 답변 생성의 공통 입력으로 쓴다.
id는 원본 데이터셋의 행 인덱스이므로, 평가 질문을 분리할 때 이 id를 제외하면 된다.
category는 원본 url(네이버 지식iN)의 d1id 대분류, answer는 원본 output이다.
"""

import json
import random
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from datasets import load_dataset

DATASET_NAME = "beomi/KoAlpaca-v1.1a"
N_SAMPLES = 1200
SEED = 42
OUT_PATH = Path(__file__).parent / "koalpaca_instructions_1200.json"

# 네이버 지식iN 대분류(d1id). 분류별 예시 질문을 보고 추정한 이름이다.
CATEGORIES = {
    "1": "컴퓨터통신", "2": "게임", "3": "엔터테인먼트/예술", "4": "경제",
    "5": "쇼핑", "6": "사회/정치", "7": "건강", "8": "생활", "9": "여행",
    "10": "스포츠/레저", "11": "교육/학문", "12": "지역", "13": "쥬니어", "20": "고민Q&A",
}


def to_record(i, row):
    d1id = parse_qs(urlparse(row["url"] or "").query).get("d1id", [""])[0]
    return {
        "id": i,
        "category": CATEGORIES.get(d1id, "기타"),
        "instruction": row["instruction"].strip(),
        "answer": (row["output"] or "").strip(),
    }


def main():
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"원본 크기: {len(ds)}")

    # 빈 instruction과 중복 instruction을 제거한 뒤 샘플링한다.
    seen = set()
    candidates = []
    for i, row in enumerate(ds):
        text = (row["instruction"] or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        candidates.append(to_record(i, row))
    print(f"정제 후 크기: {len(candidates)}")

    random.seed(SEED)
    samples = random.sample(candidates, N_SAMPLES)
    samples.sort(key=lambda x: x["id"])

    OUT_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(samples)}개 저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
