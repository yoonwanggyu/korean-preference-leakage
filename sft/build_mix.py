"""8.1 데이터 비율 실험: Gemini 답의 비율을 바꾼 혼합 학습 데이터를 만든다.

기존 학습 질문 1,000개(id)를 그대로 쓰고, 질문마다 세 교사 중 한 명의 답만 고른다(같은 질문이 두 번 들어가지 않음).
  - Gemini 답을 쓸 질문은 SEED로 섞은 순서의 앞에서부터 고른다 → 비율끼리 포함 관계(25% ⊂ 1/3 ⊂ 50% ⊂ 75%).
  - 나머지 질문은 GPT 답과 Claude 답으로 반씩 채운다(같은 SEED로 정한 두 번째 순서에서 앞 절반 GPT, 뒤 절반 Claude).
  - 같은 비율의 seed 42·43·44 학생은 모두 같은 데이터로 학습한다(seed는 학습 무작위성만 바꾼다).
검증 세트는 기존 Gemini 학생과 같은 Gemini 검증 답 199개를 쓴다(학습 중 val loss용).

사용법: python sft/build_mix.py
출력(sft/data/):
  gemini_mix{75,50,33,25,00}_train_1000.json   [{id, category, instruction, output, source}]  source = 답을 쓴 교사
  gemini_mix{75,50,33,25,00}_val.json          gemini_val.json과 같은 내용(train.py가 이름으로 찾는다)
  mix_summary.json                              비율별 교사 답 개수, Gemini 질문 id, 학습 토큰 수, 포함 관계 검증
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

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models" / os.environ.get("STUDENT_TOKENIZER", "Qwen/Qwen3-0.6B-Base").split("/")[-1]
TEACHERS = ["gpt", "claude", "gemini"]
SEED = 42
N_TRAIN = 1000
# 이름 → Gemini 답 개수 (균등 혼합은 334 / 333 / 333)
MIXES = {"gemini_mix75": 750, "gemini_mix50": 500, "gemini_mix33": 334, "gemini_mix25": 250, "gemini_mix00": 0}


def main():
    data = {t: {x["id"]: x for x in json.loads((DATA_DIR / f"{t}_train_{N_TRAIN}.json").read_text(encoding="utf-8"))}
            for t in TEACHERS}
    ids = sorted(data["gemini"])
    assert len(ids) == N_TRAIN and all(sorted(data[t]) == ids for t in TEACHERS), "세 교사의 학습 질문 id가 다르다"

    rng = random.Random(SEED)
    gemini_order = ids[:]
    rng.shuffle(gemini_order)          # 앞에서부터 Gemini 질문으로 쓴다 → 비율끼리 포함 관계
    rest_order = ids[:]
    rng.shuffle(rest_order)            # Gemini가 아닌 질문을 GPT·Claude로 나누는 순서

    tok = AutoTokenizer.from_pretrained(MODEL_DIR if (MODEL_DIR / "config.json").exists() else "Qwen/Qwen3-0.6B-Base")
    val = (DATA_DIR / "gemini_val.json").read_text(encoding="utf-8")
    summary = {"seed": SEED, "n_train": N_TRAIN, "mixes": {}}
    gem_sets = {}
    for name, n_gem in MIXES.items():
        gem = set(gemini_order[:n_gem])
        rest = [i for i in rest_order if i not in gem]
        gpt = set(rest[:len(rest) // 2])
        source = {i: "gemini" if i in gem else ("gpt" if i in gpt else "claude") for i in ids}
        rows = [{"id": i, "category": data[source[i]][i]["category"], "instruction": data[source[i]][i]["instruction"],
                 "output": data[source[i]][i]["output"], "source": source[i]} for i in ids]
        n_tokens = sum(len(tok(build_prompt(r["instruction"]) + r["output"] + END_TOKEN)["input_ids"]) for r in rows)
        (DATA_DIR / f"{name}_train_{N_TRAIN}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (DATA_DIR / f"{name}_val.json").write_text(val, encoding="utf-8")
        counts = {t: sum(s == t for s in source.values()) for t in TEACHERS}
        summary["mixes"][name] = {"gemini_ratio": n_gem / N_TRAIN, "counts": counts, "train_total_tokens": n_tokens,
                                  "gemini_ids": sorted(gem)}
        gem_sets[name] = gem
        print(f"{name}: GPT {counts['gpt']} · Claude {counts['claude']} · Gemini {counts['gemini']}, 학습 토큰 {n_tokens:,}")

    order = sorted(MIXES, key=MIXES.get)
    summary["nested_check"] = all(gem_sets[a] <= gem_sets[b] for a, b in zip(order, order[1:]))
    print("포함 관계(0% ⊂ 25% ⊂ 1/3 ⊂ 50% ⊂ 75%):", summary["nested_check"])
    (DATA_DIR / "mix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"저장: {DATA_DIR}/gemini_mix*_train_{N_TRAIN}.json, gemini_mix*_val.json, mix_summary.json")


if __name__ == "__main__":
    main()
