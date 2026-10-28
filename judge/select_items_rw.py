"""8.2 문체 제거 실험의 평가 항목을 만든다: 재작성 데이터로 학습한 학생 3명(gpt_rw·claude_rw·gemini_rw)을 같은 방식으로 채점.

1. seed 42·43·44마다 그 seed의 재작성 학생 3명이 모두 끝맺은 질문을 찾는다
   (select_items.py와 같은 완전성 규칙: finished, 1,024토큰 미만, 빈 답 아님).
2. 원래 학생 3명(GPT·Claude·Gemini 100%)의 채점이 이미 있는 질문을 먼저 넣는다. 원래 학생과 같은 질문에서
   비교할 수 있게 하기 위해서다. 순서: 본 실험 평가 질문(judge/data/selection.json) → 8.1에서 새로 채점한 질문
   (judge/data/mix/selection.json의 seed별 질문) → 나머지. 한 단계에서 모자란 만큼만 다음 단계에서 SAMPLE_SEED로 추첨한다.
3. 묶음 = seed × 질문. order에는 원래 교사 이름(gpt·claude·gemini)을 쓰고 condition = "rw"로 표시한다.
   순서 설계는 select_items.py와 같다(6가지 순열을 돌아가며 기본 순서, 순환 3개 = 라틴 방격).

사용법: python judge/select_items_rw.py
출력: judge/data/rw/eval_items.json  [{item_id, group_id, condition, question_id, category, seed, rotation, order,
                                      instruction, answers, n_tokens}]
      judge/data/rw/selection.json   seed별 후보·선택 질문, 원래 학생 채점이 있는 질문 수, 항목 수
"""

import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).parent / "data" / "rw"
OLD_SELECTION = Path(__file__).parent / "data" / "selection.json"
MIX_SELECTION = Path(__file__).parent / "data" / "mix" / "selection.json"
SEEDS = [42, 43, 44]
TEACHERS = ["gpt", "claude", "gemini"]
N_QUESTIONS = 100
N_ROTATIONS = 3
MAX_NEW_TOKENS = 1024
SAMPLE_SEED = 42


def complete(r):
    return r["finished"] and r["n_new_tokens"] < MAX_NEW_TOKENS and r["output"].strip() != ""


def main():
    old = set(json.loads(OLD_SELECTION.read_text(encoding="utf-8"))["selected_question_ids"])
    mix = json.loads(MIX_SELECTION.read_text(encoding="utf-8"))["selected_question_ids"]
    answers = {(t, s): {r["id"]: r for r in json.loads((ROOT / "inference" / "output_rw" / f"{t}_rw_seed{s}_eval400.json")
                                                       .read_text(encoding="utf-8"))}
               for t in TEACHERS for s in SEEDS}

    rng = random.Random(SAMPLE_SEED)
    chosen, info = {}, {}
    for s in SEEDS:
        cand = sorted(q for q in answers[("gpt", s)] if all(complete(answers[(t, s)][q]) for t in TEACHERS))
        tiers = [[q for q in cand if q in old],
                 [q for q in cand if q not in old and q in set(mix[str(s)])],
                 [q for q in cand if q not in old and q not in set(mix[str(s)])]]
        pick = []
        for tier in tiers:
            need = N_QUESTIONS - len(pick)
            pick += tier if len(tier) <= need else rng.sample(tier, need)
        chosen[s] = sorted(pick)
        n_old = sum(q in old for q in pick)
        n_mix = sum(q in tiers[1] for q in pick)
        info[s] = {"n_candidates": len(cand), "n_main": n_old, "n_mix": n_mix, "n_new": N_QUESTIONS - n_old - n_mix}
        print(f"seed {s}: 후보 {len(cand)}개 → 본 실험 질문 {n_old} + 8.1 질문 {n_mix} + 새 질문 {N_QUESTIONS - n_old - n_mix}")

    perms = list(itertools.permutations(TEACHERS))
    items, g = [], 0
    for s in SEEDS:
        for q in chosen[s]:
            base = list(perms[g % len(perms)])
            for rot in range(N_ROTATIONS):
                order = base[rot:] + base[:rot]
                rs = [answers[(t, s)][q] for t in order]
                items.append({"group_id": g, "condition": "rw", "question_id": q, "category": rs[0]["category"],
                              "seed": s, "rotation": rot, "order": order, "instruction": rs[0]["instruction"],
                              "answers": [r["output"] for r in rs], "n_tokens": [r["n_new_tokens"] for r in rs]})
            g += 1
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item_id"] = i

    balance = {t: [sum(it["order"][p] == t for it in items) for p in range(3)] for t in TEACHERS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    selection = {"rule": "seed별로 재작성 학생 3명 모두 완전한 질문 중 본 실험 질문 → 8.1 질문 → 나머지 순으로 100개",
                 "sample_seed": SAMPLE_SEED, "per_seed": info, "selected_question_ids": {s: chosen[s] for s in SEEDS},
                 "n_groups": g, "n_items": len(items), "position_balance": balance}
    (OUT_DIR / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"묶음 {g}개 × 순서 {N_ROTATIONS} = 평가 항목 {len(items)}개 (평가자 3명이면 {len(items) * 3:,}회)")
    print("위치 균형:", balance)


if __name__ == "__main__":
    main()
