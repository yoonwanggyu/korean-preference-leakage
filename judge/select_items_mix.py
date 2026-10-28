"""8.1 데이터 비율 실험의 평가 항목을 만든다(poster 표 8-D의 A 방식: seed별 공통 질문 100개).

1. seed 42·43·44마다 그 seed의 학생 8명(GPT·Claude·Gemini 순수 + 혼합 5비율)이 모두 끝맺은 질문을 찾는다
   (select_items.py와 같은 완전성 규칙: finished, 1,024토큰 미만, 빈 답 아님).
2. 그중 기존 평가 질문 100개(judge/data/selection.json)에 든 질문을 먼저 넣고, 모자란 만큼 나머지에서
   SAMPLE_SEED로 추첨해 seed마다 100개를 고른다. 같은 seed 안에서는 여섯 조건이 같은 질문으로 비교된다.
3. 묶음 = 조건 × seed × 질문. 묶음의 학생 3명 = GPT 순수 · Claude 순수 · 그 조건의 Gemini 계열 학생(order에는 "gemini"로 적는다).
   순서 설계는 select_items.py와 같다(6가지 순열을 돌아가며 기본 순서, 순환 3개 = 라틴 방격).
4. Gemini 100% 조건 중 기존 평가 질문인 묶음은 본 실험의 채점(judge/output/raw/)을 재사용하므로 항목을 만들지 않는다.
   새로 들어온 질문만 항목으로 만든다.

사용법: python judge/select_items_mix.py
출력: judge/data/mix/eval_items.json  [{item_id, group_id, condition, ratio, question_id, category, seed, rotation, order,
                                       instruction, answers, n_tokens}]
      judge/data/mix/selection.json   seed별 후보·선택 질문, 재사용 묶음, 항목 수
"""

import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).parent / "data" / "mix"
OLD_SELECTION = Path(__file__).parent / "data" / "selection.json"
SEEDS = [42, 43, 44]
# 조건 이름 → (Gemini 비율, 답 폴더). gemini = 기존 Gemini 100% 학생
CONDITIONS = {"gemini_mix00": (0.0, "output_mix"), "gemini_mix25": (0.25, "output_mix"), "gemini_mix33": (0.334, "output_mix"),
              "gemini_mix50": (0.5, "output_mix"), "gemini_mix75": (0.75, "output_mix"), "gemini": (1.0, "output_2nd")}
N_QUESTIONS = 100
N_ROTATIONS = 3
MAX_NEW_TOKENS = 1024
SAMPLE_SEED = 42


def complete(r):
    return r["finished"] and r["n_new_tokens"] < MAX_NEW_TOKENS and r["output"].strip() != ""


def load(name, folder, seed):
    path = ROOT / "inference" / folder / f"{name}_seed{seed}_eval400.json"
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def main():
    old = set(json.loads(OLD_SELECTION.read_text(encoding="utf-8"))["selected_question_ids"])
    answers = {}
    for s in SEEDS:
        answers[("gpt", s)], answers[("claude", s)] = load("gpt", "output_2nd", s), load("claude", "output_2nd", s)
        for c, (_, folder) in CONDITIONS.items():
            answers[(c, s)] = load(c, folder, s)

    rng = random.Random(SAMPLE_SEED)
    chosen, info = {}, {}
    for s in SEEDS:
        students = ["gpt", "claude"] + list(CONDITIONS)
        cand = sorted(q for q in answers[("gpt", s)] if all(complete(answers[(t, s)][q]) for t in students))
        keep = [q for q in cand if q in old]
        rest = [q for q in cand if q not in old]
        chosen[s] = sorted(keep + rng.sample(rest, N_QUESTIONS - len(keep)))
        info[s] = {"n_candidates": len(cand), "n_kept_old": len(keep), "n_new": N_QUESTIONS - len(keep)}
        print(f"seed {s}: 후보 {len(cand)}개, 기존 질문 유지 {len(keep)}개 + 새로 추첨 {N_QUESTIONS - len(keep)}개")

    perms = list(itertools.permutations(["gpt", "claude", "gemini"]))
    items, reused, g = [], [], 0
    for c, (ratio, _) in CONDITIONS.items():
        for s in SEEDS:
            for q in chosen[s]:
                if c == "gemini" and q in old:
                    reused.append({"seed": s, "question_id": q})
                    continue
                base = list(perms[g % len(perms)])
                who = {"gpt": "gpt", "claude": "claude", "gemini": c}
                for rot in range(N_ROTATIONS):
                    order = base[rot:] + base[:rot]
                    rs = [answers[(who[t], s)][q] for t in order]
                    items.append({"group_id": g, "condition": c, "ratio": ratio, "question_id": q,
                                  "category": rs[0]["category"], "seed": s, "rotation": rot, "order": order,
                                  "instruction": rs[0]["instruction"], "answers": [r["output"] for r in rs],
                                  "n_tokens": [r["n_new_tokens"] for r in rs]})
                g += 1
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item_id"] = i

    balance = {t: [sum(it["order"][p] == t for it in items) for p in range(3)] for t in ["gpt", "claude", "gemini"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    selection = {"rule": "seed별로 그 seed의 학생 8명(GPT·Claude·Gemini 순수 + 혼합 5) 모두 완전한 질문 중 기존 100개 우선, 나머지 추첨",
                 "sample_seed": SAMPLE_SEED, "per_seed": info, "selected_question_ids": {s: chosen[s] for s in SEEDS},
                 "conditions": {c: r for c, (r, _) in CONDITIONS.items()},
                 "reused_from_main": reused, "n_reused_groups": len(reused), "n_new_groups": g,
                 "n_items": len(items), "position_balance": balance}
    (OUT_DIR / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"새 묶음 {g}개 × 순서 {N_ROTATIONS} = 평가 항목 {len(items)}개 (평가자 3명이면 {len(items) * 3:,}회), "
          f"기존 채점 재사용 묶음 {len(reused)}개")
    print("위치 균형:", balance)


if __name__ == "__main__":
    main()
