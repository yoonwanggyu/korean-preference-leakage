"""LLM-as-a-Judge(listwise)로 평가할 묶음과 항목을 만든다.

1. 학생 15명(교사 3 × seed 5)의 2차 응답(inference/output_2nd/)을 읽는다.
2. 15명 모두의 답이 잘리지 않고(finished=true, 1024토큰 미만) 비어 있지 않은 "공통 완전 질문"(144개)을 찾고 교차 검증한다.
3. 공통 완전 질문에서 추첨용 난수 SAMPLE_SEED로 100개를 뽑는다(학생 모델의 seed와 무관).
4. 묶음(group) = 질문 1개 × 학생 seed 1개. 같은 seed의 GPT·Claude·Gemini 학생 답 3개를 한 번에 비교한다
   (비교는 항상 같은 seed끼리). 100 질문 × 5 seed = 500 묶음.
5. 위치 편향 통제: 묶음마다 기본 순서를 6가지 순열에서 돌아가며 정하고(전체적으로 균형), 그 순서를 순환시킨
   N_ROTATIONS개의 제시 순서로 평가한다. N_ROTATIONS=3이면 묶음 안에서 각 학생이 1·2·3번 위치에 정확히 한 번씩 나온다
   (라틴 방격). 평가 항목(item) = 묶음 × 제시 순서.

사용법: python judge/select_items.py
출력: judge/data/eval_items.json   [{item_id, group_id, question_id, category, seed, rotation, order, instruction,
                                    answers, n_tokens}]  order는 A·B·C 위치에 놓인 교사 순서
      judge/data/selection.json    선정 기준, 공통 완전 질문 id, 선택한 100개 id, 검증 결과, 순서 배정 방식
"""

import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "inference" / "output_2nd"
EVAL_PATH = ROOT / "data" / "koalpaca_eval_400.json"
OUT_DIR = Path(__file__).parent / "data"
TEACHERS = ["gpt", "claude", "gemini"]
SEEDS = [42, 43, 44, 45, 46]
N_QUESTIONS = 100
N_ROTATIONS = 3          # 묶음 하나를 몇 가지 순서로 평가할지 (3: 라틴 방격으로 위치 완전 균형, 1: 묶음 사이에서만 균형)
MAX_NEW_TOKENS = 1024    # 2차 생성의 최대 길이. 이 길이에 닿은 답은 잘린 답이다.
SAMPLE_SEED = 42         # 144개 중 100개를 추첨하고 평가 순서를 섞을 때 쓰는 난수 seed (학생 모델의 seed와 무관)


def complete(r):
    return r["finished"] and r["n_new_tokens"] < MAX_NEW_TOKENS and r["output"].strip() != ""


def main():
    answers = {(t, s): {r["id"]: r for r in json.loads((SRC / f"{t}_seed{s}_eval400.json").read_text(encoding="utf-8"))}
               for t in TEACHERS for s in SEEDS}
    question_ids = [x["id"] for x in json.loads(EVAL_PATH.read_text(encoding="utf-8"))]

    common = [q for q in question_ids if all(complete(a[q]) for a in answers.values())]
    excluded = [q for q in question_ids if q not in set(common)]
    assert all(complete(a[q]) for q in common for a in answers.values())
    assert all(any(not complete(a[q]) for a in answers.values()) for q in excluded)
    print(f"평가 질문 {len(question_ids)}개 중 15명 모두 완전한 공통 질문: {len(common)}개 (교차 검증 통과)")

    rng = random.Random(SAMPLE_SEED)
    chosen = sorted(rng.sample(common, N_QUESTIONS))

    perms = list(itertools.permutations(TEACHERS))   # 6가지 기본 순서
    items, groups = [], []
    for g, (s, q) in enumerate((s, q) for s in SEEDS for q in chosen):
        base = list(perms[g % len(perms)])
        groups.append({"group_id": g, "seed": s, "question_id": q})
        for rot in range(N_ROTATIONS):
            order = base[rot:] + base[:rot]          # 순환: 각 교사가 각 위치에 한 번씩
            r0 = answers[(order[0], s)][q]
            items.append({"group_id": g, "question_id": q, "category": r0["category"], "seed": s, "rotation": rot,
                          "order": order, "instruction": r0["instruction"],
                          "answers": [answers[(t, s)][q]["output"] for t in order],
                          "n_tokens": [answers[(t, s)][q]["n_new_tokens"] for t in order]})
    rng.shuffle(items)   # 같은 묶음의 순서 3개가 연달아 평가되지 않도록 섞는다
    for i, item in enumerate(items):
        item["item_id"] = i

    # 위치 균형 확인: 교사마다 각 위치(A·B·C)에 나온 횟수
    balance = {t: [sum(it["order"][p] == t for it in items) for p in range(3)] for t in TEACHERS}
    print("교사별 위치 A·B·C 등장 횟수:", balance)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    selection = {
        "source": "inference/output_2nd",
        "rule": "15명 모두 finished=true, n_new_tokens<1024, 빈 답 아님",
        "n_eval_questions": len(question_ids),
        "complete_answers_per_student": {f"{t}_seed{s}": sum(complete(r) for r in a.values())
                                         for (t, s), a in answers.items()},
        "n_common": len(common), "common_question_ids": common,
        "n_selected": N_QUESTIONS, "sample_seed": SAMPLE_SEED, "selected_question_ids": chosen,
        "format": "listwise: 같은 질문·같은 seed의 학생 3명 답을 한 번에 제시하고 기준별 1~3위",
        "n_groups": len(groups), "n_rotations": N_ROTATIONS, "n_items": len(items),
        "order_design": "묶음마다 6가지 순열을 돌아가며 기본 순서로 쓰고, 그 순서를 순환시켜 N_ROTATIONS번 제시",
        "position_balance": balance, "verified": True,
    }
    (OUT_DIR / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_DIR / 'eval_items.json'} (묶음 {len(groups)}개 × 순서 {N_ROTATIONS}개 = 평가 항목 {len(items)}개)")


if __name__ == "__main__":
    main()
