"""8.1 데이터 비율 실험의 채점 기록을 분석용 긴 표 하나로 합친다.

새 채점(judge/output/mix/raw/{judge}.jsonl)과, Gemini 100% 조건에서 재사용하는 본 실험 채점
(judge/output/raw/{judge}.jsonl 중 judge/data/mix/selection.json의 reused_from_main 묶음)을 합친다.
teacher 열은 묶음 안의 역할이다: gpt·claude = 순수 학생, gemini = 그 조건의 Gemini 계열 학생(condition 열이 비율).
묶음 id는 새 채점은 "m{번호}", 재사용은 "o{번호}"로 적어 겹치지 않게 한다.

사용법: python judge/merge_mix.py
출력: judge/output/mix/ranks_long.csv
  judge,judge_model,condition,ratio,teacher,seed,question_id,group_id,rotation,position,n_tokens,criterion,rank
"""

import json
from pathlib import Path

from test_prompt import LABELS, OUTPUT_KEYS

ROOT = Path(__file__).resolve().parent.parent
MIX = Path(__file__).parent / "output" / "mix"
MAIN_RAW = Path(__file__).parent / "output" / "raw"
SELECTION = Path(__file__).parent / "data" / "mix" / "selection.json"
JUDGES = ["gpt", "claude", "gemini"]
N_ROTATIONS = 3


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def n_tokens(name, folder, seed):
    path = ROOT / "inference" / folder / f"{name}_seed{seed}_eval400.json"
    return {r["id"]: r["n_new_tokens"] for r in json.loads(path.read_text(encoding="utf-8"))}


def main():
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    conds = sel["conditions"]
    reused = {(x["seed"], x["question_id"]) for x in sel["reused_from_main"]}
    seeds = sorted({s for s, _ in reused})
    tok = {}
    for s in seeds:
        tok[("gpt", s)], tok[("claude", s)] = n_tokens("gpt", "output_2nd", s), n_tokens("claude", "output_2nd", s)
        for c in conds:
            tok[(c, s)] = n_tokens(c, "output_2nd" if c == "gemini" else "output_mix", s)

    lines = ["judge,judge_model,condition,ratio,teacher,seed,question_id,group_id,rotation,position,n_tokens,criterion,rank"]
    for j in JUDGES:
        new = read(MIX / "raw" / f"{j}.jsonl")
        old = [r for r in read(MAIN_RAW / f"{j}.jsonl") if (r["seed"], r["question_id"]) in reused]
        for r in old:
            r["condition"], r["ratio"], r["group_id"] = "gemini", conds["gemini"], f"o{r['group_id']}"
        for r in new:
            r["group_id"] = f"m{r['group_id']}"
        print(f"{j}: 새 채점 {len(new)}개, 재사용 {len(old)}개 (재사용 묶음 {len(reused)}개 × 순서 {N_ROTATIONS} = {len(reused) * N_ROTATIONS}개 기대)")
        for r in new + old:
            who = {"gpt": "gpt", "claude": "claude", "gemini": r["condition"]}
            for k in OUTPUT_KEYS:
                for t, rank in r[k]["ranks"].items():
                    pos = LABELS[r["order"].index(t)]
                    lines.append(f"{j},{r['judge_model']},{r['condition']},{r['ratio']},{t},{r['seed']},{r['question_id']},"
                                 f"{r['group_id']},{r['rotation']},{pos},{tok[(who[t], r['seed'])][r['question_id']]},{k},{rank}")
    (MIX / "ranks_long.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {MIX / 'ranks_long.csv'} ({len(lines) - 1:,}행)")


if __name__ == "__main__":
    main()
