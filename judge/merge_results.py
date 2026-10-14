"""네 평가자의 listwise 채점 기록을 학생 응답 JSON에 붙여 하나로 합친다.

inference/output_2nd/{teacher}_seed{seed}_eval400.json의 내용(질문, 참조 답, 학생 응답, 길이, 잘림 여부 등)은
그대로 두고, 평가 대상 100문항에는 "in_judge_set": true와 "judges" 키를 추가한다. 원본 파일은 수정하지 않고
judge/output/judged/에 새로 저장한다. 한 묶음(같은 질문·같은 seed의 학생 3명)은 제시 순서를 바꿔 여러 번 평가되므로,
학생마다 순서별 순위와 그때의 위치를 모두 남기고 평균 순위를 함께 적는다.

  "judges": {
    "gpt": {
      "model": "...",
      "factuality": {"ranks": [1, 2, 1], "positions": ["A", "B", "C"], "mean_rank": 1.33,
                     "reasons": ["순서 1의 근거", "순서 2의 근거", "순서 3의 근거"]},
      ... 4개 기준,
      "overall": {"ranks": [1, 1, 2], "positions": [...], "mean_rank": 1.33, "reasons": [...]},   # 최종 순위
      "criteria_mean_rank": 1.75   # 네 기준 순위의 평균 (1이 가장 좋음)
    },
    "claude": {...}, "gemini": {...}, "hyperclova": {...}
  }
근거(reasons)는 세 답변을 비교한 문장이라 같은 묶음의 학생 3명에게 같은 문장이 붙는다.
아직 채점되지 않은 평가자는 빠진다. 다시 실행하면 최신 채점 기록으로 다시 만든다.

사용법: python judge/merge_results.py           (본 채점)
        python judge/merge_results.py --pilot   (파일럿: judge/output/pilot/의 seed 42만)
출력: judge/output/judged/{teacher}_seed{seed}_eval400.json (15개)
      judge/output/ranks_long.csv  분석용 긴 표: 평가자 × 항목(묶음 × 순서) × 학생 × 기준 한 줄
"""

import argparse
import json
from pathlib import Path

from test_prompt import CRITERIA_KEYS, LABELS, OUTPUT_KEYS

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "inference" / "output_2nd"
RAW_DIR = Path(__file__).parent / "output" / "raw"
OUT_DIR = Path(__file__).parent / "output" / "judged"
PILOT_DIR = Path(__file__).parent / "output" / "pilot"
SELECTION = Path(__file__).parent / "data" / "selection.json"
TEACHERS = ["gpt", "claude", "gemini"]
SEEDS = [42, 43, 44, 45, 46]
JUDGES = ["gpt", "claude", "gemini", "hyperclova"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    cli = parser.parse_args()
    raw_dir, out_dir, seeds = RAW_DIR, OUT_DIR, SEEDS
    selected = set(json.loads(SELECTION.read_text(encoding="utf-8"))["selected_question_ids"])
    if cli.pilot:
        raw_dir, out_dir, seeds = PILOT_DIR / "raw", PILOT_DIR / "judged", [42]
        selected = set(sorted(selected)[:10])

    raw = {}   # (judge, seed, question_id) → 그 묶음의 채점 기록들(순서별)
    for judge in JUDGES:
        path = raw_dir / f"{judge}.jsonl"
        if not path.exists():
            print(f"[없음] {judge}: 아직 채점 기록이 없습니다")
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for r in rows:
            raw.setdefault((judge, r["seed"], r["question_id"]), []).append(r)
        print(f"{judge}: {len(rows)}개 채점 기록")

    out_dir.mkdir(parents=True, exist_ok=True)
    long_lines = ["judge,judge_model,teacher,seed,question_id,group_id,rotation,position,n_tokens,criterion,rank"]
    for t in TEACHERS:
        for s in seeds:
            rows = json.loads((SRC / f"{t}_seed{s}_eval400.json").read_text(encoding="utf-8"))
            for row in rows:
                row["in_judge_set"] = row["id"] in selected
                if not row["in_judge_set"]:
                    continue
                row["judges"] = {}
                for judge in JUDGES:
                    records = sorted(raw.get((judge, s, row["id"]), []), key=lambda r: r["rotation"])
                    if not records:
                        continue
                    entry = {"model": records[0]["judge_model"]}
                    for k in OUTPUT_KEYS:
                        ranks = [r[k]["ranks"][t] for r in records]
                        positions = [LABELS[r["order"].index(t)] for r in records]
                        entry[k] = {"ranks": ranks, "positions": positions,
                                    "mean_rank": round(sum(ranks) / len(ranks), 4),
                                    "reasons": [r[k]["reason"] for r in records]}
                        for r, rank, pos in zip(records, ranks, positions):
                            long_lines.append(f"{judge},{r['judge_model']},{t},{s},{row['id']},{r['group_id']},"
                                              f"{r['rotation']},{pos},{row['n_new_tokens']},{k},{rank}")
                    entry["criteria_mean_rank"] = round(sum(entry[k]["mean_rank"] for k in CRITERIA_KEYS) / len(CRITERIA_KEYS), 4)
                    row["judges"][judge] = entry
            (out_dir / f"{t}_seed{s}_eval400.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                                               encoding="utf-8")
    (out_dir.parent / "ranks_long.csv").write_text("\n".join(long_lines) + "\n", encoding="utf-8")
    print(f"저장: {out_dir} (학생 파일 {len(TEACHERS) * len(seeds)}개), ranks_long.csv ({len(long_lines) - 1}행, "
          f"완료 시 1,500 항목 × 학생 3 × (기준 4 + 최종 1) × 평가자 4 = 90,000행)")


if __name__ == "__main__":
    main()
