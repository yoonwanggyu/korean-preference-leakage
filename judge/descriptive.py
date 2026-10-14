"""기술 통계: 최종 순위(overall)와 평가 기준 4개 각각에 대해 같은 표를 만든다.

입력: judge/output/ranks_long.csv (한 순위 = 평가자 × 묶음 × 순서 × 기준의 학생 3명 순위)
용어
  1위 비율  : 학생이 나머지 두 학생보다 모두 위(1위)에 놓인 비율. 기준선 1/3.
  쌍대 승률 : 순위 하나를 쌍 3개로 풀었을 때 학생이 상대보다 위에 놓인 비율. 1위 = 2승, 2위 = 1승 1패, 3위 = 2패. 기준선 50%.
              = (1위 횟수 × 2 + 2위 횟수) / (전체 × 2) = (3 − 평균 순위) / 2
  단순 누출 : 같은 학생에 대해 [자기 계열 평가자의 지표] − [다른 두 평가자 지표의 평균]. 다른 평가자의 자기 우대가 기준선에
              섞이는 한계가 있어 최종 결론은 analyze_models.py의 모형으로 내린다.

출력: judge/output/analysis/
  desc_counts.csv     결과(criterion) × seed(all, 42~46) × 평가자 × 학생: 1·2·3위 횟수, 평균 순위, 1위 비율, 쌍대 승률
  desc_simple_leak.csv 결과 × 지표(top1, win) × 평가자: 자기 평가자 값, 다른 평가자 평균, 단순 누출, seed별 단순 누출
  desc_position.csv   결과 × 평가자 × seed: 1위가 놓인 위치 비율(A·B·C), 카이제곱 p, 순서 3가지의 1위 일치·전체 일치 묶음 수
  desc_position_rank.csv 결과 × 평가자 × 학생: A·B·C에 놓였을 때의 평균 순위
  desc_winners.csv    결과 × seed × 평가자: 질문별 승자(순서 3가지 평균 순위 최소) 수
  desc_agreement.csv  결과 × 평가자 쌍: 같은 항목에서 1위 학생이 같은 비율

사용법: python judge/descriptive.py
"""

from pathlib import Path

import pandas as pd
from scipy import stats

LONG_PATH = Path(__file__).parent / "output" / "ranks_long.csv"
OUT = Path(__file__).parent / "output" / "analysis"
JUDGES = ["gpt", "claude", "gemini"]
SEEDS = [42, 43, 44, 45, 46]
OUTCOMES = ["overall", "factuality", "comprehension", "logical_robustness", "readability"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(LONG_PATH)
    long = long[long.judge.isin(JUDGES)]
    counts, leaks, pos_rows, pos_rank, win_rows, agree = [], [], [], [], [], []

    for c in OUTCOMES:
        d = long[long.criterion == c]
        # 1) 횟수·평균 순위·1위 비율·쌍대 승률
        for s in ["all"] + SEEDS:
            ds = d if s == "all" else d[d.seed == s]
            for j in JUDGES:
                for t in JUDGES:
                    r = ds[(ds.judge == j) & (ds.teacher == t)]["rank"]
                    counts.append({"criterion": c, "seed": s, "judge": j, "teacher": t, "n": len(r),
                                   "rank1": int((r == 1).sum()), "rank2": int((r == 2).sum()), "rank3": int((r == 3).sum()),
                                   "mean_rank": r.mean(), "top1_rate": (r == 1).mean() * 100, "win_rate": (3 - r.mean()) / 2 * 100})
        cd = pd.DataFrame([x for x in counts if x["criterion"] == c])
        # 2) 단순 누출 (같은 학생: 자기 평가자 − 다른 두 평가자 평균)
        for metric in ["top1_rate", "win_rate"]:
            for j in JUDGES:
                def leak(s):
                    v = cd[(cd.seed == s) & (cd.teacher == j)].set_index("judge")[metric]
                    return v[j], v.drop(j).mean()
                own, oth = leak("all")
                per = [leak(s)[0] - leak(s)[1] for s in SEEDS]
                leaks.append({"criterion": c, "metric": metric, "judge": j, "own": own, "others": oth, "leak": own - oth,
                              **{f"leak_{s}": v for s, v in zip(SEEDS, per)}, "n_positive": sum(v > 0 for v in per)})
        # 3) 위치 편향: 1위 위치 분포, 순서 간 일치
        for j in JUDGES:
            for s in ["all"] + SEEDS:
                ds = d[(d.judge == j) & ((d.seed == s) if s != "all" else True)]
                top = ds[ds["rank"] == 1]
                cnt = [int((top.position == p).sum()) for p in "ABC"]
                same1 = int((top.groupby("group_id").teacher.nunique() == 1).sum())
                full = int(ds.sort_values("rank").groupby(["group_id", "rotation"]).teacher.apply(tuple)
                           .groupby("group_id").nunique().eq(1).sum())
                pos_rows.append({"criterion": c, "judge": j, "seed": s, "n_groups": ds.group_id.nunique(),
                                 "top1_A": cnt[0] / sum(cnt) * 100, "top1_B": cnt[1] / sum(cnt) * 100, "top1_C": cnt[2] / sum(cnt) * 100,
                                 "chi2_p": stats.chisquare(cnt).pvalue, "same_top1_groups": same1, "same_full_groups": full})
            for t in JUDGES:
                r = d[(d.judge == j) & (d.teacher == t)].groupby("position")["rank"].mean()
                pos_rank.append({"criterion": c, "judge": j, "teacher": t, "A": r["A"], "B": r["B"], "C": r["C"]})
        # 4) 질문별 승자 (순서 3가지 평균 순위가 가장 좋은 학생, 동률 별도)
        m = d.groupby(["judge", "seed", "group_id", "teacher"])["rank"].mean().reset_index()
        best = m.groupby(["judge", "seed", "group_id"])["rank"].transform("min")
        w = m[m["rank"] == best].groupby(["judge", "seed", "group_id"]).teacher.agg(lambda x: x.iloc[0] if len(x) == 1 else "tie")
        w = w.reset_index()
        for s in ["all"] + SEEDS:
            for j in JUDGES:
                v = w[(w.judge == j) & ((w.seed == s) if s != "all" else True)].teacher.value_counts()
                win_rows.append({"criterion": c, "seed": s, "judge": j, **{t: int(v.get(t, 0)) for t in JUDGES + ["tie"]}})
        # 5) 평가자 간 1위 일치
        top = d[d["rank"] == 1].pivot_table(index=["group_id", "rotation"], columns="judge", values="teacher", aggfunc="first")
        for a, b in [("gpt", "claude"), ("gpt", "gemini"), ("claude", "gemini")]:
            agree.append({"criterion": c, "pair": f"{a}-{b}", "same_top1": (top[a] == top[b]).mean() * 100})

    for name, rows in [("desc_counts", counts), ("desc_simple_leak", leaks), ("desc_position", pos_rows),
                       ("desc_position_rank", pos_rank), ("desc_winners", win_rows), ("desc_agreement", agree)]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(leaks).round(2).to_string(index=False))
    print(f"저장: {OUT}/desc_*.csv")


if __name__ == "__main__":
    main()
