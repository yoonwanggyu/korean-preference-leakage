"""선호 누출의 통계 모형: 순위 로짓(Plackett-Luce) 주 모형 + 쌍 고정효과 선형확률모형(LPM) 강건성 분석.

입력: judge/output/ranks_long.csv (merge_results.py가 만든다). 한 순위 = (judge, group_id, rotation, criterion)의 3행.

1) 주 모형: Plackett-Luce (순위를 "3개 중 1위 선택 → 남은 2개 중 선택"으로 펼친 조건부 로짓)
     u(j, g, r, t) = θ[seed, t] + δ[j, 위치] + λ_j · 1[t = j]
   θ: 학생 15명(교사 × seed)의 품질, δ: 평가자별 위치 효과(C 기준), λ_j: 평가자 j의 자기 계열 효과(로그 오즈).
   쌍 (s, t)마다 두 학생과 무관한 제3의 평가자가 있어 품질 차이와 λ가 함께 식별된다.
   식별 가정(평가자는 자기 계열이 아닌 두 학생을 품질 외에는 같게 대함)은 평가자 × 교사 상호작용 모형과의
   우도비 검정(자유도 1)으로 확인한다. 표준오차는 질문 단위 군집 부트스트랩(질문 100개 복원 추출).
   해석용 변환: 품질이 같은 상대를 이길 확률의 증가 = σ(λ) − 0.5 (PL에서 쌍 승률은 정확히 e^ua / (e^ua + e^ub)).
   민감도: 평가자별 길이 기울기(표준화 log 토큰 수)를 넣은 모형.

2) 강건성: 쌍 고정효과 LPM. 같은 묶음의 학생 쌍마다 9번(평가자 3 × 순서 3) 판정.
     y(s가 t를 이김) = α[쌍] + Σ_j λ_j · (1[평가자=j, s=j] − 1[평가자=j, t=j]) + 평가자별 위치 더미 + ε
   α[쌍]이 그 질문에서 두 답의 품질 차이(내용, 길이, 난이도)를 흡수한다. λ_j는 %p. 표준오차는 질문 단위 군집.

3) 1위만 쓴 모형(PL_top1): 순위 중 "세 답 중 1위 선택" 단계만 쓴 조건부 로짓. λ를 1위 비율의 변화로 바꿔 쓴다:
   품질이 같은 세 답 중 자기 계열 답이 1위로 뽑힐 확률 = e^λ / (e^λ + 2), 기준선 1/3.

4) 결과(overall + 기준 4개)마다 위 세 모형(PL, PL_top1, LPM)을 같은 방식으로 적합한다. overall이 주 결과,
   기준 4개는 ablation이다. seed별로도 적합해 부호 검정·t검정을 한다.
   Holm 보정 묶음: (모형마다) overall의 평가자 3개, 기준 4 × 평가자 3 = 12개, overall seed 검정 3개.

사용법: python judge/analyze_models.py            (부트스트랩 1,000회)
        python judge/analyze_models.py --boot 200 (빠른 점검)
출력: judge/output/analysis/
  model_results.csv  결과 × 모형 × 평가자: 추정치, 95% CI, p, Holm p, 해석용 %p
  model_by_seed.csv  결과 × 모형 × seed × 평가자: seed별 추정치(%p)
  model_seed_tests.csv 결과 × 모형 × 평가자: 양수 seed 수, 부호 검정·t검정 p (overall은 Holm 포함)
  model_position.csv 결과 × 평가자: 위치 효과(로그 오즈, 오즈비)
  model_overid.csv   결과별 식별 가정 검정 (자유도 1 우도비)
  model_length.csv   overall 길이 통제 민감도
  models_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.multitest import multipletests

LONG_PATH = Path(__file__).parent / "output" / "ranks_long.csv"
OUT = Path(__file__).parent / "output" / "analysis"
JUDGES = ["gpt", "claude", "gemini"]
SEEDS = [42, 43, 44, 45, 46]
CRITERIA = ["factuality", "comprehension", "logical_robustness", "readability"]
OUTCOMES = ["overall"] + CRITERIA
OWN = [f"own_{j}" for j in JUDGES]
BOOT_SEED = 2026


def load(long, criterion):
    d = long[(long.criterion == criterion) & long.judge.isin(JUDGES)].copy()
    d["item"] = d.judge + "_" + d.group_id.astype(str) + "_" + d.rotation.astype(str)
    loglen = np.log(d.n_tokens)
    d["z_len"] = (loglen - loglen.mean()) / loglen.std()
    return d


def explode(d):
    """순위 → 선택 단계 2개. 단계 1: 세 학생 중 1위 선택, 단계 2: 남은 두 학생 중 2위 선택."""
    s1 = d.assign(stage=1, chosen=(d["rank"] == 1).astype(int))
    s2 = d[d["rank"] >= 2].assign(stage=2, chosen=lambda x: (x["rank"] == 2).astype(int))
    e = pd.concat([s1, s2], ignore_index=True)
    e["set"] = e["item"] + "_" + e["stage"].astype(str)
    return e


def design(e, length=False, full=False):
    X = pd.DataFrame(index=e.index)
    if full:   # 평가자 × 교사 상호작용 (자유도 1 검정용 일반 모형)
        for j in ["claude", "gemini"]:
            for t in ["claude", "gemini"]:
                X[f"jt_{j}_{t}"] = ((e.judge == j) & (e.teacher == t)).astype(float)
    else:
        for j in JUDGES:
            X[f"own_{j}"] = ((e.judge == j) & (e.teacher == j)).astype(float)
    for t in ["claude", "gemini"]:   # θ[seed, t]: seed마다 GPT 학생이 기준 (한 선택 집합은 seed가 같아 seed별 상수는 식별되지 않는다)
        for s in SEEDS:
            X[f"q_{t}_{s}"] = ((e.teacher == t) & (e.seed == s)).astype(float)
    for j in JUDGES:          # δ[j, 위치]: C 기준
        for p in ["A", "B"]:
            X[f"pos_{j}_{p}"] = ((e.judge == j) & (e.position == p)).astype(float)
    if length:
        for j in JUDGES:
            X[f"len_{j}"] = (e.judge == j) * e.z_len
    return X


def fit_pl(e, length=False, full=False):
    """펼친 순위 로짓의 최대우도 추정. 선택 집합마다 softmax를 벡터화해 직접 계산한다
    (statsmodels ConditionalLogit과 같은 값, 부트스트랩을 위해 수백 배 빠르게)."""
    X = design(e, length, full)
    X = X.loc[:, X.std() > 0]   # seed별 적합 등에서 빈 열 제거
    codes, _ = pd.factorize(e["set"])
    idx = np.argsort(codes, kind="stable")
    Xv, y, codes = X.values[idx], e["chosen"].values[idx].astype(float), codes[idx]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])

    def negll(b):
        eta = Xv @ b
        mx = np.maximum.reduceat(eta, starts)[codes]
        lse = np.log(np.add.reduceat(np.exp(eta - mx), starts))[codes] + mx
        prob = np.exp(eta - lse)
        ll = (y * eta).sum() - lse[starts].sum()
        return -ll, -(Xv.T @ (y - prob))

    res = minimize(negll, np.zeros(Xv.shape[1]), jac=True, method="BFGS", options={"gtol": 1e-6, "maxiter": 1000})
    return pd.Series(res.x, index=X.columns), -res.fun


def bootstrap_pl(d, B, length=False, top1=False):
    """질문 단위 군집 부트스트랩: 질문 100개를 복원 추출하고 그 질문의 모든 seed·순서·평가자 자료를 가져온다."""
    rng = np.random.default_rng(BOOT_SEED)
    qs = d.question_id.unique()
    by_q = {q: g for q, g in d.groupby("question_id")}
    draws = []
    for b in range(B):
        parts = [by_q[q].assign(item=lambda x, k=k: x["item"] + f"_b{k}") for k, q in enumerate(rng.choice(qs, len(qs)))]
        e = explode(pd.concat(parts, ignore_index=True))
        params, _ = fit_pl(e[e.stage == 1] if top1 else e, length)
        draws.append(params[OWN])
    return pd.DataFrame(draws)


def to_pp(lam, top1):
    """λ → 해석용 %p. PL: 품질이 같은 상대를 이길 확률 − 50%. PL_top1: 같은 세 답 중 1위로 뽑힐 확률 − 1/3."""
    lam = np.asarray(lam, float)
    return (np.exp(lam) / (np.exp(lam) + 2) - 1 / 3) * 100 if top1 else (1 / (1 + np.exp(-lam)) - 0.5) * 100


def summarize_pl(point, draws, criterion, top1=False):
    rows = []
    for k, j in zip(OWN, JUDGES):
        lam, bd = point[k], draws[k].values
        p = min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))   # 부트스트랩 양쪽 p
        rows.append({"criterion": criterion, "model": "PL_top1" if top1 else "PL", "judge": j, "estimate": lam,
                     "se": bd.std(ddof=1), "ci_low": np.percentile(bd, 2.5), "ci_high": np.percentile(bd, 97.5),
                     "p": max(p, 1 / len(bd)), "odds_ratio": np.exp(lam), "pp": float(to_pp(lam, top1)),
                     "pp_ci_low": float(to_pp(np.percentile(bd, 2.5), top1)), "pp_ci_high": float(to_pp(np.percentile(bd, 97.5), top1))})
    return rows


def pairs(d):
    """쌍 자료: 묶음의 학생 쌍 (s, t)마다 평가자 3 × 순서 3 = 9번 판정. s는 JUDGES 순서상 앞 교사."""
    key = ["judge", "seed", "question_id", "group_id", "rotation"]
    m = d.merge(d, on=key, suffixes=("_s", "_t"))
    order = {t: i for i, t in enumerate(JUDGES)}
    m = m[m.teacher_s.map(order) < m.teacher_t.map(order)].copy()
    m["y"] = (m.rank_s < m.rank_t).astype(float)
    m["pair"] = m.group_id.astype(str) + "_" + m.teacher_s + "_" + m.teacher_t
    return m


def fit_lpm(m):
    X = pd.DataFrame(index=m.index)
    for j in JUDGES:
        X[f"own_{j}"] = (m.judge == j) * ((m.teacher_s == j).astype(float) - (m.teacher_t == j).astype(float))
    for j in JUDGES:
        for p in ["A", "B"]:
            X[f"pos_s_{j}_{p}"] = ((m.judge == j) & (m.position_s == p)).astype(float)
            X[f"pos_t_{j}_{p}"] = ((m.judge == j) & (m.position_t == p)).astype(float)
    X = X.loc[:, X.std() > 0]
    # 쌍 고정효과는 쌍 평균을 빼서(within 변환) 흡수한다. 쌍은 질문 안에 포함되므로 질문 군집 표준오차에 자유도 보정이 필요 없다.
    y = m.y - m.groupby("pair").y.transform("mean")
    Xw = X - X.groupby(m["pair"]).transform("mean")
    res = sm.OLS(y.values, Xw.values).fit(cov_type="cluster", cov_kwds={"groups": m.question_id.values})
    return pd.DataFrame({"coef": res.params, "se": res.bse, "p": res.pvalues}, index=X.columns)


def seed_tests(values):
    v = np.asarray(values, float)
    k = int((v > 0).sum())
    return {"n_positive": k, "sign_p_one_sided": stats.binomtest(k, len(v), 0.5, alternative="greater").pvalue,
            "mean": v.mean(), "t_p_one_sided": stats.ttest_1samp(v, 0.0, alternative="greater").pvalue}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1000)
    cli = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(LONG_PATH)
    results, seed_rows, pos_rows, overid_rows = [], [], [], []

    for c in OUTCOMES:
        d = load(long, c)
        e = explode(d)
        # PL (전체 순위), PL_top1 (1위 선택만)
        for top1 in [False, True]:
            ee = e[e.stage == 1] if top1 else e
            point, llf = fit_pl(ee)
            results += summarize_pl(point, bootstrap_pl(d, cli.boot, top1=top1), c, top1)
            if not top1:
                _, llf_full = fit_pl(e, full=True)
                lr = 2 * (llf_full - llf)
                overid_rows.append({"criterion": c, "llf_lambda_model": llf, "llf_full_model": llf_full, "lr": lr, "df": 1,
                                    "p": stats.chi2.sf(lr, 1)})
                pos_rows += [{"criterion": c, "judge": j, "position": p, "delta": point[f"pos_{j}_{p}"],
                              "odds_ratio_vs_C": np.exp(point[f"pos_{j}_{p}"])} for j in JUDGES for p in ["A", "B"]]
        # LPM (쌍 고정효과)
        lpm = fit_lpm(pairs(d))
        results += [{"criterion": c, "model": "LPM", "judge": j, "estimate": lpm.loc[k, "coef"], "se": lpm.loc[k, "se"],
                     "ci_low": lpm.loc[k, "coef"] - 1.96 * lpm.loc[k, "se"], "ci_high": lpm.loc[k, "coef"] + 1.96 * lpm.loc[k, "se"],
                     "p": lpm.loc[k, "p"], "odds_ratio": np.nan, "pp": lpm.loc[k, "coef"] * 100,
                     "pp_ci_low": (lpm.loc[k, "coef"] - 1.96 * lpm.loc[k, "se"]) * 100,
                     "pp_ci_high": (lpm.loc[k, "coef"] + 1.96 * lpm.loc[k, "se"]) * 100} for k, j in zip(OWN, JUDGES)]
        # seed별
        for sd in SEEDS:
            ds = d[d.seed == sd]
            es = explode(ds)
            sp, _ = fit_pl(es)
            st, _ = fit_pl(es[es.stage == 1])
            sl = fit_lpm(pairs(ds))
            for k, j in zip(OWN, JUDGES):
                seed_rows += [{"criterion": c, "model": "PL", "seed": sd, "judge": j, "pp": float(to_pp(sp[k], False))},
                              {"criterion": c, "model": "PL_top1", "seed": sd, "judge": j, "pp": float(to_pp(st[k], True))},
                              {"criterion": c, "model": "LPM", "seed": sd, "judge": j, "pp": sl.loc[k, "coef"] * 100}]
        print(f"  {c} 완료")

    res = pd.DataFrame(results)
    for mdl in ["PL", "PL_top1", "LPM"]:   # Holm: overall 3개, 기준 12개를 각각 한 묶음으로
        for fam in [res.criterion == "overall", res.criterion != "overall"]:
            idx = res[(res.model == mdl) & fam].index
            res.loc[idx, "p_holm"] = multipletests(res.loc[idx, "p"], method="holm")[1]
    seed_df = pd.DataFrame(seed_rows)
    tests = pd.DataFrame([{"criterion": c, "model": mdl, "judge": j,
                           **seed_tests(seed_df[(seed_df.criterion == c) & (seed_df.model == mdl) & (seed_df.judge == j)].pp)}
                          for c in OUTCOMES for mdl in ["PL", "PL_top1", "LPM"] for j in JUDGES])
    for mdl in ["PL", "PL_top1", "LPM"]:
        idx = tests[(tests.criterion == "overall") & (tests.model == mdl)].index
        tests.loc[idx, "sign_p_holm"] = multipletests(tests.loc[idx, "sign_p_one_sided"], method="holm")[1]
        tests.loc[idx, "t_p_holm"] = multipletests(tests.loc[idx, "t_p_one_sided"], method="holm")[1]

    # 길이 민감도 (overall)
    d = load(long, "overall")
    lpoint, _ = fit_pl(explode(d), length=True)
    length_df = pd.DataFrame(summarize_pl(lpoint, bootstrap_pl(d, max(200, cli.boot // 5), length=True), "overall+length"))
    length_df["len_slope"] = [lpoint[f"len_{j}"] for j in JUDGES]

    res.to_csv(OUT / "model_results.csv", index=False)
    seed_df.to_csv(OUT / "model_by_seed.csv", index=False)
    tests.to_csv(OUT / "model_seed_tests.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(OUT / "model_position.csv", index=False)
    pd.DataFrame(overid_rows).to_csv(OUT / "model_overid.csv", index=False)
    length_df.to_csv(OUT / "model_length.csv", index=False)
    summary = {"boot": cli.boot, "overall": res[res.criterion == "overall"].to_dict("records"),
               "overid_overall": overid_rows[0]}
    (OUT / "models_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    cols = ["criterion", "model", "judge", "estimate", "pp", "pp_ci_low", "pp_ci_high", "p", "p_holm"]
    print(res[cols].round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(pd.DataFrame(overid_rows).round(4).to_string(index=False))
    print(length_df.round(4).to_string(index=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
