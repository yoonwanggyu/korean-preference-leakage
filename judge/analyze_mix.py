"""8.1 데이터 비율 실험: Gemini 답 비율에 따라 Gemini 평가자의 선호 누출 λ가 줄어드는지 추정한다.

입력: judge/output/mix/ranks_long.csv (merge_mix.py가 만든다), results/output/mix_summary.json (교차 loss·분류기 지표)
묶음 = 조건(Gemini 비율) × seed × 질문, 학생 3명 = GPT 순수 · Claude 순수 · 그 조건의 Gemini 계열 학생(teacher = gemini).

1) 조건별 주 모형: 본 실험(analyze_models.py)과 같은 순위 로짓(PL)을 조건마다 따로 적합한다.
     u = θ[seed, t] + δ[j, 위치] + λ_j · 1[t = j]     → 조건별 λ_Gemini(p), λ_GPT, λ_Claude
   신뢰구간: 질문 단위 군집 부트스트랩. 모든 조건이 같은 재표집을 쓰므로(질문을 뽑으면 그 질문의 모든 seed·조건 자료가 따라온다)
   조건 간 차이 λ(p) − λ(100%)와 비율 λ(p) / λ(100%)의 신뢰구간도 같은 표본에서 얻는다. 차이 5개는 Holm 보정.
2) 비율과의 관계(여섯 조건을 한 모형에): θ_Claude[seed], θ_Gemini계열[조건, seed], δ[j, 위치], λ_GPT, λ_Claude는 공통이고
   Gemini 평가자 항만 바꾼 모형들을 우도로 비교한다.
     M0     누출 없음
     M_const  λ · 1[Gemini 계열]            (H0: 누출이 비율과 무관)
     M_lin    λ · p                          (Gemini 비율에 비례)
     M_loss   λ · 교차 loss 이동 정도(0~1)    (그림 8-4 검정 선)
     M_cls    λ · 분류기 Gemini 판정 비율(0~1) (그림 8-4 초록 선)
     M_sep    조건마다 다른 λ (6개, 가장 유연한 모형)
   우도비 검정: M_const vs M_sep(자유도 5, 비율에 따라 달라지는가), M_lin vs M_sep(자유도 5, 직선으로 충분한가). AIC로 1모수 모형 비교.
   M_lin의 기울기와 M_const의 λ도 부트스트랩 신뢰구간을 낸다.
3) seed별 λ_Gemini, 기준 4개(ablation)의 조건별 λ_Gemini, 기술 통계(쌍대 승률·1위 비율·단순 누출).

사용법: python judge/analyze_mix.py              (부트스트랩 1,000회)
        python judge/analyze_mix.py --boot 100    (빠른 점검)
출력: judge/output/mix/analysis/
  mix_lambda.csv    결과 × 조건 × 평가자: λ, %p, 95% CI, p
  mix_diff.csv      조건별 λ_Gemini(p) − λ_Gemini(100%)와 λ(p)/λ(100%): 추정치, CI, p, Holm p
  mix_models.csv    여섯 조건 통합 모형의 로그우도, 모수 수, AIC, Gemini 항 추정치(·CI)
  mix_seed.csv      조건 × seed × 평가자: λ(%p)
  mix_desc.csv      조건 × 평가자 × 학생(역할): 쌍대 승률, 1위 비율
  mix_summary.json  주요 수치와 우도비 검정
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from analyze_models import JUDGES, OUTCOMES, OWN, explode, fit_design, fit_pl, load, to_pp

ROOT = Path(__file__).resolve().parent.parent
LONG_PATH = Path(__file__).parent / "output" / "mix" / "ranks_long.csv"
STYLE_SUMMARY = ROOT / "results" / "output" / "mix_summary.json"
OUT = Path(__file__).parent / "output" / "mix" / "analysis"
CONDS = ["gemini_mix00", "gemini_mix25", "gemini_mix33", "gemini_mix50", "gemini_mix75", "gemini"]
RATIO = {"gemini_mix00": 0.0, "gemini_mix25": 0.25, "gemini_mix33": 0.334, "gemini_mix50": 0.5, "gemini_mix75": 0.75, "gemini": 1.0}
SEEDS = [42, 43, 44]
BOOT_SEED = 2026
POOLED = ["M0", "M_const", "M_lin", "M_loss", "M_cls", "M_sep"]


def dose_values():
    """조건 → (교차 loss 이동 정도, 분류기 Gemini 판정 비율), 둘 다 0~1. results/mix_analysis.py의 결과."""
    s = json.loads(STYLE_SUMMARY.read_text(encoding="utf-8"))
    return ({c: s["cross_loss_gemini_gap_closed_pct"][c] / 100 for c in CONDS},
            {c: s["classified_as_gemini_pct"][c][0] / 100 for c in CONDS})


def design_pooled(e, model, loss, cls):
    """여섯 조건 통합 모형의 설계 행렬. Gemini 평가자 항만 model에 따라 바뀐다."""
    X = pd.DataFrame(index=e.index)
    for s in SEEDS:   # θ: 묶음마다 GPT 순수 학생이 기준. Claude 순수 학생은 조건과 무관하게 같은 학생이라 seed별 하나
        X[f"q_claude_{s}"] = ((e.teacher == "claude") & (e.seed == s)).astype(float)
        for c in CONDS:
            X[f"q_{c}_{s}"] = ((e.teacher == "gemini") & (e.condition == c) & (e.seed == s)).astype(float)
    for j in JUDGES:
        for p in ["A", "B"]:
            X[f"pos_{j}_{p}"] = ((e.judge == j) & (e.position == p)).astype(float)
    X["own_gpt"] = ((e.judge == "gpt") & (e.teacher == "gpt")).astype(float)
    X["own_claude"] = ((e.judge == "claude") & (e.teacher == "claude")).astype(float)
    gem = ((e.judge == "gemini") & (e.teacher == "gemini")).astype(float)
    if model == "M_const":
        X["own_gemini"] = gem
    elif model == "M_lin":
        X["own_gemini"] = gem * e.condition.map(RATIO)
    elif model == "M_loss":
        X["own_gemini"] = gem * e.condition.map(loss)
    elif model == "M_cls":
        X["own_gemini"] = gem * e.condition.map(cls)
    elif model == "M_sep":
        for c in CONDS:
            X[f"own_gemini_{c}"] = gem * (e.condition == c)
    return X


def per_condition(e):
    """조건마다 본 실험과 같은 PL을 적합해 λ 세 개를 돌려준다."""
    return {c: fit_pl(e[e.condition == c])[0][OWN] for c in CONDS}


def resample(d, rng, by_q, qs):
    parts = [by_q[q].assign(item=lambda x, k=k: x["item"] + f"_b{k}") for k, q in enumerate(rng.choice(qs, len(qs)))]
    return explode(pd.concat(parts, ignore_index=True))


def pct(draws):
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def p_two(draws):
    draws = np.asarray(draws)
    return max(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())), 1 / len(draws))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--boot-criteria", type=int, default=300, help="기준 4개(ablation)의 부트스트랩 횟수")
    cli = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(LONG_PATH)
    loss, cls = dose_values()
    rng = np.random.default_rng(BOOT_SEED)
    lam_rows, diff_rows, seed_rows = [], [], []

    for crit in OUTCOMES:
        d = load(long, crit)
        e = explode(d)
        point = per_condition(e)
        B = cli.boot if crit == "overall" else cli.boot_criteria
        qs = d.question_id.unique()
        by_q = {q: g for q, g in d.groupby("question_id")}
        draws = {c: [] for c in CONDS}
        pooled_draws = {"M_const": [], "M_lin": []}
        t0 = time.time()
        for b in range(B):
            if b and b % 100 == 0:   # 진행 표시: 결과 단위(overall·기준)마다 100회씩
                print(f"  [{crit}] 부트스트랩 {b}/{B} ({(time.time() - t0) / 60:.1f}분, 남은 예상 {(time.time() - t0) / b * (B - b) / 60:.0f}분)", flush=True)
            eb = resample(d, rng, by_q, qs)
            for c, v in per_condition(eb).items():
                draws[c].append(v)
            if crit == "overall":
                for m in pooled_draws:
                    pooled_draws[m].append(fit_design(design_pooled(eb, m, loss, cls), eb)[0]["own_gemini"])
        draws = {c: pd.DataFrame(v) for c, v in draws.items()}
        for c in CONDS:
            for k, j in zip(OWN, JUDGES):
                bd = draws[c][k].values
                lo, hi = pct(bd)
                lam_rows.append({"criterion": crit, "condition": c, "ratio": RATIO[c], "judge": j, "estimate": point[c][k],
                                 "ci_low": lo, "ci_high": hi, "p": p_two(bd), "pp": float(to_pp(point[c][k], False)),
                                 "pp_ci_low": float(to_pp(lo, False)), "pp_ci_high": float(to_pp(hi, False)), "boot": B})
        if crit == "overall":
            ref_pt, ref_bd = point["gemini"]["own_gemini"], draws["gemini"]["own_gemini"].values
            for c in CONDS[:-1]:
                pt, bd = point[c]["own_gemini"], draws[c]["own_gemini"].values
                dpp = to_pp(bd, False) - to_pp(ref_bd, False)
                rel = bd / ref_bd
                diff_rows.append({"condition": c, "ratio": RATIO[c],
                                  "diff_pp": float(to_pp(pt, False) - to_pp(ref_pt, False)), "diff_ci_low": pct(dpp)[0],
                                  "diff_ci_high": pct(dpp)[1], "p": p_two(dpp), "rel_pct": pt / ref_pt * 100,
                                  "rel_ci_low": pct(rel)[0] * 100, "rel_ci_high": pct(rel)[1] * 100})
            for s in SEEDS:
                es = e[e.seed == s]
                for c in CONDS:
                    lam = fit_pl(es[es.condition == c])[0]
                    seed_rows += [{"condition": c, "ratio": RATIO[c], "seed": s, "judge": j, "pp": float(to_pp(lam[k], False))}
                                  for k, j in zip(OWN, JUDGES)]
            e_overall, pooled_boot = e, pooled_draws
        print(f"  {crit} 완료 (부트스트랩 {B}회)")

    diff = pd.DataFrame(diff_rows)
    diff["p_holm"] = multipletests(diff.p, method="holm")[1]

    # 여섯 조건 통합 모형
    model_rows, fits = [], {}
    for m in POOLED:
        params, llf = fit_design(design_pooled(e_overall, m, loss, cls), e_overall)
        fits[m] = (params, llf)
        k_gem = {"M0": 0, "M_sep": len(CONDS)}.get(m, 1)
        row = {"model": m, "llf": llf, "k_gemini": k_gem, "aic_rel": 2 * k_gem - 2 * llf}
        if m in pooled_boot:
            lam = params["own_gemini"]
            row.update({"lambda": lam, "ci_low": pct(pooled_boot[m])[0], "ci_high": pct(pooled_boot[m])[1],
                        "p": p_two(pooled_boot[m]), "pp": float(to_pp(lam, False))})
        elif m in ("M_loss", "M_cls"):
            row.update({"lambda": params["own_gemini"], "pp": float(to_pp(params["own_gemini"], False))})
        model_rows.append(row)
    models = pd.DataFrame(model_rows)
    models["daic_vs_M0"] = models.aic_rel - models.loc[models.model == "M0", "aic_rel"].iloc[0]
    llf = dict(zip(models.model, models.llf))
    lr = {name: {"lr": 2 * (llf["M_sep"] - llf[a]), "df": len(CONDS) - 1, "p": stats.chi2.sf(2 * (llf["M_sep"] - llf[a]), len(CONDS) - 1)}
          for name, a in [("const_vs_sep", "M_const"), ("lin_vs_sep", "M_lin"), ("loss_vs_sep", "M_loss"), ("cls_vs_sep", "M_cls")]}
    lr["M0_vs_lin"] = {"lr": 2 * (llf["M_lin"] - llf["M0"]), "df": 1, "p": stats.chi2.sf(2 * (llf["M_lin"] - llf["M0"]), 1)}

    # 기술 통계
    o = long[long.criterion == "overall"]
    desc = (o.groupby(["condition", "judge", "teacher"])["rank"]
            .agg(n="size", mean_rank="mean", top1=lambda r: (r == 1).mean() * 100).reset_index())
    desc["win_rate"] = (3 - desc.mean_rank) / 2 * 100
    desc["ratio"] = desc.condition.map(RATIO)

    lam = pd.DataFrame(lam_rows)
    lam.to_csv(OUT / "mix_lambda.csv", index=False)
    diff.to_csv(OUT / "mix_diff.csv", index=False)
    models.to_csv(OUT / "mix_models.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "mix_seed.csv", index=False)
    desc.to_csv(OUT / "mix_desc.csv", index=False)
    n_items = o.groupby("judge").apply(lambda x: x.groupby(["group_id", "rotation"]).ngroups).to_dict()
    summary = {"boot": cli.boot, "boot_criteria": cli.boot_criteria, "rankings_per_judge": n_items,
               "lr_tests": lr, "pooled_models": models.to_dict("records"),
               "overall_gemini": lam[(lam.criterion == "overall") & (lam.judge == "gemini")].to_dict("records"),
               "diff_vs_100": diff.to_dict("records")}
    (OUT / "mix_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(lam[lam.criterion == "overall"].round(4).to_string(index=False))
    print(diff.round(4).to_string(index=False))
    print(models.round(4).to_string(index=False))
    print(json.dumps(lr, indent=1, default=float))
    print(desc[desc.teacher == "gemini"].round(2).to_string(index=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
