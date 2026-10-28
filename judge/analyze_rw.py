"""8.2 문체 제거 실험: 교사 답을 HCX-005로 같은 문체로 재작성해 학습한 학생에게서 선호 누출 λ가 줄어드는지 추정한다.

입력
  재작성 학생 채점: judge/output/rw/raw/{judge}.jsonl (run_judge.py --rw, 항목은 select_items_rw.py)
  원래 학생 채점(같은 질문): 본 실험 judge/output/raw/ + 8.1에서 새로 채점한 Gemini 100% 묶음 judge/output/mix/raw/
묶음 = seed × 질문, 학생 3명 = 같은 seed의 GPT·Claude·Gemini 계열 학생. condition = rw(재작성 학생) | orig(원래 학생).

1) 조건마다 본 실험(analyze_models.py)과 같은 순위 로짓(PL)을 따로 적합한다.
     u = θ[seed, t] + δ[j, 위치] + λ_j · 1[t = j]
   비교는 두 조건 모두 채점이 있는 질문(matched)에서 한다. 재작성 학생 전체 300묶음의 λ도 따로 낸다.
   신뢰구간: 질문 단위 군집 부트스트랩. 두 조건이 같은 재표집을 쓰므로 차이 λ_rw − λ_orig(%p)의 신뢰구간도 같은 표본에서 얻는다.
   차이 3개(평가자별)는 Holm 보정.
2) 길이 통제(평가자별 길이 기울기), seed별 λ, 기준 4개(ablation), 기술 통계(쌍대 승률·1위 비율).

사용법: python judge/analyze_rw.py            (부트스트랩 1,000회, 기준 4개는 300회)
        python judge/analyze_rw.py --boot 100 (빠른 점검)
출력: judge/output/rw/ranks_long.csv
      judge/output/rw/analysis/
        rw_lambda.csv   결과 × 조건(rw·orig·rw_all, 길이 통제는 *_len) × 평가자: λ, %p, 95% CI, p
        rw_diff.csv     결과 × 평가자: λ_rw − λ_orig (%p), CI, p, Holm p(overall)
        rw_seed.csv     조건 × seed × 평가자: λ(%p)
        rw_desc.csv     조건 × 평가자 × 학생: 쌍대 승률, 1위 비율
        rw_summary.json 주요 수치
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from analyze_models import JUDGES, OUTCOMES, OWN, explode, fit_pl, load, to_pp
from test_prompt import LABELS, OUTPUT_KEYS

ROOT = Path(__file__).resolve().parent.parent
RW = Path(__file__).parent / "output" / "rw"
MAIN_RAW = Path(__file__).parent / "output" / "raw"
MIX_RAW = Path(__file__).parent / "output" / "mix" / "raw"
SELECTION = Path(__file__).parent / "data" / "rw" / "selection.json"
OUT = RW / "analysis"
SEEDS = [42, 43, 44]
TEACHERS = ["gpt", "claude", "gemini"]
BOOT_SEED = 2026


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def n_tokens(name, folder, seed):
    path = ROOT / "inference" / folder / f"{name}_seed{seed}_eval400.json"
    return {r["id"]: r["n_new_tokens"] for r in json.loads(path.read_text(encoding="utf-8"))}


def merge():
    """채점 기록을 analyze_models.py와 같은 긴 표로 합친다. 묶음 id는 r(재작성)·o(본 실험)·m(8.1)을 붙여 겹치지 않게 한다."""
    chosen = {(s, q) for s, qs in json.loads(SELECTION.read_text(encoding="utf-8"))["selected_question_ids"].items() for q in qs}
    chosen = {(int(s), q) for s, q in chosen}
    tok = {}
    for s in SEEDS:
        for t in TEACHERS:
            tok[("rw", t, s)] = n_tokens(f"{t}_rw", "output_rw", s)
            tok[("orig", t, s)] = n_tokens(t, "output_2nd", s)
    rows = []
    for j in JUDGES:
        rw = [dict(r, condition="rw", group_id=f"r{r['group_id']}") for r in read(RW / "raw" / f"{j}.jsonl")]
        main = [dict(r, condition="orig", group_id=f"o{r['group_id']}") for r in read(MAIN_RAW / f"{j}.jsonl")
                if (r["seed"], r["question_id"]) in chosen]
        mix = [dict(r, condition="orig", group_id=f"m{r['group_id']}") for r in read(MIX_RAW / f"{j}.jsonl")
               if r.get("condition") == "gemini" and (r["seed"], r["question_id"]) in chosen]
        print(f"{j}: 재작성 학생 {len(rw)}개, 원래 학생 {len(main) + len(mix)}개 (본 실험 {len(main)} + 8.1 {len(mix)})")
        for r in rw + main + mix:
            for k in OUTPUT_KEYS:
                for t, rank in r[k]["ranks"].items():
                    rows.append({"judge": j, "condition": r["condition"], "teacher": t, "seed": r["seed"],
                                 "question_id": r["question_id"], "group_id": r["group_id"], "rotation": r["rotation"],
                                 "position": LABELS[r["order"].index(t)],
                                 "n_tokens": tok[(r["condition"], t, r["seed"])][r["question_id"]], "criterion": k, "rank": rank})
    long = pd.DataFrame(rows)
    long.to_csv(RW / "ranks_long.csv", index=False)
    print(f"저장: {RW / 'ranks_long.csv'} ({len(long):,}행)")
    return long


def resample(d, rng, by_q, qs):
    parts = [by_q[q].assign(item=lambda x, k=k: x["item"] + f"_b{k}") for k, q in enumerate(rng.choice(qs, len(qs)))]
    return pd.concat(parts, ignore_index=True)


def pct(draws):
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def p_two(draws):
    draws = np.asarray(draws)
    return max(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())), 1 / len(draws))


def fits(d):
    """조건별 λ: rw·orig는 두 조건 모두 채점된 질문(matched), rw_all은 재작성 학생 전체. *_len은 길이 통제."""
    out = {}
    for name, sub in [("rw", d[(d.condition == "rw") & d.matched]), ("orig", d[d.condition == "orig"]),
                      ("rw_all", d[d.condition == "rw"])]:
        e = explode(sub)
        out[name] = fit_pl(e)[0][OWN]
        if name in ("rw", "orig"):
            out[name + "_len"] = fit_pl(e, length=True)[0][OWN]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--boot-criteria", type=int, default=300, help="기준 4개(ablation)의 부트스트랩 횟수")
    cli = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    long = merge()
    orig_keys = set(zip(long[long.condition == "orig"].seed, long[long.condition == "orig"].question_id))
    long["matched"] = [(s, q) in orig_keys for s, q in zip(long.seed, long.question_id)]
    rng = np.random.default_rng(BOOT_SEED)
    lam_rows, diff_rows, seed_rows = [], [], []

    for crit in OUTCOMES:
        d = load(long, crit)
        d["item"] = d.condition + "_" + d["item"]
        point = fits(d)
        B = cli.boot if crit == "overall" else cli.boot_criteria
        qs = d.question_id.unique()
        by_q = {q: g for q, g in d.groupby("question_id")}
        draws = {k: [] for k in point}
        t0 = time.time()
        for b in range(B):
            if b and b % 100 == 0:
                print(f"  [{crit}] 부트스트랩 {b}/{B} ({(time.time() - t0) / 60:.1f}분, 남은 예상 {(time.time() - t0) / b * (B - b) / 60:.0f}분)", flush=True)
            for k, v in fits(resample(d, rng, by_q, qs)).items():
                draws[k].append(v)
        draws = {k: pd.DataFrame(v) for k, v in draws.items()}
        for c in point:
            for k, j in zip(OWN, JUDGES):
                bd = draws[c][k].values
                lo, hi = pct(bd)
                lam_rows.append({"criterion": crit, "condition": c, "judge": j, "estimate": point[c][k], "ci_low": lo, "ci_high": hi,
                                 "p": p_two(bd), "pp": float(to_pp(point[c][k], False)), "pp_ci_low": float(to_pp(lo, False)),
                                 "pp_ci_high": float(to_pp(hi, False)), "boot": B})
        for suffix in ["", "_len"]:
            for k, j in zip(OWN, JUDGES):
                dpp = to_pp(draws["rw" + suffix][k].values, False) - to_pp(draws["orig" + suffix][k].values, False)
                diff_rows.append({"criterion": crit, "length_control": bool(suffix), "judge": j,
                                  "rw_pp": float(to_pp(point["rw" + suffix][k], False)),
                                  "orig_pp": float(to_pp(point["orig" + suffix][k], False)),
                                  "diff_pp": float(to_pp(point["rw" + suffix][k], False) - to_pp(point["orig" + suffix][k], False)),
                                  "diff_ci_low": pct(dpp)[0], "diff_ci_high": pct(dpp)[1], "p": p_two(dpp)})
        if crit == "overall":
            for s in SEEDS:
                ds = d[d.seed == s]
                for c, sub in [("rw", ds[(ds.condition == "rw") & ds.matched]), ("orig", ds[ds.condition == "orig"]),
                               ("rw_all", ds[ds.condition == "rw"])]:
                    lam = fit_pl(explode(sub))[0]
                    seed_rows += [{"condition": c, "seed": s, "judge": j, "pp": float(to_pp(lam[k], False))} for k, j in zip(OWN, JUDGES)]
        print(f"  {crit} 완료 (부트스트랩 {B}회)", flush=True)

    lam = pd.DataFrame(lam_rows)
    diff = pd.DataFrame(diff_rows)
    diff["p_holm"] = np.nan
    for (crit, lc), g in diff.groupby(["criterion", "length_control"]):
        diff.loc[g.index, "p_holm"] = multipletests(g.p, method="holm")[1]
    o = long[long.criterion == "overall"]
    o = pd.concat([o[(o.condition == "rw") & o.matched].assign(condition="rw"), o[o.condition == "orig"],
                   o[o.condition == "rw"].assign(condition="rw_all")])
    desc = o.groupby(["condition", "judge", "teacher"])["rank"].agg(n="size", mean_rank="mean",
                                                                   top1=lambda r: (r == 1).mean() * 100).reset_index()
    desc["win_rate"] = (3 - desc.mean_rank) / 2 * 100

    lam.to_csv(OUT / "rw_lambda.csv", index=False)
    diff.to_csv(OUT / "rw_diff.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "rw_seed.csv", index=False)
    desc.to_csv(OUT / "rw_desc.csv", index=False)
    ov = lam[lam.criterion == "overall"]
    summary = {"boot": cli.boot, "n_groups": {c: int(long[(long.criterion == "overall") & (long.condition == c)].group_id.nunique())
                                             for c in ["rw", "orig"]},
               "n_matched_questions": int(len(orig_keys)),
               "lambda_pp": {c: {r.judge: [r.pp, r.pp_ci_low, r.pp_ci_high, r.p] for r in ov[ov.condition == c].itertuples()}
                             for c in ov.condition.unique()},
               "diff_overall": diff[diff.criterion == "overall"].to_dict(orient="records")}
    (OUT / "rw_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(diff[diff.criterion == "overall"].round(3).to_string(index=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
