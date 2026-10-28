"""8.1 데이터 비율 실험: Gemini 답 비율을 바꾼 학생 15명의 학습·답변 결과를 분석한다(평가자 채점 전 단계).

비교 대상(모두 seed 42·43·44): 혼합 학생 5비율(Gemini 0·25·33·50·75%) + 기존 순수 학생(GPT·Claude·Gemini 100%).
1. 교차 loss: 학생이 세 교사의 검증 답을 얼마나 잘 예측하는가(sft/runs/*/cross_loss.csv).
2. 문체 분류기: 교사 답 3,600개로 학습한 로지스틱 회귀(classify/logistic.py와 같은 설정, raw)가 학생 답을 누구 답으로 보는가.
3. 문체 특징: 학생 답의 형식·문체 지표(results/style_analysis.py의 features).
4. 완결성: 학생이 끝 토큰으로 스스로 끝맺은 답의 수와, 평가 질문 선정(8.1 ④)에 쓸 수 있는 공통 완전 질문 수.
2·3은 끝맺은 답(완전한 답)만 쓴다. 신뢰구간은 질문 단위 부트스트랩(1,000회, 세 seed를 함께 재표집)이다.

사용법: python results/mix_analysis.py
출력: results/output/mix_*.csv, mix_summary.json
      description/figures/34~40_그림8-1~8-7_*.png (포스터용 그림, 설명 문구는 그림에 넣지 않고 poster.html에 쓴다)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "results"))
from style_analysis import features  # noqa: E402

OUT = Path(__file__).parent / "output"
FIG = ROOT / "description" / "figures"
TEACHERS = ["gpt", "claude", "gemini"]
SEEDS = [42, 43, 44]
MAX_NEW_TOKENS = 1024
N_BOOT = 1000
# 조건 이름 → (Gemini 비율 %, 그림 표시 이름, 답 파일 폴더). 순수 GPT·Claude 학생은 비율 축 밖의 기준점이다.
CONDS = {
    "gemini_mix00": (0, "0%", "output_mix"),
    "gemini_mix25": (25, "25%", "output_mix"),
    "gemini_mix33": (33.4, "33%\n(균등)", "output_mix"),
    "gemini_mix50": (50, "50%", "output_mix"),
    "gemini_mix75": (75, "75%", "output_mix"),
    "gemini": (100, "100%", "output_2nd"),
}
PURE = {"gpt": "GPT 순수", "claude": "Claude 순수"}
CLF_C = 16  # classify/output/logistic/metrics.json의 raw 조건 best C

COLORS = {"gpt": "#2a78d6", "claude": "#eb6834", "gemini": "#1baf7a"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"
FONT = "Apple SD Gothic Neo"


def setup():
    plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False, "font.size": 15,
                         "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14})


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("그림:", name)


def complete(r):
    return r["finished"] and r["n_new_tokens"] < MAX_NEW_TOKENS and r["output"].strip() != ""


def load_answers():
    folder = {**{k: v[2] for k, v in CONDS.items()}, "gpt": "output_2nd", "claude": "output_2nd"}
    return {(s, seed): json.loads((ROOT / "inference" / folder[s] / f"{s}_seed{seed}_eval400.json").read_text(encoding="utf-8"))
            for s in folder for seed in SEEDS}


def fit_classifier():
    texts, labels = [], []
    for i, t in enumerate(TEACHERS):
        for r in json.loads((ROOT / "inference" / "output" / f"{t}_train_1200.json").read_text(encoding="utf-8")):
            if r.get(f"{t}_output"):
                texts.append(r[f"{t}_output"])
                labels.append(i)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
    clf = LogisticRegression(C=CLF_C, max_iter=3000).fit(vec.fit_transform(texts), labels)
    print(f"분류기: 교사 답 {len(texts):,}개로 학습")
    return vec, clf


def answer_table(answers):
    tok = AutoTokenizer.from_pretrained(ROOT / "sft" / "models" / "Qwen3-0.6B-Base")
    vec, clf = fit_classifier()
    rows = []
    for (s, seed), items in answers.items():
        for r in items:
            if complete(r):
                rows.append({"student": s, "seed": seed, "id": r["id"], "text": r["output"],
                             **features(r["output"], r["instruction"], tok)})
    df = pd.DataFrame(rows)
    prob = clf.predict_proba(vec.transform(df["text"]))
    for i, t in enumerate(TEACHERS):
        df[f"p_{t}"] = prob[:, i]
    df["pred"] = np.array(TEACHERS)[prob.argmax(1)]
    return df.drop(columns="text")


def boot_ci(df, col, rng):
    """질문 단위 부트스트랩: 질문 id를 복원 추출하고 그 질문의 답(seed 3개)을 모두 쓴다."""
    by_q = df.groupby("id")[col].agg(["sum", "count"])
    s, c = by_q["sum"].to_numpy(), by_q["count"].to_numpy()
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(est, [2.5, 97.5])


# ---------- 그림 ----------

def fig_data(summary):
    """그림 8-1. 조건별 학습 데이터 구성(1,000개 중 교사별 답 개수)."""
    names = list(CONDS)
    counts = {n: summary["mixes"][n]["counts"] if n in summary["mixes"] else {"gpt": 0, "claude": 0, "gemini": 1000}
              for n in names}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for t, lab in zip(["gemini", "gpt", "claude"], ["Gemini 답", "GPT 답", "Claude 답"]):
        v = np.array([counts[n][t] for n in names])
        ax.barh(y, v, left=left, height=0.68, color=COLORS[t], edgecolor=SURFACE, lw=2, label=lab)
        for yy, l, w in zip(y, left, v):
            if w >= 100:
                ax.text(l + w / 2, yy, f"{w:,}", ha="center", va="center", color="white", fontsize=15, fontweight="bold")
        left += v
    ax.set_yticks(y, [CONDS[n][1].replace("\n", " ") for n in names])
    ax.set_ylabel("Gemini 답 비율")
    ax.set_xlabel("학습 예제 수 (학생 1명당 1,000개)")
    ax.set_xlim(0, 1000)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)
    style(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    save(fig, "34_그림8-1_8.1 결과 - 학습 데이터 구성.png")


def fig_cross_loss(cl):
    """그림 8-2. Gemini 비율에 따른 세 교사 검증 답에 대한 교차 loss(seed 3개 평균, seed 간 표준편차 ≤ 0.003이라 생략)."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = [CONDS[n][0] for n in CONDS]
    for t, lab in zip(TEACHERS, ["GPT 검증 답", "Claude 검증 답", "Gemini 검증 답"]):
        sub = cl[cl["eval_teacher"] == t]
        m = [sub[sub["student_teacher"] == n]["loss"].mean() for n in CONDS]
        ax.plot(x, m, "-o", color=COLORS[t], lw=2.5, ms=10, mec=SURFACE, mew=2, label=lab, zorder=3)
        ax.annotate(f"{m[0]:.3f}", (x[0], m[0]), xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=13, color=INK_2)
        ax.annotate(f"{m[-1]:.3f}", (x[-1], m[-1]), xytext=(12, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=13, color=INK_2)
    ax.set_xticks(x, [CONDS[n][1] for n in CONDS])
    ax.set_xlim(-12, 112)
    ax.set_xlabel("학생 학습 데이터 중 Gemini 답 비율")
    ax.set_ylabel("교차 loss (낮을수록 그 교사 답과 비슷)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    style(ax)
    save(fig, "35_그림8-2_8.1 결과 - Gemini 비율에 따른 교차 loss.png")


def fig_classifier(share):
    """그림 8-3. 분류기가 학생 답을 어느 교사의 답으로 판정했는가(완전한 답 전체, 100% 누적 막대)."""
    order = ["gpt", "claude"] + list(CONDS)
    labels = ["GPT\n순수", "Claude\n순수"] + [CONDS[n][1] for n in CONDS]
    fig, ax = plt.subplots(figsize=(12, 6.4))
    x = np.arange(len(order))
    x = np.where(x >= 2, x + 0.6, x)  # 순수 기준 학생과 비율 조건 사이를 띄운다
    bottom = np.zeros(len(order))
    for t, lab in zip(["gemini", "gpt", "claude"], ["Gemini로 판정", "GPT로 판정", "Claude로 판정"]):
        v = np.array([share.loc[s, t] * 100 for s in order])
        ax.bar(x, v, bottom=bottom, width=0.72, color=COLORS[t], edgecolor=SURFACE, lw=2, label=lab)
        if t == "gemini":
            for xx, vv in zip(x, v):
                ax.text(xx, vv + 1.5 if vv < 12 else vv / 2, f"{vv:.0f}%", ha="center",
                        va="bottom" if vv < 12 else "center", fontsize=15, fontweight="bold",
                        color=INK if vv < 12 else "white")
        bottom += v
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("학생 답 중 해당 교사로 판정된 비율 (%)")
    ax.text(np.mean(x[2:]), -0.12, "Gemini 답 비율", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    style(ax)
    ax.tick_params(axis="x", length=0)
    save(fig, "36_그림8-3_8.1 결과 - 분류기가 본 학생 답의 출처.png")


def fig_shift(gap, cls, cls_ci):
    """그림 8-4. 0% → 100% 사이에서 학생이 Gemini 쪽으로 얼마나 이동했는가: 교차 loss 기준과 생성 답 기준."""
    fig, ax = plt.subplots(figsize=(10.5, 7))
    x = np.array([CONDS[n][0] for n in CONDS])
    ax.plot([0, 100], [0, 100], ls="--", color=MUTED, lw=1.8, label="비율과 똑같이 이동")
    ax.plot(x, gap, "-o", color=INK, lw=2.8, ms=11, mec=SURFACE, mew=2, label="교차 loss 기준 (Gemini 검증 답)")
    ax.errorbar(x, cls, yerr=[cls - cls_ci[:, 0], cls_ci[:, 1] - cls], fmt="-s", color=COLORS["gemini"], lw=2.8,
                ms=11, mec=SURFACE, mew=2, capsize=5, elinewidth=1.8, label="생성 답 기준 (Gemini로 판정된 비율)")
    for xx, a, b in zip(x[1:-1], gap[1:-1], cls[1:-1]):
        ax.annotate(f"{a:.0f}", (xx, a), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=14, color=INK)
        ax.annotate(f"{b:.0f}", (xx, b), xytext=(14, -20), textcoords="offset points", ha="left", fontsize=14,
                    color=COLORS["gemini"], fontweight="bold")
    ax.set_xticks(x, [CONDS[n][1] for n in CONDS])
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("학생 학습 데이터 중 Gemini 답 비율")
    ax.set_ylabel("Gemini 쪽으로 이동한 정도 (0% 학생 = 0, 100% 학생 = 100)")
    ax.legend(loc="upper left", frameon=False)
    style(ax, grid=None)
    ax.grid(True, color=GRID, lw=0.8)
    save(fig, "37_그림8-4_8.1 결과 - 학습 적합과 생성 답의 이동 비교.png")


STYLE_PANELS = [("numbered", "번호 목록(1.) 사용 (%)", 100), ("bullet", "글머리 목록(-) 사용 (%)", 100),
                ("en_paren", "영문 괄호 병기 (%)", 100), ("hedge", "완곡 표현 사용 (%)", 100),
                ("formal_ratio", "합쇼체(~니다) 비중 (%)", 100), ("chars", "답 길이 (글자 수 중앙값)", 1)]


def fig_style(df):
    """그림 8-5. 학생 답의 문체 특징이 Gemini 비율에 따라 어떻게 변하는가(점 = seed, 선 = 평균)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    x = [CONDS[n][0] for n in CONDS]
    for ax, (col, title, k) in zip(axes.ravel(), STYLE_PANELS):
        agg = "median" if col == "chars" else "mean"
        per_seed = df.groupby(["student", "seed"])[col].agg(agg) * k
        mean = df.groupby("student")[col].agg(agg) * k
        for t in ["gpt", "claude"]:
            ax.axhline(mean[t], color=COLORS[t], ls=":", lw=2.4, label=PURE[t] + " 학생")
        ax.plot(x, [mean[n] for n in CONDS], "-o", color=COLORS["gemini"], lw=2.8, ms=9, mec=SURFACE, mew=2,
                label="Gemini 비율별 학생", zorder=3)
        for n in CONDS:
            v = per_seed[n]
            ax.scatter([CONDS[n][0]] * len(v), v, s=16, color=INK_2, zorder=4)
        ax.set_title(title, fontsize=16, color=INK, loc="left", pad=8)
        ax.set_xticks(x, ["0", "25", "33", "50", "75", "100"])
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Gemini 답 비율 (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False, fontsize=15)
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    save(fig, "38_그림8-5_8.1 결과 - 문체 특징의 변화.png")


def fig_complete(n_ok):
    """그림 8-6. 학생별 잘린 답의 수(400개 중, 최대 길이 1,024토큰에 닿아 끝맺지 못한 답). 막대 = seed 3개 평균, 점 = seed별."""
    order = ["gpt", "claude"] + list(CONDS)
    labels = ["GPT\n순수", "Claude\n순수"] + [CONDS[n][1] for n in CONDS]
    color = [COLORS["gpt"], COLORS["claude"]] + ["#9fdcc4", "#7fd0b0", "#5fc49c", "#3fb98a", "#27b07e", COLORS["gemini"]]
    cut = {k: 400 - v for k, v in n_ok.items()}
    fig, ax = plt.subplots(figsize=(12, 6.4))
    x = np.arange(len(order))
    x = np.where(x >= 2, x + 0.6, x)
    means = [np.mean([cut[(s, seed)] for seed in SEEDS]) for s in order]
    ax.bar(x, means, width=0.7, color=color, edgecolor=SURFACE, lw=2)
    mk = {42: "o", 43: "s", 44: "^"}
    for seed in SEEDS:
        ax.scatter(x, [cut[(s, seed)] for s in order], marker=mk[seed], s=70, color=INK, zorder=3,
                   label=f"seed {seed}", edgecolor=SURFACE, lw=1)
    for xx, m, s in zip(x, means, order):
        top = max(cut[(s, seed)] for seed in SEEDS)
        ax.text(xx, max(m, top) + 4, f"{m:.0f}", ha="center", va="bottom", fontsize=16, fontweight="bold", color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 140)
    ax.set_ylabel("잘린 답의 수 (400개 중)")
    ax.text(np.mean(x[2:]), -0.12, "Gemini 답 비율", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)
    style(ax)
    ax.tick_params(axis="x", length=0)
    save(fig, "39_그림8-6_8.1 결과 - 학생 답의 완결성.png")


def fig_common(common):
    """그림 8-7. 평가 질문 후보: 조건별·전체 공통 완전 질문 수와 그중 기존 평가 질문 100개에 속한 수."""
    keys = list(common)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    y = np.arange(len(keys))[::-1]
    kept = np.array([common[k]["kept_old"] for k in keys])
    new = np.array([common[k]["n"] - common[k]["kept_old"] for k in keys])
    ax.barh(y, kept, height=0.66, color=INK_2, edgecolor=SURFACE, lw=2, label="기존 평가 질문 100개에 속함")
    ax.barh(y, new, left=kept, height=0.66, color="#c3c2b7", edgecolor=SURFACE, lw=2, label="새로 쓸 수 있는 질문")
    for yy, a, b in zip(y, kept, new):
        ax.text(a / 2, yy, f"{a}", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        ax.text(a + b + 3, yy, f"{a + b}", ha="left", va="center", color=INK, fontsize=15, fontweight="bold")
    ax.axvline(100, color=INK, ls="--", lw=1.6)
    ax.set_yticks(y, [common[k]["label"] for k in keys])
    ax.set_xlim(0, 230)
    ax.set_xlabel("공통으로 끝맺은 질문 수 (평가 질문 400개 중)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    style(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    save(fig, "40_그림8-7_8.1 결과 - 평가 질문 후보 수.png")


def main():
    OUT.mkdir(exist_ok=True)
    setup()
    rng = np.random.default_rng(42)
    summary = json.loads((ROOT / "sft" / "data" / "mix_summary.json").read_text())
    answers = load_answers()

    # 1. 교차 loss
    cl = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "sft" / "runs").glob("gemini_mix*_seed*/cross_loss.csv"))]
                   + [pd.read_csv(ROOT / "sft" / "runs" / "cross_loss.csv")])
    cl = cl[cl["seed"].isin(SEEDS) & cl["student_teacher"].isin(list(CONDS) + list(PURE))]
    cl_tab = cl.pivot_table(index="student_teacher", columns="eval_teacher", values="loss", aggfunc=["mean", "std"])
    cl_tab.to_csv(OUT / "mix_cross_loss.csv")
    gem = cl[cl["eval_teacher"] == "gemini"].groupby("student_teacher")["loss"].mean()
    gap = np.array([(gem["gemini_mix00"] - gem[n]) / (gem["gemini_mix00"] - gem["gemini"]) * 100 for n in CONDS])

    # 2·3. 분류기와 문체 특징 (완전한 답)
    df = answer_table(answers)
    df.to_csv(OUT / "mix_answer_features.csv", index=False)
    share = df.groupby("student")["pred"].value_counts(normalize=True).unstack(fill_value=0)
    cls = np.array([share.loc[n, "gemini"] * 100 for n in CONDS])
    df["is_gem"] = (df["pred"] == "gemini").astype(float)
    cls_ci = np.array([boot_ci(df[df["student"] == n], "is_gem", rng) * 100 for n in CONDS])
    style_cols = [c for c, _, _ in STYLE_PANELS]
    style_tab = df.groupby("student")[style_cols[:-1]].mean()
    style_tab["chars_median"] = df.groupby("student")["chars"].median()
    style_tab = style_tab.join(share.add_prefix("pred_")).join(df.groupby("student")[["p_gpt", "p_claude", "p_gemini"]].mean())
    style_tab["n_complete"] = df.groupby("student").size()
    style_tab.loc[list(PURE) + list(CONDS)].to_csv(OUT / "mix_style.csv")
    seed_share = df.groupby(["student", "seed"])["is_gem"].mean().unstack()

    # 4. 완결성과 평가 질문 후보
    ok = {k: {r["id"] for r in v if complete(r)} for k, v in answers.items()}
    n_ok = {k: len(v) for k, v in ok.items()}
    old = set(json.loads((ROOT / "judge" / "data" / "selection.json").read_text())["selected_question_ids"])
    base = set.intersection(*[ok[(t, s)] for t in PURE for s in SEEDS])
    common = {"pure": {"label": "GPT·Claude 순수만", "ids": base}}
    for n in CONDS:
        common[n] = {"label": f"+ Gemini {CONDS[n][1].replace(chr(10), ' ')}", "ids": base & set.intersection(*[ok[(n, s)] for s in SEEDS])}
    common["all"] = {"label": "여섯 조건 모두", "ids": set.intersection(*[common[n]["ids"] for n in CONDS])}
    for k, v in common.items():
        v["n"], v["kept_old"] = len(v["ids"]), len(v["ids"] & old)
    per_seed = {s: set.intersection(*[ok[(n, s)] for n in list(PURE) + list(CONDS)]) for s in SEEDS}

    out = {
        "cross_loss_gemini_gap_closed_pct": dict(zip(CONDS, gap.round(1).tolist())),
        "classified_as_gemini_pct": {n: [round(cls[i], 1), *cls_ci[i].round(1).tolist()] for i, n in enumerate(CONDS)},
        "classified_as_gemini_by_seed_pct": (seed_share * 100).round(1).to_dict(orient="index"),
        "n_complete": {f"{s}_seed{seed}": n for (s, seed), n in n_ok.items()},
        "common_complete": {k: {"n": v["n"], "kept_old": v["kept_old"]} for k, v in common.items()},
        "common_complete_per_seed_all_conditions": {s: {"n": len(v), "kept_old": len(v & old)} for s, v in per_seed.items()},
    }
    (OUT / "mix_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(cl_tab.round(4).to_string())
    print(style_tab.round(3).T.to_string())

    fig_data(summary)
    fig_cross_loss(cl)
    fig_classifier(share)
    fig_shift(gap, cls, cls_ci)
    fig_style(df)
    fig_complete(n_ok)
    fig_common(common)


if __name__ == "__main__":
    main()
