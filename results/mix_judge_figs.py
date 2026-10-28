"""8.1 데이터 비율 실험의 평가 결과 그림(포스터 그림 8-8 ~ 8-13). 설명 문구는 그림에 넣지 않고 poster.html에 쓴다.

입력: judge/output/mix/analysis/ (judge/analyze_mix.py), results/output/mix_summary.json (results/mix_analysis.py)
출력: description/figures/41~46_그림8-8~8-13_*.png, 55_그림8-P_*.png(포스터용 요약 두 장)

사용법: python results/mix_judge_figs.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "judge" / "output" / "mix" / "analysis"
STYLE = Path(__file__).parent / "output" / "mix_summary.json"
MAIN = ROOT / "judge" / "output" / "analysis" / "models_summary.json"   # 본 실험(6절) 결과
FIG = ROOT / "description" / "figures"
CONDS = ["gemini_mix00", "gemini_mix25", "gemini_mix33", "gemini_mix50", "gemini_mix75", "gemini"]
X = {"gemini_mix00": 0, "gemini_mix25": 25, "gemini_mix33": 33.4, "gemini_mix50": 50, "gemini_mix75": 75, "gemini": 100}
XLAB = ["0%", "25%", "33%\n(균등)", "50%", "75%", "100%"]
JUDGES = ["gpt", "claude", "gemini"]
NAME = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}
CRIT = [("overall", "최종 순위"), ("factuality", "사실 정확성"), ("comprehension", "질문 충족도"),
        ("logical_robustness", "논리성"), ("readability", "가독성")]

COLORS = {"gpt": "#2a78d6", "claude": "#eb6834", "gemini": "#1baf7a"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#ffffff"
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
    if grid == "both":
        ax.grid(True, color=GRID, lw=0.8)
    elif grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ratio_axis(ax):
    ax.set_xticks([X[c] for c in CONDS], XLAB)
    ax.set_xlim(-6, 106)
    ax.set_xlabel("Gemini 계열 학생의 학습 데이터 중 Gemini 답 비율")


def err(v, lo, hi):
    """errorbar용 [아래, 위] 길이. 점추정이 부트스트랩 구간 밖에 놓이는 경우 0으로 자른다."""
    v, lo, hi = (np.asarray(a, float) for a in (v, lo, hi))
    return [np.maximum(v - lo, 0), np.maximum(hi - v, 0)]


MARK = {2: "*", 1: "†", 0: ""}          # 유의 수준: 2 = Holm 보정 후에도 유의, 1 = 보정 전에만 유의(95% 신뢰구간이 0 제외), 0 = 유의하지 않음
EDGE = {2: "-", 1: (0, (3, 1.6))}        # 막대 테두리: 실선 = 보정 후 유의, 점선 = 보정 전에만 유의


def sig_levels(lo, hi, p_holm):
    unc = (np.asarray(lo) > 0) | (np.asarray(hi) < 0)
    return np.where(np.asarray(p_holm) < 0.05, 2, np.where(unc, 1, 0))


def draw_sig_edges(ax, x, h, lev, width):
    for k in (2, 1):
        m = np.asarray(lev) == k
        if m.any():
            ax.bar(np.asarray(x)[m], np.asarray(h)[m], width=width, color="none", edgecolor=INK, lw=3, ls=EDGE[k])


def sig_note(fig, y=0.03):
    """그림 아래 범례: 보정 전·후 유의 표시."""
    h = [Patch(facecolor="white", edgecolor=INK, lw=3, ls=EDGE[k]) for k in (2, 1)]
    fig.legend(h, ["*  Holm 보정 후에도 유의", "†  보정 전에만 유의 (95% 신뢰구간이 0을 포함하지 않음)"], loc="upper center",
               bbox_to_anchor=(0.5, y), ncol=2, frameon=False, fontsize=25, handlelength=2.2, handleheight=1.1, columnspacing=3)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("그림:", name)


def fig_raw(desc):
    """그림 8-8. 평가자별로 본 Gemini 계열 학생의 쌍대 승률(최종 순위)."""
    d = desc[desc.teacher == "gemini"]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.axhline(50, color=MUTED, ls="--", lw=1.6)
    for j in JUDGES:
        v = d[d.judge == j].set_index("condition").loc[CONDS, "win_rate"]
        ax.plot([X[c] for c in CONDS], v, "-o", color=COLORS[j], lw=2.8 if j == "gemini" else 2.2,
                ms=11 if j == "gemini" else 9, mec=SURFACE, mew=2, label=f"{NAME[j]} 평가자", zorder=3 if j == "gemini" else 2)
        if j == "gemini":
            for c, y in zip(CONDS, v):
                ax.annotate(f"{y:.0f}", (X[c], y), xytext=(0, 12), textcoords="offset points", ha="center",
                            fontsize=14, color=INK, fontweight="bold")
    ratio_axis(ax)
    ax.set_ylabel("Gemini 계열 학생의 쌍대 승률 (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    style(ax)
    save(fig, "41_그림8-8_8.1 평가 결과 - 평가자별 Gemini 계열 학생 승률.png")


def fig_lambda(lam, seed):
    """그림 8-9. 조건별 Gemini 평가자의 선호 누출(%p, 순위 로짓 λ의 승률 환산), 95% CI와 seed별 값."""
    g = lam[(lam.criterion == "overall") & (lam.judge == "gemini")].set_index("condition").loc[CONDS]
    s = seed[seed.judge == "gemini"]
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.axhline(0, color=INK_2, lw=1.4)
    x = np.array([X[c] for c in CONDS])
    for k, sd in enumerate(sorted(s.seed.unique())):
        v = s[s.seed == sd].set_index("condition").loc[CONDS, "pp"]
        ax.scatter(x + (k - 1) * 1.6, v, s=40, marker="os^"[k], color=MUTED, zorder=2, label=f"seed {sd}")
    ax.errorbar(x, g.pp, yerr=err(g.pp, g.pp_ci_low, g.pp_ci_high), fmt="-o", color=COLORS["gemini"], lw=3,
                ms=13, mec=SURFACE, mew=2, capsize=7, elinewidth=2.2, zorder=4, label="전체 (95% 신뢰구간)")
    for xx, v, hi in zip(x, g.pp, g.pp_ci_high):
        ax.annotate(f"{v:+.1f}", (xx, hi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=15,
                    fontweight="bold", color=INK)
    ratio_axis(ax)
    ax.set_ylabel("Gemini 평가자의 선호 누출 (%p)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False)
    style(ax)
    save(fig, "42_그림8-9_8.1 평가 결과 - Gemini 비율에 따른 선호 누출.png")


def fig_control(lam):
    """그림 8-10. 같은 묶음에서 추정한 GPT·Claude 평가자의 자기 계열(순수 학생) 선호."""
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.axhline(0, color=INK_2, lw=1.4)
    for k, j in enumerate(["gpt", "claude"]):
        g = lam[(lam.criterion == "overall") & (lam.judge == j)].set_index("condition").loc[CONDS]
        x = np.array([X[c] for c in CONDS]) + (k - 0.5) * 2.2
        ax.errorbar(x, g.pp, yerr=err(g.pp, g.pp_ci_low, g.pp_ci_high), fmt="-o", color=COLORS[j], lw=2.6, ms=11,
                    mec=SURFACE, mew=2, capsize=6, elinewidth=2, label=f"{NAME[j]} 평가자 → {NAME[j]} 순수 학생")
    ratio_axis(ax)
    ax.set_ylabel("자기 계열 선호 (%p)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    style(ax)
    save(fig, "43_그림8-10_8.1 평가 결과 - GPT·Claude 평가자의 자기 계열 선호.png")


def fig_hypotheses(lam, models, sty):
    """그림 8-11. 네 가지 가설(통합 모형 적합값)이 예측한 누출과 실제 조건별 누출 비교(%p)."""
    x = np.array([X[c] for c in CONDS])
    g = lam[(lam.criterion == "overall") & (lam.judge == "gemini")].set_index("condition").loc[CONDS]
    m = models.set_index("model")["lambda"]
    dose = {"M_const": [1] * len(CONDS), "M_lin": [X[c] / 100 for c in CONDS],
            "M_loss": [sty["cross_loss_gemini_gap_closed_pct"][c] / 100 for c in CONDS],
            "M_cls": [sty["classified_as_gemini_pct"][c][0] / 100 for c in CONDS]}
    look = {"M_const": ("비율과 무관 (상수)", MUTED, ":"), "M_lin": ("Gemini 비율에 비례", INK_2, "--"),
            "M_loss": ("교차 loss 이동에 비례", INK, "-"), "M_cls": ("분류기 판정 비율에 비례", COLORS["gemini"], "-")}
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.axhline(0, color=INK_2, lw=1.2)
    for k, (lab, col, ls) in look.items():
        pred = (1 / (1 + np.exp(-m[k] * np.array(dose[k]))) - 0.5) * 100
        ax.plot(x, pred, ls=ls, color=col, lw=2.6, label=f"예측: {lab}")
    ax.errorbar(x, g.pp, yerr=err(g.pp, g.pp_ci_low, g.pp_ci_high), fmt="D", color="#4a3aa7", ms=12, mec=SURFACE,
                mew=2, capsize=7, elinewidth=2.2, zorder=4, label="실제 추정 (95% 신뢰구간)")
    ratio_axis(ax)
    ax.set_ylabel("Gemini 평가자의 선호 누출 (%p)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left", frameon=False, fontsize=13.5)
    style(ax)
    save(fig, "44_그림8-11_8.1 평가 결과 - 누출 모양 가설과 실제 비교.png")


def fig_models(models):
    """그림 8-12. 여섯 조건 통합 모형 비교: 누출 없음(M0) 대비 AIC 감소량(클수록 자료를 잘 설명)."""
    order = [("M_const", "비율과 무관 (상수)"), ("M_lin", "Gemini 비율에 비례"), ("M_loss", "교차 loss 이동에 비례"),
             ("M_cls", "분류기 판정 비율에 비례"), ("M_sep", "비율마다 따로 (6개)")]
    m = models.set_index("model")
    v = np.array([-m.loc[k, "daic_vs_M0"] for k, _ in order])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(order))[::-1]
    best = int(np.argmax(v))
    ax.barh(y, v, height=0.62, color=["#4a3aa7" if i == best else "#a8a69e" for i in range(len(v))], edgecolor=SURFACE, lw=2)
    for yy, vv in zip(y, v):
        ax.text(vv + max(v) * 0.01, yy, f"{vv:.1f}", va="center", ha="left", fontsize=15, fontweight="bold", color=INK)
    ax.set_yticks(y, [lab for _, lab in order])
    ax.set_xlim(min(0, v.min() * 1.1), max(v) * 1.15)
    ax.axvline(0, color=INK_2, lw=1.2)
    ax.set_xlabel("누출 없음 모형 대비 AIC 감소량 (클수록 잘 맞음)")
    style(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    save(fig, "45_그림8-12_8.1 평가 결과 - 누출 모양 모형 비교.png")


def fig_criteria(lam):
    """그림 8-13. 평가 기준별 Gemini 평가자의 선호 누출(%p). 굵은 칸 = 95% 신뢰구간이 0을 포함하지 않음."""
    g = lam[lam.judge == "gemini"]
    M = np.array([[g[(g.criterion == k) & (g.condition == c)].pp.iloc[0] for c in CONDS] for k, _ in CRIT])
    sig = np.array([[(lambda r: r.pp_ci_low.iloc[0] > 0 or r.pp_ci_high.iloc[0] < 0)(g[(g.criterion == k) & (g.condition == c)])
                     for c in CONDS] for k, _ in CRIT])
    lim = np.ceil(np.abs(M).max() / 2) * 2
    fig, ax = plt.subplots(figsize=(12, 5.6))
    cmap = plt.get_cmap("PuOr_r")
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            ax.text(k, i, f"{M[i, k]:+.1f}", ha="center", va="center", fontsize=16,
                    fontweight="bold" if sig[i, k] else "normal", color="white" if abs(M[i, k]) > lim * 0.55 else INK)
    ax.set_xticks(range(len(CONDS)), [s.replace("\n", " ") for s in XLAB])
    ax.set_yticks(range(len(CRIT)), [lab for _, lab in CRIT])
    ax.set_xlabel("Gemini 답 비율")
    ax.tick_params(length=0, colors=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(CONDS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CRIT)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("선호 누출 (%p)", color=INK_2)
    cb.outline.set_visible(False)
    save(fig, "46_그림8-13_8.1 평가 결과 - 평가 기준별 선호 누출.png")


SHADES = ["#c9ece0", "#a4dfca", "#7fd2b4", "#55c49c", "#2fb784", COLORS["gemini"]]   # 포스터: 진할수록 Gemini 비율이 높다


def draw_ratio_bars(ax, lam):
    """포스터 ① 패널: 비율별 Gemini 평가자의 선호 누출 막대 + 95% 신뢰구간. 막대 6개를 Holm 보정해
    보정 후에도 유의 = 실선 테두리·*, 보정 전에만 유의 = 점선 테두리·†(sig_note 범례).
    100%는 본 실험(6절)의 Gemini 학생 그대로이므로 본 실험 추정치(+9.8%p)를 쓴다. rw_judge_figs.py도 이 함수를 쓴다."""
    g = lam[(lam.criterion == "overall") & (lam.judge == "gemini")].set_index("condition").loc[CONDS].copy()
    main = next(r for r in json.loads(MAIN.read_text(encoding="utf-8"))["overall"] if r["model"] == "PL" and r["judge"] == "gemini")
    g.loc["gemini", ["pp", "pp_ci_low", "pp_ci_high", "p"]] = main["pp"], main["pp_ci_low"], main["pp_ci_high"], main["p"]
    x = np.arange(len(CONDS))
    lev = sig_levels(g.pp_ci_low, g.pp_ci_high, multipletests(g.p, method="holm")[1])
    ax.bar(x, g.pp, width=0.66, color=SHADES)
    draw_sig_edges(ax, x, g.pp, lev, 0.66)
    ax.errorbar(x, g.pp, yerr=err(g.pp, g.pp_ci_low, g.pp_ci_high), fmt="none", ecolor=INK, elinewidth=2.4, capsize=9)
    for xx, v, hi, k in zip(x, g.pp, g.pp_ci_high, lev):
        ax.text(xx, hi + 0.45, f"{v:+.1f}" + MARK[k], ha="center", va="bottom", fontsize=29 if k else 26,
                fontweight="bold", color=INK if k else INK_2)
    ax.axhline(0, color=INK, lw=1.8)
    ax.set_xticks(x, ["0%", "25%", "33%", "50%", "75%", "100%"], fontsize=26)
    ax.set_ylim(-5, 15)
    ax.set_yticks(range(-4, 15, 2))
    ax.set_xlabel("Gemini 학생의 학습 데이터 중 Gemini 답 비율", fontsize=25, color=INK_2, labelpad=12)
    ax.set_ylabel("Gemini 평가자의 선호 누출 (%p)", fontsize=25, color=INK_2)
    style(ax)
    ax.tick_params(axis="x", length=0)
    return g, lev


def fig_poster(lam, models, sty):
    """그림 8-P(포스터용). ① 비율별 Gemini 평가자의 선호 누출 ② 학생 답의 Gemini다움(분류기 판정 비율)과 누출.
    점 색은 ①의 막대 색과 같아서 그림 안에 조건 이름을 따로 쓰지 않는다."""
    plt.rcParams.update({"font.size": 24, "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 24})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(29, 10.5), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.2})
    g, lev = draw_ratio_bars(ax1, lam)

    # ② 학생 답이 Gemini처럼 보이는 정도(분류기 판정 비율)와 누출: 점 = 비율 조건
    cls = np.array([sty["classified_as_gemini_pct"][c][0] for c in CONDS])
    ax2.axhline(0, color=INK, lw=1.8)
    ax2.scatter(cls, g.pp, s=[700 if k == 2 else 460 for k in lev], c=SHADES, edgecolors=INK, linewidths=2.2, zorder=4)
    ax2.set_xlim(-6, 108)
    ax2.set_ylim(-5, 15)
    ax2.set_yticks(range(-4, 15, 2))
    ax2.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax2.set_xlabel("학생 답을 Gemini 답으로 판정한 비율 (분류기)", fontsize=25, color=INK_2, labelpad=12)
    ax2.set_ylabel("Gemini 평가자의 선호 누출 (%p)", fontsize=25, color=INK_2)
    style(ax2, grid="both")

    fig.canvas.draw()
    top = ax1.get_position().y1 + 0.05
    for ax, t in ((ax1, "① Gemini 답 비율을 줄이면 선호 누출이 줄어든다"), (ax2, "② 학생 답이 Gemini처럼 보일수록 누출이 커진다")):
        fig.text(ax.get_position().x0 - 0.045, top, t, fontsize=31, fontweight="bold", color=INK)
    sig_note(fig)
    save(fig, "55_그림8-P_8.1 포스터용 - 데이터 비율과 선호 누출.png")


def main():
    setup()
    lam = pd.read_csv(AN / "mix_lambda.csv")
    seed = pd.read_csv(AN / "mix_seed.csv")
    diff = pd.read_csv(AN / "mix_diff.csv")
    models = pd.read_csv(AN / "mix_models.csv")
    desc = pd.read_csv(AN / "mix_desc.csv")
    sty = json.loads(STYLE.read_text(encoding="utf-8"))
    fig_raw(desc)
    fig_lambda(lam, seed)
    fig_control(lam)
    fig_hypotheses(lam, models, sty)
    fig_models(models)
    fig_criteria(lam)
    fig_poster(lam, models, sty)


if __name__ == "__main__":
    main()
