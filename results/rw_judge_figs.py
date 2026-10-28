"""8.2 문체 제거 실험의 평가 결과 그림(poster 그림 8-24 ~ 8-27)과 포스터용 8.1·8.2 합친 그림(8-R).
설명 문구는 그림에 넣지 않고 poster.html에 쓴다. 글자는 최소 23pt, 흰 배경.

입력: judge/output/rw/analysis/ (judge/analyze_rw.py), 8.1 결과 judge/output/mix/analysis/mix_lambda.csv
출력: description/figures/61~64_그림8-24~8-27_*.png, 65_그림8-R_*.png

사용법: python results/rw_judge_figs.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests

from mix_judge_figs import AN as MIX_AN, MAIN, MARK, draw_ratio_bars, draw_sig_edges, err, save, setup, sig_levels, sig_note, style

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
AN = ROOT / "judge" / "output" / "rw" / "analysis"
JUDGES = ["gpt", "claude", "gemini"]
NAMES = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}
COLORS = {"gpt": "#2a78d6", "claude": "#eb6834", "gemini": "#1baf7a"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
LIGHT = 0.35   # 재작성 학생 막대의 투명도(원래 학생 = 진한 막대)
CRIT = [("overall", "최종 순위"), ("factuality", "사실 정확성"), ("comprehension", "질문 충족도"),
        ("logical_robustness", "논리성"), ("readability", "가독성")]


def fonts():
    plt.rcParams.update({"font.size": 24, "axes.labelsize": 25, "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 24})


def legend_orig_rw(ax, loc="upper left", **kw):
    h = [plt.Rectangle((0, 0), 1, 1, color=INK_2), plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=LIGHT)]
    ax.legend(h, ["원래 학생", "재작성 학생"], loc=loc, frameon=False, fontsize=25, **kw)


def color_ticks(ax):
    for lab, j in zip(ax.get_xticklabels(), JUDGES):
        lab.set_color(COLORS[j])
        lab.set_fontweight("bold")


def draw_orig_rw(ax, lam, crit="overall", orig_main=False):
    """평가자별 원래 학생 vs 재작성 학생의 선호 누출(같은 질문 276묶음) 막대 + 95% 신뢰구간.
    조건마다 평가자 3명을 Holm 보정: 보정 후에도 유의 = 실선 테두리·*, 보정 전에만 유의 = 점선 테두리·†.
    orig_main=True면 원래 학생 막대를 본 실험 전체(seed 5개, 500묶음) 순위 로짓 값으로 바꾼다(포스터용, 왼쪽 패널의 100%와 같은 값)."""
    x = np.arange(3)
    for k, cond in enumerate(("orig", "rw")):
        g = lam[(lam.criterion == crit) & (lam.condition == cond)].set_index("judge").loc[JUDGES]
        if cond == "orig" and orig_main:
            main = {r["judge"]: r for r in json.loads(MAIN.read_text(encoding="utf-8"))["overall"] if r["model"] == "PL"}
            for j in JUDGES:
                g.loc[j, ["pp", "pp_ci_low", "pp_ci_high", "p"]] = (main[j]["pp"], main[j]["pp_ci_low"], main[j]["pp_ci_high"],
                                                                   main[j]["p"])
        xx = x + (k - 0.5) * 0.4
        lev = sig_levels(g.pp_ci_low, g.pp_ci_high, multipletests(g.p, method="holm")[1])
        ax.bar(xx, g.pp, width=0.38, color=[COLORS[j] for j in JUDGES], alpha=1 if cond == "orig" else LIGHT,
               edgecolor=[COLORS[j] for j in JUDGES], lw=2.5)
        # 유의한 막대는 8.1 그림(왼쪽 패널)처럼 검은 테두리(투명도 없이 겹쳐 그림)
        draw_sig_edges(ax, xx, g.pp, lev, 0.38)
        ax.errorbar(xx, g.pp, yerr=err(g.pp, g.pp_ci_low, g.pp_ci_high), fmt="none", ecolor=INK, elinewidth=2.2, capsize=8)
        for a, v, lo, hi, lv in zip(xx, g.pp, g.pp_ci_low, g.pp_ci_high, lev):
            y = hi + 0.45 if v >= 0 else lo - 0.45
            ax.text(a, y, f"{v:+.1f}" + MARK[lv], ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=27 if lv else 24, fontweight="bold", color=INK if lv else INK_2)
    ax.axhline(0, color=INK, lw=1.8)
    ax.set_xticks(x, [NAMES[j] + " 평가자" for j in JUDGES], fontsize=26)
    color_ticks(ax)
    style(ax)
    ax.tick_params(axis="x", length=0)


def fig_lambda(lam, diff):
    """그림 8-24. ① 평가자별 선호 누출: 원래 학생 vs 재작성 학생 ② 재작성 후 변화(재작성 − 원래)와 95% 신뢰구간."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.25})
    draw_orig_rw(ax1, lam)
    ax1.set_ylim(-7, 15)
    ax1.set_yticks(range(-6, 15, 2))
    ax1.set_ylabel("평가자의 선호 누출 (%p)")
    legend_orig_rw(ax1)
    d = diff[(diff.criterion == "overall") & ~diff.length_control].set_index("judge").loc[JUDGES]
    y = np.arange(3)[::-1]
    for yy, j in zip(y, JUDGES):
        r = d.loc[j]
        lv = int(sig_levels(r.diff_ci_low, r.diff_ci_high, r.p_holm))
        ax2.plot([r.diff_ci_low, r.diff_ci_high], [yy, yy], color=COLORS[j], lw=5, solid_capstyle="round")
        ax2.scatter(r.diff_pp, yy, s=520 if lv == 2 else 380, color=COLORS[j], edgecolors=INK, linewidths=2.2 if lv == 2 else 1, zorder=3)
        ax2.text(r.diff_ci_high + 0.5, yy, f"{r.diff_pp:+.1f}" + MARK[lv], va="center", fontsize=27 if lv else 24,
                 fontweight="bold", color=INK if lv else INK_2)
    ax2.axvline(0, color=INK, lw=1.8)
    ax2.set_yticks(y, [NAMES[j] + " 평가자" for j in JUDGES], fontsize=26)
    for lab, j in zip(ax2.get_yticklabels(), JUDGES):
        lab.set_color(COLORS[j])
        lab.set_fontweight("bold")
    ax2.set_xlim(-12, 10)
    ax2.set_ylim(-0.6, 2.6)
    ax2.set_xlabel("재작성 후 누출 변화 (%p, 재작성 - 원래)")
    style(ax2, grid="both")
    fig.canvas.draw()
    top = ax1.get_position().y1 + 0.05
    for ax, t in ((ax1, "① 평가자별 선호 누출"), (ax2, "② 재작성 후 누출 변화")):
        fig.text(ax.get_position().x0 - 0.04, top, t, fontsize=30, fontweight="bold", color=INK)
    sig_note(fig)
    save(fig, "61_그림8-24_8.2 평가 결과 - 원래 학생과 재작성 학생의 선호 누출.png")


def fig_seed(seed):
    """그림 8-25. seed별 선호 누출: 원래 학생 → 재작성 학생(같은 seed를 선으로 잇는다)."""
    fig, axes = plt.subplots(1, 3, figsize=(28, 9.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, j in zip(axes, JUDGES):
        s = seed[seed.judge == j].pivot(index="seed", columns="condition", values="pp")
        for sd, r in s.iterrows():
            ax.plot([0, 1], [r.orig, r.rw], color=COLORS[j], lw=3, alpha=0.8)
            ax.scatter([0], [r.orig], s=320, color=COLORS[j], edgecolors=INK, zorder=3)
            ax.scatter([1], [r.rw], s=320, color=COLORS[j], alpha=LIGHT + 0.15, edgecolors=INK, zorder=3)
        ys = s.rw.sort_values()   # seed 이름이 겹치지 않게 아래부터 최소 1.1%p 간격으로 놓는다
        last = -np.inf
        for sd, v in ys.items():
            last = max(v, last + 1.1)
            ax.text(1.08, last, f"seed {sd}", va="center", fontsize=23, color=INK_2)
        ax.axhline(0, color=INK, lw=1.6)
        ax.set_xticks([0, 1], ["원래 학생", "재작성 학생"], fontsize=25)
        ax.set_xlim(-0.3, 1.45)
        ax.set_title(f"{NAMES[j]} 평가자", fontsize=29, fontweight="bold", color=COLORS[j], pad=14)
        style(ax)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("평가자의 선호 누출 (%p)")
    axes[0].set_yticks(range(-4, 15, 2))
    save(fig, "62_그림8-25_8.2 평가 결과 - seed별 선호 누출.png")


def fig_criteria(lam, diff):
    """그림 8-26. Gemini 평가자의 기준별 선호 누출: 원래 학생 vs 재작성 학생."""
    fig, ax = plt.subplots(figsize=(28, 9.5))
    x = np.arange(len(CRIT))
    for k, cond in enumerate(("orig", "rw")):
        g = lam[(lam.condition == cond) & (lam.judge == "gemini")].set_index("criterion").loc[[c for c, _ in CRIT]]
        xx = x + (k - 0.5) * 0.4
        sig = (g.pp_ci_low > 0) | (g.pp_ci_high < 0)
        ax.bar(xx, g.pp, width=0.38, color=COLORS["gemini"], alpha=1 if cond == "orig" else LIGHT, edgecolor=COLORS["gemini"], lw=2.5)
        ax.errorbar(xx, g.pp, yerr=err(g.pp, g.pp_ci_low, g.pp_ci_high), fmt="none", ecolor=INK, elinewidth=2.2, capsize=8)
        for a, v, lo, hi, s_ in zip(xx, g.pp, g.pp_ci_low, g.pp_ci_high, sig):
            y = hi + 0.45 if v >= 0 else lo - 0.45
            ax.text(a, y, f"{v:+.1f}" + ("*" if s_ else ""), ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=26 if s_ else 24, fontweight="bold", color=INK if s_ else INK_2)
    ax.axhline(0, color=INK, lw=1.8)
    ax.set_xticks(x, [n for _, n in CRIT], fontsize=26)
    ax.set_ylim(-10, 17)
    ax.set_ylabel("Gemini 평가자의 선호 누출 (%p)")
    legend_orig_rw(ax, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    style(ax)
    ax.tick_params(axis="x", length=0)
    save(fig, "63_그림8-26_8.2 평가 결과 - Gemini 평가자의 기준별 선호 누출.png")


def fig_raw(desc):
    """그림 8-27. 원자료: 평가자별로 각 계열 학생이 받은 쌍대 승률(최종 순위), 원래 학생 vs 재작성 학생."""
    fig, axes = plt.subplots(1, 3, figsize=(28, 9.5), sharey=True, gridspec_kw={"wspace": 0.1})
    x = np.arange(3)
    for ax, j in zip(axes, JUDGES):
        for k, cond in enumerate(("orig", "rw")):
            g = desc[(desc.condition == cond) & (desc.judge == j)].set_index("teacher").loc[JUDGES]
            xx = x + (k - 0.5) * 0.4
            ax.bar(xx, g.win_rate, width=0.38, color=[COLORS[t] for t in JUDGES], alpha=1 if cond == "orig" else LIGHT,
                   edgecolor=[COLORS[t] for t in JUDGES], lw=2.5)
            for a, v in zip(xx, g.win_rate):
                ax.text(a, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=23, color=INK)
        ax.axhline(50, color=MUTED, ls="--", lw=2)
        ax.set_xticks(x, [NAMES[t] + "\n학생" for t in JUDGES], fontsize=24)
        color_ticks(ax)
        ax.set_ylim(30, 72)
        ax.set_title(f"{NAMES[j]} 평가자", fontsize=29, fontweight="bold", color=COLORS[j], pad=14)
        style(ax)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("쌍대 승률 (%)")
    legend_orig_rw(axes[2])
    save(fig, "64_그림8-27_8.2 평가 결과 - 평가자별 학생 승률.png")


def fig_poster(lam, mix_lam):
    """그림 8-R(포스터용). 왼쪽 = 8.1 데이터 비율 실험(그림 8-P의 ①), 오른쪽 = 8.2 문체 제거 실험(그림 8-24의 ①, 원래 학생은 본 실험 값)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(29, 10.5), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.2})
    draw_ratio_bars(ax1, mix_lam)
    draw_orig_rw(ax2, lam, orig_main=True)
    ax2.set_ylim(-7, 15)
    ax2.set_yticks(range(-6, 15, 2))
    ax2.set_ylabel("평가자의 선호 누출 (%p)", fontsize=25, color=INK_2)
    legend_orig_rw(ax2)
    fig.canvas.draw()
    top = ax1.get_position().y1 + 0.05
    for ax, t in ((ax1, "① 학습 데이터의 Gemini 답 비율 줄이기"), (ax2, "② 세 교사 답을 같은 문체로 재작성하기")):
        fig.text(ax.get_position().x0 - 0.045, top, t, fontsize=31, fontweight="bold", color=INK)
    sig_note(fig)
    save(fig, "65_그림8-R_포스터용 - 선호 누출 억제 두 방법.png")


def main():
    setup()
    fonts()
    lam = pd.read_csv(AN / "rw_lambda.csv")
    diff = pd.read_csv(AN / "rw_diff.csv")
    fig_lambda(lam, diff)
    fig_seed(pd.read_csv(AN / "rw_seed.csv"))
    fig_criteria(lam, diff)
    fig_raw(pd.read_csv(AN / "rw_desc.csv"))
    fig_poster(lam, pd.read_csv(MIX_AN / "mix_lambda.csv"))


if __name__ == "__main__":
    main()
