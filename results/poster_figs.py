"""포스터에 옮기기 좋게 다시 그린 요약 그림. 그림에는 소제목·축·범례만 두고 설명 문장은 poster.html에 쓴다.

그림 4-7: ① 세 교사의 형식·문체 특징(results/output/style_summary.csv)
          ② 답변만으로 생성 모델을 맞힌 test 정확도(classify/output/logistic/metrics.json, roberta/metrics_*.json)
그림 6-I: ① 평가자별 선호 누출(judge/output/analysis/model_results.csv, 최종 순위, 세 모형) ② seed별 순위 로짓 추정(model_by_seed.csv)
그림 5-C: ① 학생 val loss 곡선(sft/runs/{teacher}_seed{n}/val_loss.csv, seed 5개 평균)
          ② 교차 loss 행렬(sft/runs/cross_loss.csv, seed 5개 평균; 학습 전 Qwen = 각 교사 run의 step 0 val loss)

사용법: python results/poster_figs.py
출력: description/figures/05_그림4-7_4.8 결과 해석.png, 29_그림6-I_*.png, 13_그림5-C_5.4 학생이 자기 교사의 스타일을 흡수했는가 - 교차 loss.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLS = ROOT / "classify" / "output"
FIG = ROOT / "description" / "figures"
MODELS = ["GPT", "Claude", "Gemini"]
COLORS = {"GPT": "#2a78d6", "Claude": "#eb6834", "Gemini": "#1baf7a"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#ffffff"
ACCENT = "#4a3aa7"
# ① 막대로 그릴 특징: (style_summary.csv의 지표 이름, 그림 이름)
STYLE = [("굵게(**) 사용", "굵게"), ("글머리 목록(-, *) 사용", "글머리 목록"), ("번호 목록(1.) 사용", "번호 목록"),
         ("영문 괄호 병기", "영문 괄호"), ("요약형 마무리", "요약 마무리"), ("완곡 표현(~수 있습니다 등)", "완곡 표현")]
RUNS = [("logistic", "raw", "로지스틱 · 원문"), ("logistic", "norm", "로지스틱 · 정규화"),
        ("roberta", "raw", "RoBERTa · 원문"), ("roberta", "norm", "RoBERTa · 정규화")]


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=31, fontweight="bold", color=INK, pad=24)


def accuracy():
    log = json.loads((CLS / "logistic" / "metrics.json").read_text())
    out = {}
    for clf, cond, _ in RUNS:
        m = log[cond] if clf == "logistic" else json.loads((CLS / "roberta" / f"metrics_{cond}.json").read_text())
        out[(clf, cond)] = (m["test_acc"] * 100, m["test_ci95"][0] * 100, m["test_ci95"][1] * 100)
    return out


def fig_4_7():
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 24,
                         "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 24})
    summ = pd.read_csv(ROOT / "results" / "output" / "style_summary.csv").set_index("지표")
    acc = accuracy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(29, 11), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})

    # ① 형식·문체 특징
    x = np.arange(len(STYLE))
    w = 0.27
    for j, m in enumerate(MODELS):
        v = np.array([summ.loc[k, m] * 100 for k, _ in STYLE])
        xx = x + (j - 1) * w
        ax1.bar(xx, v, width=w - 0.02, color=COLORS[m], label=m, edgecolor=SURFACE, lw=1.5)
        for a, b in zip(xx, v):
            ax1.text(a, b + 1.2, f"{b:.0f}", ha="center", va="bottom", fontsize=23, color=INK)
    ax1.set_xticks(x, [lab.replace(" ", "\n") for _, lab in STYLE])
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("그 특징이 나타난 답변 비율 (%)", fontsize=24, color=INK_2)
    ax1.legend(loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.02))
    style(ax1)
    ax1.tick_params(axis="x", length=0)
    title(ax1, "① 세 교사의 답은 형식부터 다르다")

    # ② 정확도
    y = np.arange(len(RUNS))[::-1]
    v = np.array([acc[(c, k)][0] for c, k, _ in RUNS])
    lo = np.array([acc[(c, k)][1] for c, k, _ in RUNS])
    hi = np.array([acc[(c, k)][2] for c, k, _ in RUNS])
    ax2.barh(y, v, height=0.62, color=ACCENT, edgecolor=SURFACE, lw=1.5)
    ax2.errorbar(v, y, xerr=[v - lo, hi - v], fmt="none", ecolor=INK, elinewidth=1.8, capsize=7)
    for yy, a, b in zip(y, v, hi):
        ax2.text(b + 1.2, yy, f"{a:.1f}%", va="center", ha="left", fontsize=25, fontweight="bold", color=INK)
    ax2.axvline(100 / 3, color=MUTED, ls="--", lw=1.8)
    ax2.text(100 / 3 + 1, y[-1] - 0.42, "우연 33.3%", fontsize=23, color=MUTED, va="top")
    ax2.set_yticks(y, [lab for _, _, lab in RUNS])
    ax2.set_ylim(y[-1] - 0.85, y[0] + 0.5)
    ax2.set_xlim(0, 112)
    ax2.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax2.set_xlabel("test 정확도", fontsize=24, color=INK_2)
    style(ax2, grid="x")
    ax2.tick_params(axis="y", length=0)
    title(ax2, "② 답변만 보고 어느 모델이 썼는지 맞힌다")

    fig.patch.set_facecolor(SURFACE)
    name = "05_그림4-7_4.8 결과 해석.png"
    fig.savefig(FIG / name, dpi=250, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("그림:", name)
    return acc


def fig_5_c():
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 24,
                         "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 24})
    runs = ROOT / "sft" / "runs"
    teachers = ["gpt", "claude", "gemini"]
    seeds = [42, 43, 44, 45, 46]
    curves = {t: pd.concat([pd.read_csv(runs / f"{t}_seed{n}" / "val_loss.csv").drop_duplicates("step").set_index("step")["eval_loss"]
                            for n in seeds], axis=1).mean(axis=1) for t in teachers}
    cl = pd.read_csv(runs / "cross_loss.csv")
    cl = cl[cl.seed.isin(seeds)].groupby(["student_teacher", "eval_teacher"])["loss"].mean()
    M = np.array([[curves[e].iloc[0] for e in teachers]] + [[cl[(s, e)] for e in teachers] for s in teachers])
    rows = ["학습 전 Qwen"] + [f"{m} 학생" for m in MODELS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10), gridspec_kw={"width_ratios": [1.05, 1], "wspace": 0.3})

    # ① val loss 곡선
    for t, m in zip(teachers, MODELS):
        c = curves[t]
        ax1.plot(c.index, c.values, color=COLORS[m], lw=3.4)
        ax1.text(c.index[-1] + 3, c.values[-1] + {"gpt": 0, "claude": 0.012, "gemini": -0.012}[t], f"{m} 학생 {c.values[-1]:.3f}", va="center", fontsize=24,
                 color=COLORS[m], fontweight="bold")
    for e in (1, 2):
        ax1.axvline(e * 63, color=MUTED, ls="--", lw=1.6)
        ax1.text(e * 63 + 2, 2.665, f"{e} epoch", fontsize=23, color=MUTED, va="top")
    ax1.set_xlim(0, 270)
    ax1.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax1.set_ylim(2.1, 2.68)
    ax1.set_xlabel("학습 step", fontsize=24, color=INK_2)
    ax1.set_ylabel("자기 교사 검증 답의 loss (nats/토큰)", fontsize=24, color=INK_2)
    style(ax1)

    # ② 교차 loss 행렬 (진할수록 loss가 낮다)
    ax2.imshow(-M, cmap="Blues", vmin=-2.75, vmax=-2.05, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            own = i >= 1 and j == i - 1
            txt = f"{M[i, j]:.3f}" + (" ★" if own else "")
            ax2.text(j, i - 0.13, txt, ha="center", va="center", fontsize=29 if own else 27,
                     fontweight="bold" if own else "normal", color="white" if own else INK)
            ax2.text(j, i + 0.25, f"ppl {np.exp(M[i, j]):.1f}", ha="center", va="center", fontsize=23,
                     color="white" if own else INK_2)
            if own:
                ax2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=4))
    ax2.axhline(0.5, color="white", lw=8)
    ax2.set_xticks([])
    for j, m in enumerate(MODELS):   # 열 머리: 교사 색 글씨
        ax2.text(j, -0.62, f"{m} 교사의 답", ha="center", va="bottom", fontsize=25, fontweight="bold", color=COLORS[m])
    ax2.set_yticks(range(4), rows, fontsize=25)
    for lab, m in zip(ax2.get_yticklabels(), [None] + MODELS):
        lab.set_color(COLORS[m] if m else INK_2)
        lab.set_fontweight("bold")
    ax2.tick_params(length=0, pad=10)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_xlabel("칸 = 교차 loss (nats/토큰, 답 토큰 평균 교차 엔트로피) · ppl = e^loss", fontsize=23, color=INK_2, labelpad=14)
    # 두 소제목을 같은 높이에 둔다(② 위에 열 머리가 있어 set_title로는 높이가 달라진다)
    fig.canvas.draw()
    top = ax2.get_position().y1 + 0.085
    for ax, t in ((ax1, "① 학습이 잘 되었나"), (ax2, "② 학생은 자기 교사의 문체를 배웠나")):
        x0 = ax.get_position().x0 - (0.075 if ax is ax2 else 0)
        fig.text(x0, top, t, fontsize=31, fontweight="bold", color=INK, ha="left", va="bottom")

    fig.patch.set_facecolor(SURFACE)
    name = "13_그림5-C_5.4 학생이 자기 교사의 스타일을 흡수했는가 - 교차 loss.png"
    fig.savefig(FIG / name, dpi=250, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("그림:", name)
    return curves, M


def fig_6_i():
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 23,
                         "xtick.labelsize": 23, "ytick.labelsize": 25, "legend.fontsize": 23})
    an = ROOT / "judge" / "output" / "analysis"
    res = pd.read_csv(an / "model_results.csv")
    res = res[res.criterion == "overall"].set_index(["model", "judge"])
    seed = pd.read_csv(an / "model_by_seed.csv")
    seed = seed[(seed.criterion == "overall") & (seed.model == "PL")]
    judges = ["gpt", "claude", "gemini"]
    names = dict(zip(judges, MODELS))
    verdict = {"gemini": "뚜렷한 누출", "gpt": "약한 증거", "claude": "누출 없음"}
    y = {"gpt": 2, "claude": 1, "gemini": 0}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(29, 8.8), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.1})

    # ① 세 모형의 선호 누출: 주 모형(순위 로짓)은 크게, 나머지 두 모형은 작게(강건성). 두 모형은 표식(▲·■)과 선(실선·점선)으로 구분
    others = [("PL_top1", 0.24, "1위 모형", "^", "-"), ("LPM", -0.24, "쌍 고정효과 모형", "s", "--")]
    for j in judges:
        c, yy = COLORS[names[j]], y[j]
        ax1.axhspan(yy - 0.45, yy + 0.45, color=c, alpha=0.06, lw=0)
        for m, off, _, mk, ls in others:
            r = res.loc[(m, j)]
            ax1.plot([r.pp_ci_low, r.pp_ci_high], [yy + off] * 2, color=c, lw=2.4, ls=ls, alpha=0.7)
            ax1.plot(r.pp, yy + off, mk, ms=13, mfc="white", mec=c, mew=2.5)
        r = res.loc[("PL", j)]
        ax1.plot([r.pp_ci_low, r.pp_ci_high], [yy] * 2, color=c, lw=5, solid_capstyle="round")
        ax1.plot(r.pp, yy, "o", ms=19, color=c, mec="white", mew=2.5, zorder=5)
        # * = Holm 보정 후에도 유의, † = 보정 전에만 유의(95% 신뢰구간이 0 제외). 범례는 그림 아래
        mark = "*" if r.p_holm < 0.05 else "†" if r.pp_ci_low > 0 or r.pp_ci_high < 0 else ""
        ax1.text(16.0, yy, f"{r.pp:+.1f}%p{mark}", va="center", ha="left", fontsize=26, fontweight="bold", color=c)
        ax1.text(22.0, yy, verdict[j], va="center", ha="left", fontsize=25, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.35", fc=c, ec="none"))
    ax1.axvline(0, color=INK, lw=1.6)
    ax1.set_yticks([y[j] for j in judges], [f"{names[j]} 평가자" for j in judges], fontsize=24, fontweight="bold")
    for lab, j in zip(ax1.get_yticklabels(), judges):
        lab.set_color(COLORS[names[j]])
    ax1.set_ylim(-0.6, 2.6)
    ax1.set_xlim(-6, 29.5)
    ax1.set_xticks([-5, 0, 5, 10, 15], ["-5", "0", "+5", "+10", "+15"])
    ax1.set_xlabel("품질·위치가 같은 상대를 이길 확률의 증가 (%p)", fontsize=23, color=INK_2)
    h = [plt.Line2D([], [], marker="o", ms=15, color=INK_2, lw=4, label="순위 로짓 모형 (주 결과)")] + \
        [plt.Line2D([], [], marker=mk, ms=13, mfc="white", mec=INK_2, mew=2.5, color=INK_2, lw=2.4, ls=ls, label=lab)
         for _, _, lab, mk, ls in others]
    ax1.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)
    style(ax1, grid="x")
    ax1.tick_params(axis="y", length=0)
    ax1.spines["left"].set_visible(False)

    # ② seed 5개의 순위 로짓 추정
    for j in judges:
        c, yy = COLORS[names[j]], y[j]
        v = seed[seed.judge == j].pp.to_numpy()
        ax2.axhspan(yy - 0.45, yy + 0.45, color=c, alpha=0.06, lw=0)
        ax2.scatter(v, [yy] * len(v), s=260, color=c, edgecolor="white", lw=2, zorder=4)
        ax2.text(17.2, yy, f"{(v > 0).sum()}/5", va="center", ha="left", fontsize=26, fontweight="bold", color=c)
    ax2.axvline(0, color=INK, lw=1.6)
    ax2.set_yticks([y[j] for j in judges], ["", "", ""])
    ax2.set_ylim(-0.6, 2.6)
    ax2.set_xlim(-6, 16.5)
    ax2.set_xticks([-5, 0, 5, 10, 15], ["-5", "0", "+5", "+10", "+15"])
    ax2.set_xlabel("seed별 추정치 (%p) · 오른쪽 숫자 = 양수인 seed 수", fontsize=23, color=INK_2)
    style(ax2, grid="x")
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)

    fig.canvas.draw()
    top = ax1.get_position().y1 + 0.11
    for ax, t in ((ax1, "① 통계 모형으로 추정한 평가자별 선호 누출"), (ax2, "② 다섯 번의 독립 학습(seed)에서도 반복되나")):
        fig.text(ax.get_position().x0 - (0.075 if ax is ax1 else 0), top, t, fontsize=29, fontweight="bold", color=INK)
    fig.text(0.5, ax1.get_position().y0 - 0.125, "*  Holm 보정 후에도 유의          †  보정 전에만 유의 (95% 신뢰구간이 0을 포함하지 않음)",
             ha="center", va="top", fontsize=25, color=INK)
    fig.patch.set_facecolor(SURFACE)
    name = "29_그림6-I_6.5 결과 ③ 다섯 번의 독립 학습(seed)에서도 반복되는가.png"
    fig.savefig(FIG / name, dpi=250, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("그림:", name)


if __name__ == "__main__":
    acc = fig_4_7()
    for (c, k, lab) in RUNS:
        print(lab, f"정확도 {acc[(c, k)][0]:.1f}")
    fig_6_i()
    curves, M = fig_5_c()
    print("교차 loss 행렬 (행: 학습 전, GPT·Claude·Gemini 학생 / 열: GPT·Claude·Gemini 교사 답)")
    print(np.round(M, 3))
