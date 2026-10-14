"""학생 SFT 결과(교사 3개 × seed 5개)를 요약하고 그림으로 만든다.

입력: sft/runs/{teacher}_seed{n}/(train_loss.csv, val_loss.csv, run_info.json), sft/runs/cross_loss.csv
출력(results/output/):
  sft_loss_curves.png   교사별 train/val loss 곡선. 굵은 선은 seed 5개 평균, 옅은 띠는 최소~최대
  sft_cross_loss.png    학생 × 평가한 교사 답변의 loss와 학습 전 원본 모델 대비 감소량(seed 평균)
  sft_summary.json      표에 쓰는 수치(교사별 val loss 평균·표준편차, 교차 loss 행렬 등)

원본 모델(base)의 교차 loss는 각 교사 run의 학습 전(step 0) val loss를 쓴다. 학습 전 모델은 원본 그대로이므로
원본 모델이 그 교사의 검증 답변에서 내는 loss와 같다(5개 seed 모두 같은 값).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "sft" / "runs"
OUT = Path(__file__).parent / "output"
TEACHERS = ["gpt", "claude", "gemini"]
NAMES = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}
SEEDS = [42, 43, 44, 45, 46]
COLORS = {"gpt": "#2a78d6", "claude": "#eb6834", "gemini": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FONT = "Apple SD Gothic Neo"
STEPS_PER_EPOCH = 63


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def load_curves():
    train, val = {}, {}
    for t in TEACHERS:
        tr = [pd.read_csv(RUNS / f"{t}_seed{s}" / "train_loss.csv").set_index("step")["loss"] for s in SEEDS]
        va = [pd.read_csv(RUNS / f"{t}_seed{s}" / "val_loss.csv").drop_duplicates("step").set_index("step")["eval_loss"] for s in SEEDS]
        train[t], val[t] = pd.concat(tr, axis=1), pd.concat(va, axis=1)
    return train, val


def plot_curves(train, val, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300, sharey=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, data, title in ((axes[0], train, "train loss (매 step)"), (axes[1], val, "val loss (학습에 안 쓴 199문항, 10 step마다)")):
        for t in TEACHERS:
            d = data[t]
            if data is train:
                d = d.rolling(5, min_periods=1).mean()   # 매 step 값은 배치마다 출렁여 5 step 이동평균으로 보여 준다
            ax.fill_between(d.index, d.min(axis=1), d.max(axis=1), color=COLORS[t], alpha=0.18, lw=0)
            ax.plot(d.index, d.mean(axis=1), color=COLORS[t], lw=2, label=NAMES[t],
                    marker="o" if data is val else None, ms=3)
        for e in (1, 2):
            ax.axvline(e * STEPS_PER_EPOCH, color=GRID, lw=1)
            ax.text(e * STEPS_PER_EPOCH, ax.get_ylim()[1], f" {e} epoch", fontsize=7, color=INK_2, va="top")
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("학습 step (전체 189 = 3 epoch)", fontsize=8.5, color=INK_2)
        style(ax)
    axes[0].set_ylabel("cross-entropy loss (답변 토큰 평균)", fontsize=8.5, color=INK_2)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    for t in TEACHERS:   # val 최저점 표시
        m = val[t].mean(axis=1)
        axes[1].plot(m.idxmin(), m.min(), "v", color=COLORS[t], ms=7, markeredgecolor=SURFACE)
    fig.suptitle("학생 SFT loss · 선은 seed 5개 평균, 띠는 최소~최대, ▼는 val loss 최저점", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def cross_matrix(val):
    c = pd.read_csv(RUNS / "cross_loss.csv")
    c = c[c.student_teacher.isin(TEACHERS)]
    mean = c.groupby(["student_teacher", "eval_teacher"])["loss"].mean().unstack().loc[TEACHERS, TEACHERS]
    sd = c.groupby(["student_teacher", "eval_teacher"])["loss"].std().unstack().loc[TEACHERS, TEACHERS]
    base = pd.Series({t: float(val[t].loc[0].mean()) for t in TEACHERS})   # 학습 전 val loss = 원본 모델
    wins = sum(c[(c.student_teacher == t) & (c.seed == s)].set_index("eval_teacher")["loss"].idxmin() == t
               for t in TEACHERS for s in SEEDS)
    return mean, sd, base, wins


def plot_cross(mean, base, path):
    drop = base.values[None, :] - mean.values          # 원본 대비 loss 감소량(클수록 그 교사 답변을 더 잘 예측)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=300, gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(SURFACE)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    table = np.vstack([base.values[None, :], mean.values])
    labels_y = ["원본 모델"] + [f"{NAMES[t]} 학생" for t in TEACHERS]
    for ax, data, title, fmt, rows in (
            (axes[0], table, "loss (낮을수록 자연스럽게 예측)", "{:.3f}", labels_y),
            (axes[1], drop, "원본 모델 대비 loss 감소량 (클수록 흡수)", "{:+.3f}", labels_y[1:])):
        shown = -data if data is table else data             # 왼쪽은 낮을수록 진하게
        ax.imshow(shown, cmap=cmap, aspect="auto")
        lo, hi = shown.min(), shown.max()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                diag = (data is drop and i == j) or (data is table and i >= 1 and i - 1 == j)
                dark = (shown[i, j] - lo) / (hi - lo + 1e-9) > 0.55
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=10,
                        fontweight="bold" if diag else "normal", color="white" if dark else INK)
        ax.set_xticks(range(3), [f"{NAMES[t]} 답변" for t in TEACHERS], fontsize=8.5, color=INK)
        ax.set_yticks(range(len(rows)), rows, fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, pad=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("교차 loss: 학생이 세 교사의 검증 답변(각 199문항)을 얼마나 자연스럽게 예측하는가 · seed 5개 평균",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = FONT
    plt.rcParams["axes.unicode_minus"] = False
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = load_curves()
    plot_curves(train, val, OUT / "sft_loss_curves.png")
    mean, sd, base, wins = cross_matrix(val)
    plot_cross(mean, base, OUT / "sft_cross_loss.png")

    summary = {"teachers": {}, "cross_loss_mean": mean.round(4).to_dict(), "cross_loss_sd": sd.round(4).to_dict(),
               "base_loss": base.round(4).to_dict(), "self_lowest_students": f"{wins}/15"}
    for t in TEACHERS:
        infos = [json.loads((RUNS / f"{t}_seed{s}" / "run_info.json").read_text()) for s in SEEDS]
        final = [i["final_val_loss"] for i in infos]
        vm = val[t].mean(axis=1)
        summary["teachers"][t] = {
            "val_before": round(float(val[t].loc[0].mean()), 4),
            "val_min_mean": round(float(vm.min()), 4), "val_min_step": int(vm.idxmin()),
            "val_final_mean": round(float(np.mean(final)), 4), "val_final_sd": round(float(np.std(final, ddof=1)), 4),
            "train_last10_mean": round(float(train[t].iloc[-10:].mean().mean()), 4),
            "runtime_min_mean": round(float(np.mean([i["train_runtime_sec"] for i in infos])) / 60, 2),
            "answer_tokens": infos[0]["data"]["train"]["answer_tokens"],
        }
    (OUT / "sft_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
