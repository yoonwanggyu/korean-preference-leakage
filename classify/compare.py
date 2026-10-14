"""로지스틱 회귀와 KLUE-RoBERTa의 test 정확도를 원문/정규화 조건별로 한 그림에 모은다.

logistic.py와 roberta.py(raw, norm)를 먼저 실행해야 한다.
출력: classify/output/compare_accuracy.png, compare_per_model.png(생성 모델별 변화), compare.md
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, GRID, INK, INK_2, OUT_DIR, SURFACE, setup_plot, style_axes

CONDS = [("raw", "원문"), ("norm", "정규화\n(마크다운·따옴표·줄바꿈 제거)")]


def spread(values, gap):
    """값 라벨이 겹치지 않도록, 가까운 값끼리 최소 gap만큼 위아래로 벌린 y 위치를 돌려준다."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if pos[b] - pos[a] < gap:
            pos[b] = pos[a] + gap
    return pos + (np.array(values) - pos).mean()  # 전체를 원래 중심으로 되돌린다


def plot_per_model(results, path):
    """분류기별 패널에서, 생성 모델마다 원문 → 정규화 재현율 변화를 선으로 잇는다(기울기 그래프)."""
    models = ["GPT", "Claude", "Gemini"]
    fig, axes = plt.subplots(1, len(results), figsize=(7.2, 3.8), dpi=300, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, res) in zip(axes, results.items()):
        rec = {m: [r["per_class"][m]["recall"] * 100 for r in res] for m in models}
        for m in models:
            ax.plot([0, 1], rec[m], color=COLORS[m], lw=2, marker="o", ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        for x, ha, dx in ((0, "right", -0.08), (1, "left", 0.08)):
            vals = [rec[m][x] for m in models]
            for m, y, yl in zip(models, vals, spread(vals, 1.3)):
                label = f"{y:.1f}" if x == 0 else f"{m} {y:.1f}"
                ax.text(x + dx, yl, label, ha=ha, va="center", fontsize=8.5, color=INK)
        drops = {m: rec[m][1] - rec[m][0] for m in models}
        worst = min(drops, key=drops.get)
        ax.set_title(f"{name}\n가장 큰 변화: {worst} {drops[worst]:+.1f}%p", fontsize=9.5, color=INK)
        ax.set_xticks([0, 1], ["원문", "정규화"], fontsize=9, color=INK)
        ax.set_xlim(-0.45, 1.6)
        style_axes(ax)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(84, 100.5)
    axes[0].set_ylabel("생성 모델별 test 정확도 (재현율, %)", fontsize=8.5, color=INK_2)
    fig.suptitle("겉모양(마크다운·따옴표·줄바꿈)을 지웠을 때 생성 모델별 식별 정확도 변화", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    setup_plot()
    lr = json.loads((OUT_DIR / "logistic" / "metrics.json").read_text(encoding="utf-8"))
    rb = {c: json.loads((OUT_DIR / "roberta" / f"metrics_{c}.json").read_text(encoding="utf-8")) for c, _ in CONDS}
    results = {
        "다항 로지스틱 회귀": [lr[c] for c, _ in CONDS],
        "KLUE-RoBERTa": [rb[c] for c, _ in CONDS],
    }
    # 분류기는 모델(GPT/Claude/Gemini)과 다른 범주라, 생성 모델 색과 겹치지 않게 잉크 계열 두 톤을 쓴다.
    colors = {"다항 로지스틱 회귀": INK_2, "KLUE-RoBERTa": INK}

    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    w = 0.36
    for k, (name, res) in enumerate(results.items()):
        xs = np.arange(len(CONDS)) + (k - 0.5) * w
        accs = np.array([r["test_acc"] for r in res]) * 100
        lo = accs - np.array([r["test_ci95"][0] for r in res]) * 100
        hi = np.array([r["test_ci95"][1] for r in res]) * 100 - accs
        ax.bar(xs, accs, width=w - 0.04, color=colors[name], label=name, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, accs, yerr=[lo, hi], fmt="none", ecolor=COLORS["GPT"], elinewidth=1.4, capsize=4)
        for x, a in zip(xs, accs):
            ax.text(x, a + 3.2, f"{a:.1f}%", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(100 / 3, color=INK_2, lw=1, ls="--")
    ax.text((len(CONDS) - 1) / 2, 100 / 3 + 1.5, "우연 33.3%", ha="center", fontsize=8, color=INK_2)
    ax.set_xlim(-0.65, len(CONDS) - 0.35)
    ax.set_xticks(range(len(CONDS)), [n for _, n in CONDS], fontsize=9, color=INK)
    ax.set_ylim(0, 106)
    ax.set_ylabel("test 정확도 (%)", fontsize=9, color=INK_2)
    ax.set_title("답변만으로 생성 모델(GPT/Claude/Gemini) 식별 · 파란 막대는 95% 신뢰구간", fontsize=9.5, color=INK, pad=28)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style_axes(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(OUT_DIR / "compare_accuracy.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    plot_per_model(results, OUT_DIR / "compare_per_model.png")

    lines = ["| 분류기 | 조건 | test 정확도 | 95% CI | GPT F1 | Claude F1 | Gemini F1 |", "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        for (c, _), r in zip(CONDS, res):
            f1 = [r["per_class"][m]["f1"] for m in ("GPT", "Claude", "Gemini")]
            lines.append(f"| {name} | {c} | {r['test_acc']:.1%} | {r['test_ci95'][0]:.1%}–{r['test_ci95'][1]:.1%} | "
                         + " | ".join(f"{x:.3f}" for x in f1) + " |")
    (OUT_DIR / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
