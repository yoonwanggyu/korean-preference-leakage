"""다항 로지스틱 회귀로 답변의 생성 모델(GPT/Claude/Gemini)을 분류한다.

입력: 문자 2~4-gram TF-IDF (train split에만 fit)
하이퍼파라미터: 규제 강도 C를 val split 정확도로 고르고, 고른 모델을 test split에서 한 번만 평가한다.
raw(원문)와 norm(마크다운·따옴표·줄바꿈 제거) 두 조건을 각각 실험한다.

출력(classify/output/logistic/):
  metrics.json               조건별 val 탐색 결과, test 정확도와 95% CI, 클래스별 P/R/F1
  confusion_{raw,norm}.png   test 혼동행렬
  top_features_{raw,norm}.png 모델별로 가중치가 가장 큰 글자 조각
  val_curve.png              C에 따른 val 정확도
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from common import (COLORS, GRID, INK, INK_2, MODELS, OUT_DIR, SURFACE, bootstrap_ci, get_split, save_predictions, save_split,
                    load_dataset, plot_confusion, setup_plot, style_axes)

C_GRID = [0.25, 1, 4, 16, 64, 256]
NGRAM = (2, 4)
MIN_DF = 3
TOP_K = 10
OUT = OUT_DIR / "logistic"
COND_NAME = {"raw": "원문", "norm": "정규화 (마크다운·따옴표·줄바꿈 제거)"}


def run(rows, cond):
    x_tr, y_tr, _ = get_split(rows, "train", cond)
    x_va, y_va, _ = get_split(rows, "val", cond)
    x_te, y_te, g_te = get_split(rows, "test", cond)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM, min_df=MIN_DF, sublinear_tf=True)
    X_tr, X_va, X_te = vec.fit_transform(x_tr), vec.transform(x_va), vec.transform(x_te)

    val_acc = {}
    fitted = {}
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000).fit(X_tr, y_tr)  # 3개 클래스 → softmax(다항)
        val_acc[c] = float((clf.predict(X_va) == y_va).mean())
        fitted[c] = clf
    best_c = max(C_GRID, key=lambda c: val_acc[c])
    clf = fitted[best_c]

    pred = clf.predict(X_te)
    acc = float((pred == y_te).mean())
    lo, hi = bootstrap_ci(y_te, pred, g_te)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=range(len(MODELS)), zero_division=0)
    print(f"[{cond}] 특징 {X_tr.shape[1]:,}개, best C={best_c}, val {val_acc[best_c]:.3f}, "
          f"test {acc:.3f} (95% CI {lo:.3f}-{hi:.3f})")

    for split, X, y, g in (("val", X_va, y_va, get_split(rows, "val", cond)[2]), ("test", X_te, y_te, g_te)):
        save_predictions(g, y, clf.predict_proba(X), OUT / f"predictions_{cond}_{split}.csv")
    save_weights(clf, vec, OUT / f"weights_{cond}.csv")

    plot_confusion(y_te, pred,f"다항 로지스틱 · {COND_NAME[cond].split(' ')[0]}: test 정확도 {acc:.1%}",
                   OUT / f"confusion_{cond}.png")
    plot_top_features(clf, vec, cond, OUT / f"top_features_{cond}.png")

    return {
        "n_features": X_tr.shape[1],
        "n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te),
        "val_acc_by_C": {str(c): a for c, a in val_acc.items()},
        "best_C": best_c,
        "test_acc": acc, "test_ci95": [lo, hi],
        "per_class": {m: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
                      for i, m in enumerate(MODELS)},
    }


def save_weights(clf, vec, path, top=200):
    """모델별로 가중치가 큰 글자 조각 상위 top개를 CSV로 남긴다 (중복 제거 전 원본 순위)."""
    vocab = vec.get_feature_names_out()
    lines = ["model,rank,ngram,weight"]
    for k, m in enumerate(MODELS):
        for rank, i in enumerate(np.argsort(-clf.coef_[k])[:top], 1):
            lines.append(f'{m},{rank},"{vocab[i].replace(chr(34), chr(34) * 2)}",{clf.coef_[k][i]:.6f}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_top_features(clf, vec, cond, path):
    """모델마다 가중치가 가장 큰 글자 조각 TOP_K개. 공백은 _로 표시한다."""
    vocab = np.array(vec.get_feature_names_out())
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9, 3.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    for k, (ax, m) in enumerate(zip(axes, MODELS)):
        # "보통", " 보통", "보통 "처럼 공백만 다른 조각이나 서로 포함되는 조각은 가장 가중치가 큰 하나만 남긴다.
        order, seen = [], []
        for i in np.argsort(-clf.coef_[k]):
            key = vocab[i].strip()
            if key and not any(key in s or s in key for s in seen):
                order.append(i)
                seen.append(key)
            if len(order) == TOP_K:
                break
        labels = [vocab[i].replace(" ", "_").replace("\n", "⏎") for i in order]
        ax.barh(range(TOP_K), clf.coef_[k][order], color=COLORS[m], height=0.7)
        ax.set_yticks(range(TOP_K), labels, fontsize=8.5, color=INK)
        ax.invert_yaxis()
        ax.set_title(m, fontsize=10, color=INK, fontweight="bold")
        style_axes(ax)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlabel("로지스틱 회귀 가중치 (클수록 해당 모델 쪽 근거)", fontsize=8.5, color=INK_2)
    fig.suptitle(f"모델별로 가장 강한 근거가 된 글자 조각 · {COND_NAME[cond]}", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_val_curve(results, path):
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    styles = {"raw": dict(color=INK, ls="-"), "norm": dict(color=INK_2, ls="--")}
    for cond, res in results.items():
        accs = [res["val_acc_by_C"][str(c)] * 100 for c in C_GRID]
        ax.plot(range(len(C_GRID)), accs, marker="o", ms=5, lw=2, **styles[cond])
        ax.annotate(COND_NAME[cond].split(" ")[0], (len(C_GRID) - 1, accs[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=styles[cond]["color"])
        b = C_GRID.index(res["best_C"])
        ax.plot(b, accs[b], marker="o", ms=9, mfc="none", mec=styles[cond]["color"], mew=1.5)
    ax.set_xticks(range(len(C_GRID)), [str(c) for c in C_GRID])
    ax.set_xlabel("규제 강도 C (클수록 규제 약함)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("val 정확도 (%)", fontsize=8.5, color=INK_2)
    ax.set_title("C 선택: 속이 빈 원이 val 정확도가 가장 높은 값", fontsize=9, color=INK)
    style_axes(ax)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    setup_plot()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_dataset()
    save_split(rows)
    n_q = {s: len({r["id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")}
    print(f"답변 {len(rows)}개, 질문 수 {n_q}")

    results = {cond: run(rows, cond) for cond in ("raw", "norm")}
    plot_val_curve(results, OUT / "val_curve.png")
    (OUT / "metrics.json").write_text(json.dumps({"questions": n_q, **results}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
