"""8.2 문체 제거 실험: HCX-005 재작성 데이터로 학습한 학생 9명(gpt_rw·claude_rw·gemini_rw × seed 42~44)의
학습·답 생성 결과를 원래 학생(같은 seed)과 비교한다(평가자 채점 전 단계).

1. 교차 loss: 학생이 세 교사의 "원래" 검증 답을 얼마나 잘 예측하는가(sft/runs/*/cross_loss.csv).
   자기 교사 이점 = 자기 교사 원문 답에서의 loss − 같은 답에서 다른 두 학생의 평균 loss (음수일수록 자기 교사를 닮음).
2. 출처 판별: 학생이 써낸 답만 보고 어느 교사의 학생인지 맞힐 수 있는가.
   원래 학생은 교사 원문으로 학습한 분류기로, 재작성 학생은 교사 재작성본으로 학습한 분류기로 판별한다
   (각 학생이 배운 글과 같은 종류의 글로 분류기를 학습, 4절과 같은 글자 n-gram 로지스틱 회귀).
   추가로 교사 원문 분류기가 재작성 학생의 답을 누구로 보는지도 센다.
3. 문체 특징: 학생 답의 형식·문체 지표(results/style_analysis.py의 features), 답 길이.
4. 완결성: 잘린 답 수와, seed별로 학생 셋이 모두 끝맺은 평가 질문 수(평가 질문 선정용).
2·3은 끝맺은 답만 쓴다. 신뢰구간은 질문 단위 부트스트랩 1,000회(세 seed를 함께 재표집).

사용법: python results/rw_analysis.py
출력: results/output/rw_*.csv, rw_summary.json
      description/figures/56~59_그림8-20~8-23_*.png(상세), 60_그림8-Q_*.png(포스터용 두 장)
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
NAMES = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}
SEEDS = [42, 43, 44]
MAX_NEW_TOKENS = 1024
N_BOOT = 1000
C_RAW, C_RW = 16, 4   # 4절 원문 조건 best C, 재작성본 분류(classify/output/rewrite/metrics_logistic_rw.json)의 best C

COLORS = {"gpt": "#2a78d6", "claude": "#eb6834", "gemini": "#1baf7a"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#ffffff"
FONT = "Apple SD Gothic Neo"
KIND = {"orig": "원래 학생 (교사 원문으로 학습)", "rw": "재작성 학생 (HCX-005 재작성본으로 학습)"}


def setup():
    plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False, "font.size": 24,
                         "axes.labelsize": 24, "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 23})


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


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=250, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("그림:", name)


def complete(r):
    return r["finished"] and r["n_new_tokens"] < MAX_NEW_TOKENS and r["output"].strip() != ""


def load_answers():
    """(kind, teacher, seed) → 평가 질문 400개 답."""
    out = {}
    for t in TEACHERS:
        for s in SEEDS:
            out[("orig", t, s)] = json.loads((ROOT / "inference" / "output_2nd" / f"{t}_seed{s}_eval400.json").read_text(encoding="utf-8"))
            out[("rw", t, s)] = json.loads((ROOT / "inference" / "output_rw" / f"{t}_rw_seed{s}_eval400.json").read_text(encoding="utf-8"))
    return out


def teacher_texts(kind):
    texts, labels = [], []
    for k, t in enumerate(TEACHERS):
        if kind == "orig":
            rows = json.loads((ROOT / "inference" / "output" / f"{t}_train_1200.json").read_text(encoding="utf-8"))
            texts += [r[f"{t}_output"] for r in rows if r.get(f"{t}_output")]
            labels += [k] * sum(1 for r in rows if r.get(f"{t}_output"))
        else:
            rows = json.loads((ROOT / "sft" / "data" / f"{t}_rw_train_1000.json").read_text(encoding="utf-8"))
            texts += [r["output"] for r in rows]
            labels += [k] * len(rows)
    return texts, labels


def fit(kind):
    texts, labels = teacher_texts(kind)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
    clf = LogisticRegression(C=C_RAW if kind == "orig" else C_RW, max_iter=3000).fit(vec.fit_transform(texts), labels)
    print(f"분류기({kind}): 교사 답 {len(texts):,}개로 학습")
    return vec, clf


def boot_ci(df, col, rng):
    by_q = df.groupby("id")[col].agg(["sum", "count"])
    s, c = by_q["sum"].to_numpy(), by_q["count"].to_numpy()
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


def cross_loss():
    rows = [pd.read_csv(p) for p in sorted((ROOT / "sft" / "runs").glob("*_rw_seed*/cross_loss.csv"))]
    cl = pd.concat(rows + [pd.read_csv(ROOT / "sft" / "runs" / "cross_loss.csv")])
    cl = cl[cl.seed.isin(SEEDS)].copy()
    cl["kind"] = np.where(cl.student_teacher.str.endswith("_rw"), "rw", "orig")
    cl["teacher"] = cl.student_teacher.str.replace("_rw", "", regex=False)
    cl = cl[cl.teacher.isin(TEACHERS)]
    M = cl.groupby(["kind", "teacher", "eval_teacher"])["loss"].mean()
    adv = []
    for kind in ("orig", "rw"):
        for t in TEACHERS:
            for s in SEEDS:   # seed별 자기 교사 이점: 같은 seed의 다른 두 학생과 비교
                d = cl[(cl.kind == kind) & (cl.seed == s) & (cl.eval_teacher == t)].set_index("teacher")["loss"]
                adv.append({"kind": kind, "teacher": t, "seed": s, "adv": d[t] - d.drop(t).mean()})
    return cl, M, pd.DataFrame(adv)


def answer_table(answers, clf_orig, clf_rw):
    tok = AutoTokenizer.from_pretrained(ROOT / "sft" / "models" / "Qwen3-0.6B-Base")
    rows = []
    for (kind, t, s), items in answers.items():
        for r in items:
            if complete(r):
                rows.append({"kind": kind, "teacher": t, "seed": s, "id": r["id"], "text": r["output"],
                             **features(r["output"], r["instruction"], tok)})
    df = pd.DataFrame(rows)
    for name, (vec, clf) in (("orig", clf_orig), ("rw", clf_rw)):
        df[f"pred_{name}"] = np.array(TEACHERS)[clf.predict(vec.transform(df.text))]
    # 출처 판별: 학생이 배운 글과 같은 종류의 분류기
    df["pred_match"] = np.where(df.kind == "orig", df.pred_orig, df.pred_rw)
    df["correct"] = (df.pred_match == df.teacher).astype(float)
    return df.drop(columns="text")


# ---------- 그림 ----------

def fig_cross(M):
    """그림 8-20. 교차 loss 행렬: 원래 학생 vs 재작성 학생(세 교사의 원래 검증 답, seed 42~44 평균)."""
    fig, axes = plt.subplots(1, 2, figsize=(25, 9.5), gridspec_kw={"wspace": 0.42})
    for ax, kind in zip(axes, ("orig", "rw")):
        A = np.array([[M[(kind, s, e)] for e in TEACHERS] for s in TEACHERS])
        ax.imshow(-(A - A.min(axis=0)), cmap="Blues", vmin=-0.25, vmax=0.02, aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{A[i, j]:.3f}" + (" ★" if i == j else ""), ha="center", va="center", fontsize=27,
                        fontweight="bold" if i == j else "normal", color=INK)
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=4))
        ax.set_xticks([])
        for j, t in enumerate(TEACHERS):
            ax.text(j, -0.62, f"{NAMES[t]} 원문 답", ha="center", va="bottom", fontsize=24, fontweight="bold", color=COLORS[t])
        ax.set_yticks(range(3), [f"{NAMES[t]}{'-재작성' if kind == 'rw' else ''} 학생" for t in TEACHERS], fontsize=24)
        for lab, t in zip(ax.get_yticklabels(), TEACHERS):
            lab.set_color(COLORS[t])
            lab.set_fontweight("bold")
        ax.tick_params(length=0, pad=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(KIND[kind], fontsize=26, fontweight="bold", color=INK, pad=62, loc="left")
    save(fig, "56_그림8-20_8.2 결과 - 교차 loss 행렬 비교.png")


def fig_verdict(df):
    """그림 8-21. 교사 원문으로 학습한 분류기가 학생 답을 누구로 판정했나(원래 학생 vs 재작성 학생)."""
    fig, ax = plt.subplots(figsize=(24, 8.5))
    groups = [(k, t) for k in ("orig", "rw") for t in TEACHERS]
    x = np.arange(len(groups)) + np.array([0, 0, 0, 0.6, 0.6, 0.6])
    bottom = np.zeros(len(groups))
    for p in ["gemini", "gpt", "claude"]:
        v = np.array([(df[(df.kind == k) & (df.teacher == t)].pred_orig == p).mean() * 100 for k, t in groups])
        ax.bar(x, v, bottom=bottom, width=0.72, color=COLORS[p], edgecolor=SURFACE, lw=2, label=f"{NAMES[p]}로 판정")
        for xx, b, vv in zip(x, bottom, v):
            if vv >= 9:
                ax.text(xx, b + vv / 2, f"{vv:.0f}%", ha="center", va="center", fontsize=24, fontweight="bold", color="white")
        bottom += v
    ax.set_xticks(x, [f"{NAMES[t]}\n{'원래' if k == 'orig' else '재작성'} 학생" for k, t in groups], fontsize=23)
    ax.set_ylim(0, 100)
    ax.set_ylabel("학생 답의 판정 비율 (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False)
    style(ax)
    ax.tick_params(axis="x", length=0)
    save(fig, "57_그림8-21_8.2 결과 - 원문 분류기가 본 학생 답.png")


STYLE = [("bold", "굵게"), ("bullet", "글머리 목록"), ("numbered", "번호 목록"), ("en_paren", "영문 괄호"),
         ("hedge", "완곡 표현")]


def fig_style(df):
    """그림 8-22. 학생 답의 형식·문체 특징(%)과 길이: 원래 학생(진한 막대) vs 재작성 학생(옅은 막대)."""
    fig, axes = plt.subplots(1, 2, figsize=(27, 8.5), gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.18})
    ax = axes[0]
    xs = np.arange(len(STYLE))
    for k, t in enumerate(TEACHERS):
        for j, kind in enumerate(("orig", "rw")):
            v = [df[(df.kind == kind) & (df.teacher == t)][c].mean() * 100 for c, _ in STYLE]
            xx = xs + (k - 1) * 0.28 + (j - 0.5) * 0.13
            ax.bar(xx, v, width=0.125, color=COLORS[t], alpha=1 if kind == "orig" else 0.38, edgecolor=SURFACE, lw=1)
    ax.set_xticks(xs, [lab for _, lab in STYLE], fontsize=24)
    ax.set_ylim(0, 105)
    ax.set_ylabel("특징이 나타난 답 비율 (%)")
    h = [plt.Rectangle((0, 0), 1, 1, color=COLORS[t]) for t in TEACHERS] + \
        [plt.Rectangle((0, 0), 1, 1, color=INK_2), plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=0.38)]
    ax.legend(h, [f"{NAMES[t]}" for t in TEACHERS] + ["원래 학생", "재작성 학생"], loc="upper center",
              bbox_to_anchor=(0.62, 1.15), ncol=5, frameon=False)
    style(ax)
    ax.tick_params(axis="x", length=0)
    ax = axes[1]
    for k, t in enumerate(TEACHERS):
        for j, kind in enumerate(("orig", "rw")):
            v = df[(df.kind == kind) & (df.teacher == t)].chars.median()
            ax.bar(k + (j - 0.5) * 0.38, v, width=0.36, color=COLORS[t], alpha=1 if kind == "orig" else 0.38)
            ax.text(k + (j - 0.5) * 0.38, v + 8, f"{v:.0f}", ha="center", va="bottom", fontsize=23, color=INK)
    ax.set_xticks(range(3), [NAMES[t] for t in TEACHERS])
    ax.set_ylabel("답 길이 (글자 수 중앙값)")
    ax.set_ylim(0, 760)
    style(ax)
    ax.tick_params(axis="x", length=0)
    save(fig, "58_그림8-22_8.2 결과 - 학생 답의 문체 특징과 길이.png")


def fig_complete(n_cut):
    """그림 8-23. 학생별 잘린 답의 수(400개 중): 원래 학생 vs 재작성 학생, 점 = seed."""
    fig, ax = plt.subplots(figsize=(20, 8))
    groups = [(k, t) for t in TEACHERS for k in ("orig", "rw")]
    x = np.arange(len(groups)) + np.repeat([0, 0.5, 1.0], 2)
    for xx, (k, t) in zip(x, groups):
        v = [n_cut[(k, t, s)] for s in SEEDS]
        ax.bar(xx, np.mean(v), width=0.8, color=COLORS[t], alpha=1 if k == "orig" else 0.38)
        ax.scatter([xx] * 3, v, s=90, color=INK, zorder=3)
        ax.text(xx, max(v) + 4, f"{np.mean(v):.0f}", ha="center", va="bottom", fontsize=25, fontweight="bold")
    ax.set_xticks(x, [f"{NAMES[t]}\n{'원래' if k == 'orig' else '재작성'}" for k, t in groups])
    ax.set_ylim(0, 160)
    ax.set_ylabel("잘린 답의 수 (400개 중)")
    style(ax)
    ax.tick_params(axis="x", length=0)
    save(fig, "59_그림8-23_8.2 결과 - 학생 답의 완결성.png")


def fig_poster(adv, acc):
    """그림 8-Q(포스터용). ① 자기 교사 원문을 얼마나 더 잘 예측하나 ② 학생 답만으로 교사를 맞힌 비율."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(29, 10.5), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.22})
    # ① 자기 교사 이점(부호를 바꿔 "더 잘 예측한 정도"로): 원래 학생 vs 재작성 학생
    x = np.arange(3)
    for j, kind in enumerate(("orig", "rw")):
        v = np.array([-adv[(adv.kind == kind) & (adv.teacher == t)].adv.mean() for t in TEACHERS])
        pts = [-adv[(adv.kind == kind) & (adv.teacher == t)].adv.to_numpy() for t in TEACHERS]
        xx = x + (j - 0.5) * 0.4
        ax1.bar(xx, v, width=0.38, color=[COLORS[t] for t in TEACHERS], alpha=1 if kind == "orig" else 0.35,
                edgecolor=[COLORS[t] for t in TEACHERS], lw=2.5)
        for a, b, pp in zip(xx, v, pts):
            ax1.text(a, b + 0.006, f"{b:.3f}", ha="center", va="bottom", fontsize=25, fontweight="bold", color=INK)
    ax1.axhline(0, color=INK, lw=1.6)
    ax1.set_xticks(x, [NAMES[t] for t in TEACHERS], fontsize=26)
    for lab, t in zip(ax1.get_xticklabels(), TEACHERS):
        lab.set_color(COLORS[t])
        lab.set_fontweight("bold")
    ax1.set_ylabel("자기 교사 원문을 더 잘 예측한 정도\n(다른 학생 대비 loss 차, nats/토큰)", fontsize=24, color=INK_2)
    ax1.set_ylim(0, 0.2)
    h = [plt.Rectangle((0, 0), 1, 1, color=INK_2), plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=0.35)]
    ax1.legend(h, ["원래 학생", "재작성 학생"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=25)
    style(ax1)
    ax1.tick_params(axis="x", length=0)
    # ② 학생 답만으로 교사를 맞힌 비율
    labels = ["원래 학생", "재작성 학생"]
    v = [acc[k][0] for k in ("orig", "rw")]
    lo = [acc[k][1] for k in ("orig", "rw")]
    hi = [acc[k][2] for k in ("orig", "rw")]
    ax2.bar([0, 1], v, width=0.55, color=["#4a3aa7", "#b9b2e8"], edgecolor=SURFACE)
    ax2.errorbar([0, 1], v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=INK, elinewidth=2.4, capsize=10)
    for xx, vv, h_ in zip([0, 1], v, hi):
        ax2.text(xx, h_ + 1.5, f"{vv:.0f}%", ha="center", va="bottom", fontsize=31, fontweight="bold", color=INK)
    ax2.axhline(100 / 3, color=MUTED, ls="--", lw=2)
    ax2.text(1.42, 100 / 3 + 1.2, "우연 33%", ha="right", va="bottom", fontsize=23, color=MUTED)
    ax2.set_xticks([0, 1], labels, fontsize=26)
    ax2.set_xlim(-0.55, 1.5)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("학생 답만 보고 교사를 맞힌 비율 (%)", fontsize=24, color=INK_2)
    style(ax2)
    ax2.tick_params(axis="x", length=0)
    fig.canvas.draw()
    top = ax1.get_position().y1 + 0.12
    for ax, t in ((ax1, "① 재작성 학생은 자기 교사의 원래 문체를 거의 배우지 않았다"), (ax2, "② 학생 답만으로는 교사를 구별하기 어려워졌다")):
        fig.text(ax.get_position().x0 - 0.05, top, t, fontsize=30, fontweight="bold", color=INK)
    save(fig, "60_그림8-Q_8.2 포스터용 - 문체 제거 학생의 조작 점검.png")


def main():
    OUT.mkdir(exist_ok=True)
    setup()
    rng = np.random.default_rng(42)
    answers = load_answers()

    cl, M, adv = cross_loss()
    adv.to_csv(OUT / "rw_own_advantage.csv", index=False)
    M.unstack().to_csv(OUT / "rw_cross_loss.csv")

    clf_orig, clf_rw = fit("orig"), fit("rw")
    df = answer_table(answers, clf_orig, clf_rw)
    df.to_csv(OUT / "rw_answer_features.csv", index=False)
    acc = {}
    for kind in ("orig", "rw"):
        d = df[df.kind == kind]
        lo, hi = boot_ci(d, "correct", rng)
        acc[kind] = (d.correct.mean() * 100, lo * 100, hi * 100)
    acc_seed = df.groupby(["kind", "seed"]).correct.mean().unstack() * 100
    acc_teacher = df.groupby(["kind", "teacher"]).correct.mean().unstack() * 100
    verdict = (df.groupby(["kind", "teacher"]).pred_orig.value_counts(normalize=True).unstack(fill_value=0) * 100)
    sty = df.groupby(["kind", "teacher"])[["bold", "bullet", "numbered", "en_paren", "hedge", "formal_ratio", "summary_close"]].mean() * 100
    sty["chars_median"] = df.groupby(["kind", "teacher"]).chars.median()
    sty["n_complete"] = df.groupby(["kind", "teacher"]).size()
    sty.to_csv(OUT / "rw_style.csv")

    n_cut = {(k, t, s): sum(not complete(r) for r in answers[(k, t, s)]) for (k, t, s) in answers}
    ok = {key: {r["id"] for r in v if complete(r)} for key, v in answers.items()}
    old = set(json.loads((ROOT / "judge" / "data" / "selection.json").read_text())["selected_question_ids"])
    per_seed = {s: set.intersection(*[ok[("rw", t, s)] for t in TEACHERS]) for s in SEEDS}

    summary = {
        "cross_loss_mean": {f"{k}|{t}|{e}": round(v, 4) for (k, t, e), v in M.items()},
        "own_advantage_mean": {f"{k}|{t}": v for (k, t), v in adv.groupby(["kind", "teacher"]).adv.mean().round(4).items()},
        "own_advantage_by_seed": {f"{r.kind}|{r.teacher}|{r.seed}": round(r.adv, 4) for r in adv.itertuples()},
        "teacher_identification_pct": {k: [round(x, 1) for x in v] for k, v in acc.items()},
        "teacher_identification_by_seed": acc_seed.round(1).to_dict(orient="index"),
        "teacher_identification_by_teacher": acc_teacher.round(1).to_dict(orient="index"),
        "orig_classifier_verdict_pct": {f"{k}|{t}": row.round(1).to_dict() for (k, t), row in verdict.iterrows()},
        "n_truncated": {f"{k}|{t}|{s}": n for (k, t, s), n in n_cut.items()},
        "common_complete_rw_per_seed": {s: {"n": len(v), "kept_old": len(v & old)} for s, v in per_seed.items()},
    }
    (OUT / "rw_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(M.unstack().round(3))
    print(adv.groupby(["kind", "teacher"]).adv.agg(["mean", "min", "max"]).round(4))
    print("출처 판별 정확도", {k: [round(x, 1) for x in v] for k, v in acc.items()})
    print(acc_seed.round(1)); print(acc_teacher.round(1)); print(verdict.round(1)); print(sty.round(1))
    print("잘린 답", n_cut)
    print("seed별 재작성 학생 공통 완전 질문", {s: (len(v), len(v & old)) for s, v in per_seed.items()})

    fig_cross(M)
    fig_verdict(df)
    fig_style(df)
    fig_complete(n_cut)
    fig_poster(adv, acc)


if __name__ == "__main__":
    main()
