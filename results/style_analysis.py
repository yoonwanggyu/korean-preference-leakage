"""GPT/Claude/Gemini 생성 답변의 형태·문체 특징을 비교하고 포스터용 표와 그림을 만든다.

1. 답변마다 길이, 형식(마크다운), 문체 특징을 계산해 모델별로 요약한다.
2. 결과를 results/output/에 CSV, Markdown, PNG로 저장한다.
답변만으로 생성 모델을 맞히는 분류 실험은 classify/(logistic.py, roberta.py)에서 따로 한다.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "inference" / "output"
OUT_DIR = Path(__file__).parent / "output"
TOKENIZER = "Qwen/Qwen3-0.6B-Base"
MAX_LEN = 1024
MIN_GAP = 0.10  # 막대그림에 넣을 지표의 최소 모델 간 차이 (10%p)
SEED = 42

# (표시 이름, 파일 접두어). 색은 CVD 검증을 통과한 범주형 팔레트 앞 3칸을 고정 순서로 쓴다.
MODELS = [("GPT", "gpt"), ("Claude", "claude"), ("Gemini", "gemini")]
COLORS = {"GPT": "#2a78d6", "Claude": "#eb6834", "Gemini": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FONT = "Apple SD Gothic Neo"

SUMMARY_WORDS = ("요약하면", "정리하면", "결론적으로", "요컨대", "한마디로", "종합하면")
HEDGE_WORDS = ("일반적으로", "보통", "대체로", "경우에 따라", "수 있습니다", "수 있어요")


def load_answers():
    frames = []
    for name, prefix in MODELS:
        path = GEN_DIR / f"{prefix}_train_1200.json"
        if not path.exists():
            print(f"[건너뜀] {path.name} 없음")
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame({
            "id": [x["id"] for x in items],
            "instruction": [x["instruction"] for x in items],
            "model": name,
            "text": [x[f"{prefix}_output"] for x in items],
        }))
    df = pd.concat(frames, ignore_index=True)
    # 모든 모델이 답한 질문만 남겨 같은 질문 집합에서 비교한다.
    ok = df.dropna(subset=["text"]).groupby("id")["model"].nunique()
    keep = ok[ok == df["model"].nunique()].index
    return df[df["id"].isin(keep)].reset_index(drop=True)


def features(text, instruction, tok):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    formal = len(re.findall(r"니다[.!?]", text))
    polite = len(re.findall(r"요[.!?]", text))
    return {
        "chars": len(text),
        "qwen_tokens": len(tok(instruction + "\n" + text)["input_ids"]) + 1,  # +1: EOS
        "sentences": len(sentences),
        "sent_len": len(text) / max(len(sentences), 1),
        "paragraphs": len(paragraphs),
        "bold": "**" in text,
        "bullet": any(re.match(r"[-*•]\s", l) for l in lines),
        "numbered": any(re.match(r"\d+[.)]\s", l) for l in lines),
        "en_paren": bool(re.search(r"\([^)]*[A-Za-z][^)]*\)", text)),
        "hanja": bool(re.search(r"[一-鿿]", text)),
        "formal_ratio": formal / (formal + polite) if formal + polite else np.nan,
        "summary_close": any(w in paragraphs[-1] for w in SUMMARY_WORDS) if paragraphs else False,
        "hedge": any(w in text for w in HEDGE_WORDS),
    }


# 표에 넣을 지표: (열 이름, 표시 이름, 묶음, 형식). 비율은 해당 특징이 있는 답변의 비율이다.
ROWS = [
    ("chars", "글자 수 (중앙값)", "길이", "median"),
    ("sentences", "문장 수 (중앙값)", "길이", "median"),
    ("sent_len", "문장당 글자 수 (중앙값)", "길이", "median"),
    ("paragraphs", "문단 수 (중앙값)", "길이", "median"),
    ("bold", "굵게(**) 사용", "형식", "rate"),
    ("bullet", "글머리 목록(-, *) 사용", "형식", "rate"),
    ("numbered", "번호 목록(1.) 사용", "형식", "rate"),
    ("en_paren", "영문 괄호 병기", "문체", "rate"),
    ("hanja", "한자 병기", "문체", "rate"),
    ("formal_ratio", "합쇼체(~니다) 비중", "문체", "mean"),
    ("summary_close", "요약형 마무리", "문체", "rate"),
    ("hedge", "완곡 표현(~수 있습니다 등)", "문체", "rate"),
]


def summarize(feat):
    models = [m for m, _ in MODELS if m in feat["model"].unique()]
    rows = []
    for col, label, group, how in ROWS:
        g = feat.groupby("model")[col]
        vals = g.median() if how == "median" else g.mean()
        rows.append({"묶음": group, "지표": label, "how": how, **{m: vals[m] for m in models}})
    return pd.DataFrame(rows), models


def distinctive_phrases(df, models, top=5, min_df=30, min_rate=0.10):
    """모델별로 다른 모델보다 훨씬 자주 쓰는 어절(1~2-gram)을 문서 빈도 비율로 뽑는다."""
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df, binary=True,
                          use_idf=False, norm=None, token_pattern=r"[가-힣A-Za-z]{2,}")
    X = vec.fit_transform(df["text"].tolist())
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for m in models:
        mask = (df["model"] == m).to_numpy(dtype=bool)
        p_in = (np.asarray(X[mask].sum(0)).ravel() + 1) / (mask.sum() + 2)
        p_out = (np.asarray(X[~mask].sum(0)).ravel() + 1) / ((~mask).sum() + 2)
        score = np.log(p_in / p_out)
        score[p_in < min_rate] = -np.inf  # 해당 모델 답변의 10% 이상에서 쓰인 표현만 후보로 둔다
        order = np.argsort(-score)[:top]
        out[m] = [(vocab[i], p_in[i], p_out[i]) for i in order]
    return out


def fmt(v, how):
    if pd.isna(v):
        return "-"
    return f"{v:.0f}" if how == "median" else f"{v * 100:.0f}%"


def draw_table(summary, models, phrases, path):
    """포스터용 비교표. 각 행에서 가장 큰 값을 굵게 표시한다."""
    body = []
    for _, r in summary.iterrows():
        vals = [r[m] for m in models]
        # 최댓값이 하나뿐일 때만 굵게 표시한다.
        best = int(np.nanargmax(vals)) if np.sum(np.isclose(vals, np.nanmax(vals))) == 1 else -1
        body.append((r["묶음"], r["지표"], [fmt(v, r["how"]) for v in vals], best))
    for k in range(2):
        body.append(("표현", "대표 표현 (사용률)" if k == 0 else "",
                     [f"{phrases[m][k][0]} {phrases[m][k][1]:.0%}" for m in models], -1))

    n = len(body)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * (n + 2)), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 1.2, -0.2)
    x_group, x_label = 0.01, 0.10
    x_cols = [0.60 + i * 0.14 for i in range(len(models))]

    for x, m in zip(x_cols, models):
        ax.add_patch(plt.Rectangle((x - 0.035, 0.12), 0.07, 0.14, color=COLORS[m], lw=0))
        ax.text(x, 0.62, m, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.axhline(1.0, color=INK, lw=1.0)

    prev = None
    for i, (group, label, cells, best) in enumerate(body):
        y = i + 1.5
        if group != prev:
            if prev is not None:
                ax.axhline(y - 0.5, color=GRID, lw=0.8)
            ax.text(x_group, y, group, va="center", fontsize=9.5, color=INK_2, fontweight="bold")
            prev = group
        ax.text(x_label, y, label, va="center", fontsize=9.5, color=INK)
        for j, (x, c) in enumerate(zip(x_cols, cells)):
            ax.text(x, y, c, ha="center", va="center", fontsize=10,
                    fontweight="bold" if j == best else "normal", color=INK if j == best else INK_2)
    ax.axhline(n + 1.0, color=INK, lw=1.0)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def draw_bars(summary, models, path):
    """형식·문체 비율 지표를 모델별 가로 막대로 비교한다. 값은 막대 끝에 직접 적는다.
    모델 간 차이가 MIN_GAP 미만인 지표는 표에만 두고 그림에서는 뺀다."""
    rates = summary[summary["how"] != "median"]
    rates = rates[rates[models].max(axis=1) - rates[models].min(axis=1) >= MIN_GAP].reset_index(drop=True)
    n, k = len(rates), len(models)
    h = 0.8 / k
    fig, ax = plt.subplots(figsize=(7.5, 0.62 * n + 0.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for j, m in enumerate(models):
        ys = np.arange(n) + (j - (k - 1) / 2) * h
        vals = rates[m].values * 100
        ax.barh(ys, vals, height=h - 0.03, color=COLORS[m], label=m, edgecolor=SURFACE, linewidth=1)
        for y, v in zip(ys, vals):
            ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(np.arange(n), rates["지표"], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("해당 특징이 나타난 답변 비율 (%)", fontsize=9, color=INK_2)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=k, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False, fontsize=9)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = FONT
    plt.rcParams["axes.unicode_minus"] = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_answers()
    print(f"비교 질문 {df['id'].nunique()}개, 답변 {len(df)}개")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    feat = pd.DataFrame([features(t, q, tok) for t, q in zip(df["text"], df["instruction"])])
    feat.insert(0, "model", df["model"])
    feat.insert(0, "id", df["id"])
    feat.to_csv(OUT_DIR / "style_features.csv", index=False)

    summary, models = summarize(feat)
    phrases = distinctive_phrases(df, models)

    over = feat.assign(over=feat["qwen_tokens"] > MAX_LEN).groupby("model")["over"].sum()
    over_ids = feat.loc[feat["qwen_tokens"] > MAX_LEN, "id"].nunique()

    summary.drop(columns="how").to_csv(OUT_DIR / "style_summary.csv", index=False)
    lines = ["| 묶음 | 지표 | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for _, r in summary.iterrows():
        lines.append(f"| {r['묶음']} | {r['지표']} | " + " | ".join(fmt(r[m], r["how"]) for m in models) + " |")
    lines += ["", f"- Qwen 토큰 {MAX_LEN} 초과: " + ", ".join(f"{m} {int(over[m])}개" for m in models)
              + f" (합집합 질문 {over_ids}개)", "", "특징적 표현 (해당 모델 사용률 / 다른 모델 사용률):"]
    for m in models:
        lines.append(f"- {m}: " + ", ".join(f"'{w}' {a:.0%}/{b:.0%}" for w, a, b in phrases[m]))
    report = "\n".join(lines)
    (OUT_DIR / "style_summary.md").write_text(report, encoding="utf-8")
    print(report)

    draw_table(summary, models, phrases, OUT_DIR / "fig_style_table.png")
    draw_bars(summary, models, OUT_DIR / "fig_style_bars.png")
    print(f"저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
