"""생성 모델 분류 실험(logistic.py, roberta.py)이 함께 쓰는 데이터 분할, 전처리, 그림 함수.

- 세 교사가 모두 답한 질문만 쓰고, 질문 단위로 train/val/test를 나눈다.
  같은 질문에 대한 세 답변은 항상 같은 split에 들어가므로, 모델은 질문 내용이 아니라 답변 스타일로만 맞혀야 한다.
- 텍스트 조건은 두 가지다.
  raw : 교사 답변 원문
  norm: 마크다운 기호, 목록 기호, 따옴표 종류, 줄바꿈을 모두 없애 "겉모양"을 지운 텍스트
"""

import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "inference" / "output"
OUT_DIR = Path(__file__).parent / "output"

MODELS = ["GPT", "Claude", "Gemini"]
PREFIX = {"GPT": "gpt", "Claude": "claude", "Gemini": "gemini"}
LABEL2ID = {m: i for i, m in enumerate(MODELS)}
SPLIT_RATIO = (0.70, 0.15, 0.15)  # train / val / test (질문 수 기준)
SEED = 42
N_BOOT = 1000

# 그림 색: CVD 검증을 통과한 범주형 팔레트 앞 3칸 (results/style_analysis.py와 동일)
COLORS = {"GPT": "#2a78d6", "Claude": "#eb6834", "Gemini": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FONT = "Apple SD Gothic Neo"


def normalize(text):
    """마크다운·목록·따옴표·줄바꿈을 지워 어휘와 문장만 남긴다."""
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"^\s*(#+|[-*•]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"[“”‘’\"']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset():
    """[{id, label, model, text, split}] 목록을 돌려준다. 분할은 SEED로 고정된다."""
    answers = {}
    for m in MODELS:
        items = json.loads((GEN_DIR / f"{PREFIX[m]}_train_1200.json").read_text(encoding="utf-8"))
        answers[m] = {x["id"]: x[f"{PREFIX[m]}_output"] for x in items}
    ids = sorted(i for i in answers[MODELS[0]] if all(answers[m].get(i) for m in MODELS))

    rng = random.Random(SEED)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_train = int(len(ids) * SPLIT_RATIO[0])
    n_val = int(len(ids) * SPLIT_RATIO[1])
    split_of = {i: "train" for i in shuffled[:n_train]}
    split_of.update({i: "val" for i in shuffled[n_train:n_train + n_val]})
    split_of.update({i: "test" for i in shuffled[n_train + n_val:]})

    return [{"id": i, "label": LABEL2ID[m], "model": m, "text": answers[m][i], "split": split_of[i]}
            for i in ids for m in MODELS]


def get_split(rows, split, cond):
    part = [r for r in rows if r["split"] == split]
    texts = [r["text"] if cond == "raw" else normalize(r["text"]) for r in part]
    return texts, np.array([r["label"] for r in part]), np.array([r["id"] for r in part])


def save_split(rows):
    """어떤 질문 id가 어느 split인지 파일로 남긴다."""
    split = {s: sorted({r["id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")}
    (OUT_DIR / "split.json").write_text(json.dumps(split), encoding="utf-8")


def save_predictions(ids, y_true, probs, path):
    """답변별 예측 결과를 CSV로 저장한다: id, 실제 모델, 예측 모델, 정답 여부, 세 모델 확률."""
    lines = ["id,true_model,pred_model,correct," + ",".join(f"p_{m}" for m in MODELS)]
    for i, y, p in zip(ids, y_true, probs):
        k = int(np.argmax(p))
        lines.append(f"{i},{MODELS[y]},{MODELS[k]},{int(k == y)}," + ",".join(f"{v:.6f}" for v in p))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bootstrap_ci(y_true, y_pred, groups):
    """질문 단위로 재표집한 정확도의 95% 신뢰구간."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    accs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq))])
        accs.append((y_true[idx] == y_pred[idx]).mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def setup_plot():
    plt.rcParams["font.family"] = FONT
    plt.rcParams["axes.unicode_minus"] = False


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(MODELS)), normalize="true") * 100
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=100)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > 55 else INK)
    ax.set_xticks(range(len(MODELS)), MODELS, fontsize=9, color=INK)
    ax.set_yticks(range(len(MODELS)), MODELS, fontsize=9, color=INK)
    ax.set_xlabel("예측한 생성 모델", fontsize=9, color=INK_2)
    ax.set_ylabel("실제 생성 모델", fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=9.5, color=INK, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
