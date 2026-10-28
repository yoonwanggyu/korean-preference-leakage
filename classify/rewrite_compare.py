"""8.2 조작 점검: HCX-005로 다시 쓴 답만 보고도 원래 교사(GPT/Claude/Gemini)를 맞힐 수 있는가.

4절(logistic.py, roberta.py)과 같은 분류기·하이퍼파라미터·test 질문(181개)을 쓰고 입력 글만 바꾼다.
재작성은 SFT 학습 질문 1,000개에만 있으므로 4절 분할(output/split.json)을 그 1,000개로 제한한다:
train 839 → 640개(나머지 199개는 재작성본이 없음), val 179·test 181은 그대로(모두 1,000개 안에 있다).
공정한 비교를 위해 원문·정규화도 같은 640개로 다시 학습한다.

조건
  raw   교사 원문
  norm  원문 정규화(굵게·목록·따옴표·줄바꿈 제거, 4.6)
  rw    HCX-005 재작성(sft/data/{teacher}_rw_train_1000.json)
추가: 원문(raw)으로 학습한 로지스틱 분류기가 재작성 test 답을 어느 교사로 판정하는지(cross_logistic.json).

사용법
  python classify/rewrite_compare.py                      로지스틱 세 조건 + 원문→재작성 판정
  python classify/rewrite_compare.py --roberta raw        KLUE-RoBERTa 한 조건 (raw | norm | rw), Mac MPS로 약 15분
  python classify/rewrite_compare.py --plot               포스터 그림 8-16~8-19 (설명 문구는 poster.html에 쓴다)
출력(classify/output/rewrite/): metrics_{logistic,roberta}_{cond}.json, predictions_*_test.csv, cross_logistic.json
      description/figures/51~54_그림8-16~8-19_*.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from common import COLORS, MODELS, OUT_DIR, PREFIX, SEED, bootstrap_ci, normalize, save_predictions
from logistic import C_GRID, MIN_DF, NGRAM
from roberta import (BATCH_SIZE, EPOCHS, EVAL_STEPS, GRAD_ACCUM, LEARNING_RATE, LOGGING_STEPS, MAX_LEN, MODEL_NAME,
                     PAD_MULTIPLE, WARMUP_RATIO, WEIGHT_DECAY, EmptyMpsCache, compute_metrics, to_dataset)

ROOT = Path(__file__).resolve().parent.parent
SFT_DATA = ROOT / "sft" / "data"
OUT = OUT_DIR / "rewrite"
CKPT = Path(__file__).parent / "checkpoints"
CONDS = {"raw": "원문", "norm": "원문 정규화", "rw": "HCX-005 재작성"}
FIG = ROOT / "description" / "figures"
CLFS = {"logistic": "로지스틱 회귀", "roberta": "KLUE-RoBERTa"}
COND_LAB = {"raw": "원문", "norm": "서식 제거\n(4.6 정규화)", "rw": "HCX-005\n재작성"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#fcfcfb"
CLF_COLOR = {"logistic": "#4a3aa7", "roberta": "#a8a69e"}


def load_rows():
    """[{id, label, text_raw, text_rw, split}]: 4절 분할을 재작성본이 있는 질문으로 제한한다."""
    split = json.loads((OUT_DIR / "split.json").read_text())
    orig = {m: {x["id"]: x["output"] for x in json.loads((SFT_DATA / f"{PREFIX[m]}_train_1000.json").read_text(encoding="utf-8"))}
            for m in MODELS}
    rw = {m: {x["id"]: x["output"] for x in json.loads((SFT_DATA / f"{PREFIX[m]}_rw_train_1000.json").read_text(encoding="utf-8"))}
          for m in MODELS}
    # 4절은 inference/output의 교사 답(1,200개)을 썼고, SFT 데이터는 그 답에서 길이 필터를 거친 같은 답이다.
    src = {m: {x["id"]: x[f"{PREFIX[m]}_output"] for x in json.loads((ROOT / "inference" / "output" / f"{PREFIX[m]}_train_1200.json").read_text(encoding="utf-8"))}
           for m in MODELS}
    same = sum(src[m][i] == orig[m][i] for m in MODELS for i in orig[m])
    print(f"SFT 학습 답과 4절 교사 답 일치: {same}/{sum(len(orig[m]) for m in MODELS)}")
    rows = []
    for s in ("train", "val", "test"):
        for i in split[s]:
            if i in rw[MODELS[0]]:
                rows += [{"id": i, "label": k, "split": s, "raw": src[m][i], "rw": rw[m][i]} for k, m in enumerate(MODELS)]
    n = {s: len({r["id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")}
    print(f"질문 수: train {n['train']}, val {n['val']}, test {n['test']}")
    return rows


def part(rows, split, cond):
    p = [r for r in rows if r["split"] == split]
    text = [r["rw"] if cond == "rw" else (normalize(r["raw"]) if cond == "norm" else r["raw"]) for r in p]
    return text, np.array([r["label"] for r in p]), np.array([r["id"] for r in p])


def summarize(y, pred, groups, extra):
    lo, hi = bootstrap_ci(y, pred, groups)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(len(MODELS)), zero_division=0)
    return {**extra, "test_acc": float((pred == y).mean()), "test_ci95": [lo, hi],
            "per_class": {m: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, m in enumerate(MODELS)},
            "confusion": confusion_matrix(y, pred, labels=range(len(MODELS))).tolist()}


def run_logistic(rows):
    fitted = {}
    for cond in CONDS:
        x_tr, y_tr, _ = part(rows, "train", cond)
        x_va, y_va, _ = part(rows, "val", cond)
        x_te, y_te, g_te = part(rows, "test", cond)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM, min_df=MIN_DF, sublinear_tf=True)
        X_tr, X_va, X_te = vec.fit_transform(x_tr), vec.transform(x_va), vec.transform(x_te)
        val_acc = {c: float((LogisticRegression(C=c, max_iter=3000).fit(X_tr, y_tr).predict(X_va) == y_va).mean()) for c in C_GRID}
        best = max(C_GRID, key=lambda c: val_acc[c])
        clf = LogisticRegression(C=best, max_iter=3000).fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)
        m = summarize(y_te, prob.argmax(1), g_te, {"classifier": "logistic", "cond": cond, "best_C": best,
                                                     "val_acc_by_C": {str(c): a for c, a in val_acc.items()},
                                                     "n_train_questions": len(set(_ for _ in part(rows, "train", cond)[2]))})
        (OUT / f"metrics_logistic_{cond}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        save_predictions(g_te, y_te, prob, OUT / f"predictions_logistic_{cond}_test.csv")
        fitted[cond] = (vec, clf)
        print(f"로지스틱 [{CONDS[cond]}] C={best}, test {m['test_acc']:.1%} ({m['test_ci95'][0]:.1%}~{m['test_ci95'][1]:.1%}), "
              f"F1 " + ", ".join(f"{k} {v['f1']:.2f}" for k, v in m["per_class"].items()))

    # 원문으로 학습한 분류기가 재작성 test 답을 어떻게 판정하나
    vec, clf = fitted["raw"]
    x_te, y_te, _ = part(rows, "test", "rw")
    pred = clf.predict(vec.transform(x_te))
    cross = {MODELS[k]: {MODELS[j]: float((pred[y_te == k] == j).mean()) for j in range(len(MODELS))} for k in range(len(MODELS))}
    (OUT / "cross_logistic.json").write_text(json.dumps({"train": "raw", "test": "rw", "share": cross}, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print("원문 학습 분류기 → 재작성 test 판정 비율:", {k: {j: round(v, 2) for j, v in d.items()} for k, d in cross.items()})


def run_roberta(rows, cond):
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    splits = {s: part(rows, s, cond) for s in ("train", "val", "test")}
    ds = {s: to_dataset(x, y, tok) for s, (x, y, _) in splits.items()}
    run_dir = CKPT / f"roberta_rewrite_{cond}"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(MODELS), id2label=dict(enumerate(MODELS)), label2id={m: i for i, m in enumerate(MODELS)})
    args = TrainingArguments(   # roberta.py와 같은 설정
        output_dir=str(run_dir), num_train_epochs=EPOCHS, learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM, gradient_checkpointing=True, warmup_steps=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY, lr_scheduler_type="linear", eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_strategy="steps", save_steps=EVAL_STEPS, save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="accuracy", greater_is_better=True, logging_steps=LOGGING_STEPS,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(), seed=SEED, data_seed=SEED,
        report_to="none", dataloader_pin_memory=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=PAD_MULTIPLE), processing_class=tok,
                      compute_metrics=compute_metrics, callbacks=[EmptyMpsCache()])
    trainer.train()
    best_step = int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1])
    prob = torch.softmax(torch.tensor(trainer.predict(ds["test"]).predictions), -1).numpy()
    _, y_te, g_te = splits["test"]
    m = summarize(y_te, prob.argmax(1), g_te, {"classifier": "roberta", "cond": cond, "best_step": best_step,
                                                 "best_val_acc": trainer.state.best_metric})
    (OUT / f"metrics_roberta_{cond}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    save_predictions(g_te, y_te, prob, OUT / f"predictions_roberta_{cond}_test.csv")
    print(f"RoBERTa [{CONDS[cond]}] best step {best_step}, test {m['test_acc']:.1%} ({m['test_ci95'][0]:.1%}~{m['test_ci95'][1]:.1%})")


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    if grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("그림:", name)


def plot_all():
    plt.rcParams.update({"font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 15,
                         "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14})
    M = {(c, k): json.loads((OUT / f"metrics_{c}_{k}.json").read_text()) for c in CLFS for k in CONDS}

    # 그림 8-16: 세 조건 × 두 분류기 test 정확도
    fig, ax = plt.subplots(figsize=(11, 6.4))
    x = np.arange(len(CONDS))
    for j, c in enumerate(CLFS):
        acc = np.array([M[(c, k)]["test_acc"] * 100 for k in CONDS])
        lo = np.array([M[(c, k)]["test_ci95"][0] * 100 for k in CONDS])
        hi = np.array([M[(c, k)]["test_ci95"][1] * 100 for k in CONDS])
        xx = x + (j - 0.5) * 0.36
        ax.bar(xx, acc, width=0.34, color=CLF_COLOR[c], edgecolor=SURFACE, lw=2, label=CLFS[c])
        ax.errorbar(xx, acc, yerr=[acc - lo, hi - acc], fmt="none", ecolor=INK, elinewidth=1.6, capsize=5)
        for a, v, b in zip(xx, acc, hi):
            ax.text(a, b + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK)
    ax.axhline(100 / 3, color=MUTED, ls="--", lw=1.6)
    ax.text(x[-1] + 0.62, 100 / 3 + 1, "우연 33.3", ha="right", va="bottom", fontsize=13, color=MUTED)
    ax.set_xticks(x, [COND_LAB[k] for k in CONDS])
    ax.set_ylim(0, 108)
    ax.set_ylabel("출처(교사) 분류 test 정확도 (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
    style(ax)
    ax.tick_params(axis="x", length=0)
    save(fig, "51_그림8-16_8.2 재작성 - 분류 정확도 비교.png")

    # 그림 8-17: 교사별 F1
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    shade = {"raw": 1.0, "norm": 0.6, "rw": 0.3}
    for ax, c in zip(axes, CLFS):
        for j, k in enumerate(CONDS):
            f1 = [M[(c, k)]["per_class"][m]["f1"] for m in MODELS]
            xx = np.arange(len(MODELS)) + (j - 1) * 0.27
            ax.bar(xx, f1, width=0.25, color=[COLORS[m] for m in MODELS], alpha=shade[k], edgecolor=SURFACE, lw=1.5,
                   label=COND_LAB[k].replace("\n", " "))
            for a, b in zip(xx, f1):
                ax.text(a, b + 0.015, f"{b:.2f}", ha="center", va="bottom", fontsize=12.5, color=INK)
        ax.set_xticks(range(len(MODELS)), MODELS)
        ax.set_title(CLFS[c], fontsize=17, color=INK, loc="left")
        ax.set_ylim(0, 1.12)
        style(ax)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("교사별 F1")
    h = [plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=shade[k]) for k in CONDS]
    fig.legend(h, [COND_LAB[k].replace("\n", " ") for k in CONDS], loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "52_그림8-17_8.2 재작성 - 교사별 F1.png")

    # 그림 8-18: 혼동행렬 (행 = 실제 교사, 열 = 예측, 값 = 행 기준 %)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.5))
    for r, c in enumerate(CLFS):
        for col, k in enumerate(["raw", "rw"]):
            ax = axes[r, col]
            cm = np.array(M[(c, k)]["confusion"], float)
            pct = cm / cm.sum(1, keepdims=True) * 100
            ax.imshow(pct, cmap="Purples", vmin=0, vmax=100)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{pct[a, b]:.0f}%", ha="center", va="center", fontsize=17, fontweight="bold",
                            color="white" if pct[a, b] > 55 else INK)
            ax.set_xticks(range(3), MODELS)
            ax.set_yticks(range(3), MODELS)
            ax.set_title(f"{CLFS[c]} · {COND_LAB[k].replace(chr(10), ' ')}", fontsize=16, color=INK, loc="left")
            ax.tick_params(length=0, colors=INK_2)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col == 0:
                ax.set_ylabel("실제 교사")
            if r == 1:
                ax.set_xlabel("분류기가 예측한 교사")
    fig.tight_layout(h_pad=2.5, w_pad=2)
    save(fig, "53_그림8-18_8.2 재작성 - 혼동행렬 비교.png")

    # 그림 8-19: 원문으로 학습한 분류기가 재작성 답을 어디로 판정하나
    cross = json.loads((OUT / "cross_logistic.json").read_text())["share"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(MODELS))[::-1]
    left = np.zeros(len(MODELS))
    for m in MODELS:
        v = np.array([cross[t][m] * 100 for t in MODELS])
        ax.barh(y, v, left=left, height=0.62, color=COLORS[m], edgecolor=SURFACE, lw=2, label=f"{m}로 판정")
        for yy, l, w in zip(y, left, v):
            if w >= 8:
                ax.text(l + w / 2, yy, f"{w:.0f}%", ha="center", va="center", fontsize=15, fontweight="bold", color="white")
        left += v
    ax.set_yticks(y, [f"{m} 답의 재작성" for m in MODELS])
    ax.set_xlim(0, 100)
    ax.set_xlabel("원문으로 학습한 분류기의 판정 비율 (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False)
    style(ax, grid="x")
    ax.tick_params(axis="y", length=0)
    save(fig, "54_그림8-19_8.2 재작성 - 원문 분류기의 재작성 판정.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roberta", choices=list(CONDS), help="KLUE-RoBERTa를 이 조건으로 학습·평가")
    parser.add_argument("--plot", action="store_true", help="저장된 결과로 그림만 그린다")
    cli = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if cli.plot:
        plot_all()
        return
    rows = load_rows()
    if cli.roberta:
        run_roberta(rows, cli.roberta)
    else:
        run_logistic(rows)


if __name__ == "__main__":
    main()
