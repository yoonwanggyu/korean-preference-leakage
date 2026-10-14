"""포스터용 요약 문서(description/poster.html)를 만든다. 모든 그림(인라인 SVG)과 표는 분석 결과 파일에서 직접 읽는다.

입력: classify/output/compare.md, results/output/style_summary.csv, results/output/sft_summary.json,
      sft/runs/*/val_loss.csv, judge/data/selection.json, judge/output/analysis/{desc_*,model_*}.csv,
      description/result.html(연구 질문 절을 그대로 가져온다)
사용법: python results/poster.py  → description/poster.html (PDF는 Chrome 인쇄로 만든다)
"""

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "judge" / "output" / "analysis"
OUT = ROOT / "description" / "poster.html"
J = ["gpt", "claude", "gemini"]
N = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini"}
COL = {"gpt": "var(--c-gpt)", "claude": "var(--c-claude)", "gemini": "var(--c-gemini)"}
SEEDS = [42, 43, 44, 45, 46]
CN = {"overall": "최종 순위", "factuality": "사실 정확성", "comprehension": "질문 충족도",
      "logical_robustness": "논리성", "readability": "가독성"}
CRIT = ["factuality", "comprehension", "logical_robustness", "readability"]
e = html.escape


def sgn(x, d=1):
    return f"{x:+.{d}f}".replace("-", "−")


def pf(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def pb(p, mult=1):
    """부트스트랩 p: 1,000번 중 0번이면 실제 값은 하한(0.001) 이하이므로 ≤로 적는다."""
    return f"≤ {0.001 * mult:.3f}" if p <= 0.001 * mult + 1e-12 else pf(p)


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ------------------------------------------------------------------ SVG 그림 도구
FONT = 'font-family="Apple SD Gothic Neo, Pretendard, Noto Sans KR, system-ui, sans-serif"'


def svg_open(w, h, label):
    return f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{e(label)}" {FONT}>'


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{e(n)}</span>' for n, c in items) + "</div>"


def xaxis(x0, x1, y, ticks, scale, fmt, h_top):
    s = f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" class="axis"/>'
    for t in ticks:
        x = scale(t)
        s += f'<line x1="{x}" x2="{x}" y1="{h_top}" y2="{y}" class="grid"/><text x="{x}" y="{y + 16}" class="tick" text-anchor="middle">{fmt(t)}</text>'
    return s


def hbars(rows, xmax, ref=None, ref_label="", fmt=lambda v: f"{v:.1f}%", color="var(--c-neutral)", label_w=170, w=640):
    """rows: [(label, value, lo, hi)] 가로 막대 + 신뢰구간."""
    bh, gap, top = 26, 12, 14
    h = top + len(rows) * (bh + gap) + 30
    sc = lambda v: label_w + v / xmax * (w - label_w - 70)
    s = svg_open(w, h, "가로 막대 그림")
    s += xaxis(label_w, w - 70, h - 26, [0, 25, 50, 75, 100], sc, lambda t: f"{t}%", top - 6)
    for i, (lab, v, lo, hi) in enumerate(rows):
        y = top + i * (bh + gap)
        s += f'<text x="{label_w - 10}" y="{y + bh / 2 + 5}" class="lab" text-anchor="end">{e(lab)}</text>'
        s += f'<rect x="{label_w}" y="{y}" width="{sc(v) - label_w}" height="{bh}" rx="4" fill="{color}"><title>{e(lab)}: {fmt(v)}</title></rect>'
        if lo is not None:
            s += f'<line x1="{sc(lo)}" x2="{sc(hi)}" y1="{y + bh / 2}" y2="{y + bh / 2}" class="ci"/>'
            s += f'<line x1="{sc(lo)}" x2="{sc(lo)}" y1="{y + 7}" y2="{y + bh - 7}" class="ci"/><line x1="{sc(hi)}" x2="{sc(hi)}" y1="{y + 7}" y2="{y + bh - 7}" class="ci"/>'
        s += f'<text x="{sc(hi if hi else v) + 8}" y="{y + bh / 2 + 5}" class="val">{fmt(v)}</text>'
    if ref is not None:
        x = sc(ref)
        s += f'<line x1="{x}" x2="{x}" y1="{top - 8}" y2="{h - 26}" class="ref"/><text x="{x + 4}" y="{top - 2}" class="refl">{e(ref_label)}</text>'
    return s + "</svg>"


def grouped_bars(groups, series, values, ymin, ymax, ref=None, ref_label="", fmt=lambda v: f"{v:.1f}", w=640, h=300,
                 ticks=None, unit="", highlight=None, label_vals=True):
    """groups: x축 묶음 이름, series: [(이름, 색)], values[g][s]. highlight[g][s]=True면 굵은 테두리."""
    left, right, top, bottom = 52, 12, 22, 46
    pw, ph = w - left - right, h - top - bottom
    sy = lambda v: top + ph - (v - ymin) / (ymax - ymin) * ph
    gw = pw / len(groups)
    bw = min(34, (gw - 24) / len(series))
    s = svg_open(w, h, "묶음 막대 그림")
    for t in (ticks or np.linspace(ymin, ymax, 5)):
        s += f'<line x1="{left}" x2="{w - right}" y1="{sy(t)}" y2="{sy(t)}" class="grid"/><text x="{left - 6}" y="{sy(t) + 4}" class="tick" text-anchor="end">{fmt(t)}{unit}</text>'
    base = sy(max(ymin, 0)) if ymin < 0 else sy(ymin)
    s += f'<line x1="{left}" x2="{w - right}" y1="{base}" y2="{base}" class="axis"/>'
    for gi, g in enumerate(groups):
        gx = left + gi * gw + (gw - bw * len(series) - 2 * (len(series) - 1)) / 2
        for si, (sn, c) in enumerate(series):
            v = values[gi][si]
            x = gx + si * (bw + 2)
            y0, y1 = (sy(v), base) if v >= 0 else (base, sy(v))
            hl = highlight and highlight[gi][si]
            s += f'<rect x="{x}" y="{y0}" width="{bw}" height="{max(1, y1 - y0)}" rx="3" fill="{c}"{" class=\"hl\"" if hl else ""}><title>{e(g)} · {e(sn)}: {fmt(v)}{unit}</title></rect>'
            if label_vals:
                ty = y0 - 5 if v >= 0 else y1 + 13
                s += f'<text x="{x + bw / 2}" y="{ty}" class="val{" b" if hl else ""}" text-anchor="middle">{fmt(v)}</text>'
        s += f'<text x="{left + gi * gw + gw / 2}" y="{h - bottom + 20}" class="lab" text-anchor="middle">{e(g)}</text>'
    if ref is not None:
        s += f'<line x1="{left}" x2="{w - right}" y1="{sy(ref)}" y2="{sy(ref)}" class="ref"/><text x="{w - right - 4}" y="{sy(ref) - 5}" class="refl" text-anchor="end">{e(ref_label)}</text>'
    return s + "</svg>"


def forest(rows, xmin, xmax, ticks, w=700, label_w=260, unit="%p"):
    """rows: [(label, est, lo, hi, color, strong, note)] 점 + 신뢰구간, 0 기준선."""
    rh, top = 30, 16
    h = top + len(rows) * rh + 34
    sc = lambda v: label_w + (v - xmin) / (xmax - xmin) * (w - label_w - 90)
    s = svg_open(w, h, "추정치와 95% 신뢰구간")
    s += xaxis(label_w, w - 90, h - 28, ticks, sc, lambda t: f"{sgn(t, 0) if t else '0'}", top - 6)
    s += f'<line x1="{sc(0)}" x2="{sc(0)}" y1="{top - 8}" y2="{h - 28}" class="zero"/>'
    for i, (lab, est, lo, hi, c, strong, note) in enumerate(rows):
        y = top + i * rh + rh / 2
        if lab.startswith("§"):
            s += f'<text x="8" y="{y + 5}" class="lab b">{e(lab[1:])}</text>'
            continue
        s += f'<text x="{label_w - 10}" y="{y + 5}" class="lab" text-anchor="end">{e(lab)}</text>'
        s += f'<line x1="{sc(lo)}" x2="{sc(hi)}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2.5"/>'
        s += f'<circle cx="{sc(est)}" cy="{y}" r="{7 if strong else 5.5}" fill="{c if strong else "var(--surface)"}" stroke="{c}" stroke-width="2.5"><title>{e(lab)}: {sgn(est)}{unit} [{sgn(lo)}, {sgn(hi)}]</title></circle>'
        s += f'<text x="{w - 84}" y="{y + 5}" class="val{" b" if strong else ""}">{sgn(est)}{unit} {note}</text>'
    return s + "</svg>"


def lines(series, xmax, ymin, ymax, yticks, w=640, h=300, xlab="step", ylab="", vlines=()):
    """series: [(name, color, xs, mean, lo, hi)] 평균선 + 최소~최대 띠."""
    left, right, top, bottom = 52, 130, 16, 44
    pw, ph = w - left - right, h - top - bottom
    sx = lambda v: left + v / xmax * pw
    sy = lambda v: top + ph - (v - ymin) / (ymax - ymin) * ph
    s = svg_open(w, h, "선 그림")
    for t in yticks:
        s += f'<line x1="{left}" x2="{left + pw}" y1="{sy(t)}" y2="{sy(t)}" class="grid"/><text x="{left - 6}" y="{sy(t) + 4}" class="tick" text-anchor="end">{t:.1f}</text>'
    for t in range(0, xmax + 1, 30):
        s += f'<text x="{sx(t)}" y="{h - bottom + 18}" class="tick" text-anchor="middle">{t}</text>'
    for v, lab in vlines:
        s += f'<line x1="{sx(v)}" x2="{sx(v)}" y1="{top}" y2="{top + ph}" class="ref"/><text x="{sx(v) + 4}" y="{top + 12}" class="refl">{e(lab)}</text>'
    s += f'<line x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}" class="axis"/><text x="{left + pw / 2}" y="{h - 6}" class="tick" text-anchor="middle">{e(xlab)}</text>'
    ends = sorted([(sy(m[-1]), i) for i, (_, _, _, m, _, _) in enumerate(series)])
    label_y, last = {}, -1e9
    for y, i in ends:   # 끝 라벨이 겹치지 않게 최소 15px 간격
        last = max(y, last + 15)
        label_y[i] = last
    for i, (name, c, xs, m, lo, hi) in enumerate(series):
        band = " ".join(f"{sx(x)},{sy(v)}" for x, v in zip(xs, hi)) + " " + " ".join(f"{sx(x)},{sy(v)}" for x, v in zip(xs[::-1], lo[::-1]))
        s += f'<polygon points="{band}" fill="{c}" opacity="0.18"/>'
        s += f'<polyline points="{" ".join(f"{sx(x)},{sy(v)}" for x, v in zip(xs, m))}" fill="none" stroke="{c}" stroke-width="2"><title>{e(name)}</title></polyline>'
        s += f'<text x="{sx(xs[-1]) + 6}" y="{label_y[i] + 4}" class="lab">{e(name)} {m[-1]:.3f}</text>'
    return s + "</svg>"


def strip(rows, xmin, xmax, ticks, ref=None, ref_label="", w=660, label_w=150, unit="%"):
    """rows: [(label, [(value, color, name, own)])] 한 줄에 점 여러 개."""
    rh, top = 40, 18
    h = top + len(rows) * rh + 34
    sc = lambda v: label_w + (v - xmin) / (xmax - xmin) * (w - label_w - 30)
    s = svg_open(w, h, "점 그림")
    s += xaxis(label_w, w - 30, h - 28, ticks, sc, lambda t: f"{t:g}{unit}", top - 8)
    if ref is not None:
        s += f'<line x1="{sc(ref)}" x2="{sc(ref)}" y1="{top - 10}" y2="{h - 28}" class="ref"/><text x="{sc(ref) + 4}" y="{top - 2}" class="refl">{e(ref_label)}</text>'
    for i, (lab, pts) in enumerate(rows):
        y = top + i * rh + rh / 2
        s += f'<text x="{label_w - 10}" y="{y + 5}" class="lab" text-anchor="end">{e(lab)}</text>'
        s += f'<line x1="{label_w}" x2="{w - 30}" y1="{y}" y2="{y}" class="grid"/>'
        for v, c, name, own in sorted(pts, key=lambda p: p[3]):
            s += f'<circle cx="{sc(v)}" cy="{y}" r="{9 if own else 6.5}" fill="{c}" class="dot{" own" if own else ""}"><title>{e(name)} 평가자 → {e(lab)}: {v:.1f}{unit}</title></circle>'
            if own:
                s += f'<text x="{sc(v)}" y="{y - 14}" class="val b" text-anchor="middle">{v:.1f}</text>'
    return s + "</svg>"


def table(head, rows, num_from=1, cls=""):
    th = "".join(f'<th{" class=\"num\"" if i >= num_from else ""}>{h}</th>' for i, h in enumerate(head))
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>' + "".join(f"<tr>{r}</tr>" for r in rows) + "</tbody></table></div>"


def fig(svg, caption, leg=None):
    return f'<figure>{leg or ""}{svg}<figcaption>{caption}</figcaption></figure>'


# ------------------------------------------------------------------ 데이터 읽기
cmp_rows = [l.strip().strip("|").split("|") for l in (ROOT / "classify/output/compare.md").read_text().splitlines()[2:] if l.strip()]
clf = [{"model": r[0].strip(), "cond": r[1].strip(), "acc": float(r[2].strip("% ")), "ci": [float(x) for x in re.findall(r"[\d.]+", r[3])],
        "f1": [float(r[4]), float(r[5]), float(r[6])]} for r in cmp_rows]
style = pd.read_csv(ROOT / "results/output/style_summary.csv")
sft = json.loads((ROOT / "results/output/sft_summary.json").read_text())
val = {t: pd.concat([pd.read_csv(ROOT / f"sft/runs/{t}_seed{s}/val_loss.csv").assign(seed=s) for s in SEEDS]) for t in J}
sel = json.loads((ROOT / "judge/data/selection.json").read_text())
cnt = pd.read_csv(A / "desc_counts.csv"); cnt["seed"] = cnt.seed.astype(str)
leak = pd.read_csv(A / "desc_simple_leak.csv")
pos = pd.read_csv(A / "desc_position.csv"); pos["seed"] = pos.seed.astype(str)
prank = pd.read_csv(A / "desc_position_rank.csv")
agree = pd.read_csv(A / "desc_agreement.csv")
res = pd.read_csv(A / "model_results.csv")
bys = pd.read_csv(A / "model_by_seed.csv")
tests = pd.read_csv(A / "model_seed_tests.csv")
mpos = pd.read_csv(A / "model_position.csv")
overid = pd.read_csv(A / "model_overid.csv")
length = pd.read_csv(A / "model_length.csv")
R = lambda c, m, j: res[(res.criterion == c) & (res.model == m) & (res.judge == j)].iloc[0]
C = lambda c, j, t, s="all": cnt[(cnt.criterion == c) & (cnt.seed == s) & (cnt.judge == j) & (cnt.teacher == t)].iloc[0]

# 연구 질문: result.html의 1절을 그대로 가져온다
rh = (ROOT / "description/result.html").read_text(encoding="utf-8")
rq = rh[rh.index('<h2 id="q">'):rh.index("<!-- =====", rh.index('<h2 id="q">'))]
rq = re.sub(r'<h2 id="q">[^<]*</h2>', "", rq)
rq = rq.replace("→ 4절(최종 순위)", "→ 6절(최종 순위)").replace("→ 6절(기준별 ablation)", "→ 6절(기준별 분석)").replace("→ 4.4, 6절", "→ 6절(seed 반복)")

# ------------------------------------------------------------------ 그림 만들기
fig_clf = fig(hbars([(f"{r['model'].replace('다항 ', '')} · {'원문' if r['cond'] == 'raw' else '정규화'}", r["acc"], r["ci"][0], r["ci"][1]) for r in clf],
                    100, ref=33.3, ref_label="우연 33.3%"),
              "그림 4-1. 답변만 보고 생성 모델을 맞힌 test 정확도와 95% 신뢰구간(test 질문 181개 × 3 = 답변 543개, 질문 단위 bootstrap). 정규화 = 굵게·목록·따옴표·줄바꿈을 모두 지운 텍스트.")
fig_f1 = fig(grouped_bars([f"{r['model'].replace('다항 로지스틱 회귀', '로지스틱').replace('KLUE-', '')} {'원문' if r['cond'] == 'raw' else '정규화'}" for r in clf],
                          [(N[t], COL[t]) for t in J], [r["f1"] for r in clf], 0.8, 1.0, fmt=lambda v: f"{v:.2f}", ticks=[0.8, 0.85, 0.9, 0.95, 1.0], h=280),
             "그림 4-2. 생성 모델별 F1 점수. 어느 모델도 특별히 헷갈리지 않고 세 모델 모두 0.91 이상으로 구별된다.",
             legend([(N[t], COL[t]) for t in J]))
feat = ["굵게(**) 사용", "글머리 목록(-, *) 사용", "번호 목록(1.) 사용", "영문 괄호 병기", "요약형 마무리", "완곡 표현(~수 있습니다 등)"]
fv = [[style[style["지표"] == f][N[t] if t != "gpt" else "GPT"].iloc[0] * 100 for t in J] for f in feat]
fig_style = fig(grouped_bars(["굵게", "글머리 목록", "번호 목록", "영문 괄호", "요약 마무리", "완곡 표현"], [(N[t], COL[t]) for t in J], fv, 0, 100,
                             fmt=lambda v: f"{v:.0f}", ticks=[0, 25, 50, 75, 100], unit="", h=290),
                "그림 4-3. 교사 답변 3,597개에서 각 형식·문체 특징이 나타난 답변 비율(%). 같은 질문, 같은 시스템 프롬프트로 생성했다.",
                legend([(N[t], COL[t]) for t in J]))

ser = []
for t in J:
    g = val[t].groupby("step").eval_loss
    xs = g.mean().index.tolist()
    ser.append((N[t] + " 학생", COL[t], xs, g.mean().values, g.min().values, g.max().values))
fig_val = fig(lines(ser, 189, 2.1, 2.7, [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7], vlines=[(63, "1 epoch"), (126, "2 epoch")]),
              "그림 5-1. 학습 중 val loss의 변화. 학습에 쓰지 않은 질문 199개에서 자기 교사의 답에 대한 loss(정의는 5.2). 낮을수록 학생이 교사의 답을 \'자기가 쓸 법한 글\'로 본다. step 0(학습 전), 10 step마다, 마지막 step 189에 측정. 선은 seed 5개 평균, 옅은 띠는 seed 간 최소~최대(차이가 매우 작아 선과 거의 겹친다). 점선은 epoch 경계.",
              legend([(N[t] + " 학생", COL[t]) for t in J]))
cl = sft["cross_loss_mean"]; base = sft["base_loss"]
red = [[base[a] - cl[a][s] for a in J] for s in J]   # 학생 s가 교사 a 답변에서 줄인 loss
fig_cross = fig(grouped_bars([f"{N[s]} 학생" for s in J], [(f"{N[a]} 답변", COL[a]) for a in J], red, 0, 0.4, fmt=lambda v: f"{v:.2f}",
                             ticks=[0, 0.1, 0.2, 0.3, 0.4], highlight=[[a == s for a in J] for s in J], h=280),
                "그림 5-2. 교차 loss 감소량 = 표 5-2의 학습 전 Qwen 값 − 학생 값. 막대 묶음 = 학생, 막대 색 = 보여 준 교사의 답변. 클수록 학습으로 그 교사의 답에 더 가까워졌다(seed 5개 평균). 굵은 테두리 = 자기 교사의 답.",
                legend([(f"{N[a]} 답변", COL[a]) for a in J]))

fig_top1 = fig(strip([(f"{N[t]} 학생", [(C("overall", j, t).top1_rate, COL[j], N[j], j == t) for j in J]) for t in J], 15, 50, [20, 30, 40, 50],
                     ref=33.3, ref_label="기준선 33.3%"),
               "그림 6-1. 같은 학생의 답을 세 평가자가 1위로 놓은 비율(최종 순위, 평가자·학생마다 1,500번). 점 색 = 평가자, 큰 점과 숫자 = 자기 계열 평가자.",
               legend([(f"{N[j]} 평가자", COL[j]) for j in J]))
frows = []
for j in J:
    frows.append((f"§{N[j]} 평가자", 0, 0, 0, "", False, ""))
    for m, lab in [("PL_top1", "① 1위 모형 · 1위 확률 차"), ("PL", "② 순위 로짓 모형(PL) · 승률 차"), ("LPM", "③ 쌍 고정효과 모형(LPM) · 승률 차")]:
        r = R("overall", m, j)
        frows.append((lab, r.pp, r.pp_ci_low, r.pp_ci_high, COL[j], r.p_holm < 0.05, star(r.p_holm)))
fig_forest = fig(forest(frows, -6, 15, [-5, 0, 5, 10, 15]),
                 "그림 6-2. 주 결과: 평가자별 자기 계열 효과(최종 순위). 점 = 추정치(%p), 선 = 95% 신뢰구간, 0 = 편향 없음. 채운 점·별표 = Holm 보정 p &lt; 0.05. 학생 품질과 답변 위치를 통제한 값이다.")
fig_seed = fig(strip([(f"{N[j]} 평가자", [(bys[(bys.criterion == "overall") & (bys.model == "PL") & (bys.judge == j) & (bys.seed == s)].pp.iloc[0], COL[j], f"seed {s}", False) for s in SEEDS]) for j in J],
                     -6, 16, [-5, 0, 5, 10, 15], ref=0, ref_label="0", unit="%p"),
               "그림 6-3. 다섯 번의 독립 학습(seed 42~46)에서 따로 추정한 자기 계열 효과(순위 로짓 PL, %p). 점 하나 = seed 하나.")
crow = []
for c in CRIT:
    crow.append((f"§{CN[c]}", 0, 0, 0, "", False, ""))
    for j in J:
        r = R(c, "PL", j)
        crow.append((f"{N[j]} 평가자", r.pp, r.pp_ci_low, r.pp_ci_high, COL[j], r.p_holm < 0.05, star(r.p_holm)))
fig_crit = fig(forest(crow, -6, 15, [-5, 0, 5, 10, 15]),
               "그림 6-4. 평가 기준별 자기 계열 효과(순위 로짓 PL, %p, 95% CI). 채운 점 = 기준 4 × 평가자 3 = 12개 검정에 대한 Holm 보정 p &lt; 0.05.")
fig_pos = fig(grouped_bars(["A (첫 번째)", "B", "C (마지막)"], [(f"{N[j]} 평가자", COL[j]) for j in J],
                           [[pos[(pos.criterion == "overall") & (pos.judge == j) & (pos.seed == "all")].iloc[0][f"top1_{p}"] for j in J] for p in "ABC"],
                           0, 60, ref=33.3, ref_label="편향 없음 33.3%", fmt=lambda v: f"{v:.0f}", ticks=[0, 15, 30, 45, 60], unit="%", h=280),
               "그림 6-5. 평가자가 1위로 뽑은 답이 놓여 있던 위치(최종 순위, 평가자마다 1,500번). 각 학생은 A·B·C에 정확히 500번씩 놓였으므로, 편향이 없다면 세 위치 모두 33.3%여야 한다.",
               legend([(f"{N[j]} 평가자", COL[j]) for j in J]))
simple_vs = [[leak[(leak.criterion == "overall") & (leak.metric == "win_rate") & (leak.judge == j)].leak.iloc[0], R("overall", "PL", j).pp] for j in J]
fig_simple = fig(grouped_bars([f"{N[j]} 평가자" for j in J], [("단순 계산 (다른 두 평가자 평균 대비)", "var(--c-neutral2)"), ("② 순위 로짓 모형(PL) (세 평가자 동시 추정)", "var(--c-neutral)")],
                              simple_vs, -3, 11, fmt=lambda v: sgn(v), ticks=[-2.5, 0, 2.5, 5, 7.5, 10], unit="", h=280),
                 "그림 7-1. 같은 주 지표(쌍대 승률)로 잰 자기 계열 효과(%p): 단순 계산 vs 통계 모형. 단순 계산에서는 Gemini 평가자의 강한 자기 우대가 다른 평가자의 기준선을 낮춰 GPT·Claude의 값이 부풀려진다. 모형으로 세 평가자를 동시에 추정하면 GPT는 줄고 Claude는 0 아래로 내려간다.",
                 legend([("단순 계산", "var(--c-neutral2)"), ("② 순위 로짓 모형(PL)", "var(--c-neutral)")]))

# ------------------------------------------------------------------ 표 만들기
def own(j, t):
    return ' class="num own"' if j == t else ' class="num"'


t_clf = table(["분류기", "텍스트", "test 정확도", "95% CI", "GPT F1", "Claude F1", "Gemini F1"],
              [f"<td>{r['model']}</td><td>{'원문' if r['cond'] == 'raw' else '정규화'}</td><td class='num'><b>{r['acc']:.1f}%</b></td><td class='num'>{r['ci'][0]:.1f}–{r['ci'][1]:.1f}%</td>"
               + "".join(f"<td class='num'>{v:.3f}</td>" for v in r["f1"]) for r in clf], num_from=2)
t_sft = table(["학생", "학습 전 val loss", "최종 val loss (평균 ± SD)", "감소량", "최저 val (step)", "학습 시간"],
              [f"<td>{N[t]} 학생 × 5</td><td class='num'>{d['val_before']:.3f}</td><td class='num'>{d['val_final_mean']:.4f} ± {d['val_final_sd']:.4f}</td><td class='num'><b>{d['val_final_mean'] - d['val_before']:+.3f}</b></td><td class='num'>{d['val_min_mean']:.3f} ({d['val_min_step']})</td><td class='num'>{d['runtime_min_mean']:.1f}분</td>"
               for t, d in sft["teachers"].items()])
t_cross = table(["학생 \\ 평가한 답변", "GPT 답변", "Claude 답변", "Gemini 답변"],
                [f"<td>학습 전 Qwen</td>" + "".join(f"<td class='num'>{base[a]:.3f}</td>" for a in J)] +
                [f"<td>{N[s]} 학생</td>" + "".join(f"<td{own(s, a)}>{cl[a][s]:.3f}</td>" for a in J) for s in J])
t_complete = table(["학생", "seed 42", "43", "44", "45", "46"],
                   [f"<td>{N[t]} 학생</td>" + "".join(f"<td class='num'>{sel['complete_answers_per_student'][f'{t}_seed{s}']}</td>" for s in SEEDS) for t in J])
t_counts = table(["평가자 \\ 학생", "GPT 학생", "Claude 학생", "Gemini 학생"],
                 [f"<td>{N[j]} 평가자</td>" + "".join(f"<td{own(j, t)}>{C('overall', j, t).rank1} / {C('overall', j, t).rank2} / {C('overall', j, t).rank3}</td>" for t in J) for j in J])
t_rates = table(["평가자 \\ 학생", "1위 비율: GPT", "Claude", "Gemini", "쌍대 승률: GPT", "Claude", "Gemini"],
                [f"<td>{N[j]} 평가자</td>" + "".join(f"<td{own(j, t)}>{C('overall', j, t).top1_rate:.1f}%</td>" for t in J)
                 + "".join(f"<td{own(j, t)}>{C('overall', j, t).win_rate:.1f}%</td>" for t in J) for j in J])


def leak_rows(c):
    rows = []
    for j in J:
        l = leak[(leak.criterion == c) & (leak.metric == "top1_rate") & (leak.judge == j)].iloc[0]
        cells = f"<td>{N[j]} 평가자</td><td class='num'>{l.own:.1f}% vs {l.others:.1f}% <b>({sgn(l.leak)})</b></td>"
        for m in ["PL_top1", "PL", "LPM"]:
            r = R(c, m, j)
            cells += f"<td class='num{' sig' if r.p_holm < 0.05 else ''}'><b>{sgn(r.pp)}</b>{star(r.p_holm)}<br><span class='m'>[{sgn(r.pp_ci_low)}, {sgn(r.pp_ci_high)}] · p<sub>Holm</sub> {pf(r.p_holm)}</span></td>"
        t = tests[(tests.criterion == c) & (tests.model == "PL") & (tests.judge == j)].iloc[0]
        cells += f"<td class='num'>{int(t.n_positive)}/5</td>"
        rows.append(cells)
    return rows


LH = ["평가자", "단순 계산 (1위 비율)<br>자기 평가자 vs 다른 평가자 평균", "① 1위 모형<br>1위 확률 차 %p", "② 순위 로짓 모형(PL)<br>승률 차 %p (주 결과)", "③ 쌍 고정효과 모형(LPM)<br>승률 차 %p", "PL: 양수인 seed"]
t_leak = table(LH, leak_rows("overall"))
abl = []
for c in ["overall"] + CRIT:
    for k, j in enumerate(J):
        cells = (f"<td rowspan='3'>{'<b>' if c == 'overall' else ''}{CN[c]}{'</b>' if c == 'overall' else ''}</td>" if k == 0 else "") + f"<td>{N[j]}</td>"
        for m in ["PL_top1", "PL", "LPM"]:
            r = R(c, m, j)
            cells += f"<td class='num{' sig' if r.p_holm < 0.05 else ''}'>{sgn(r.pp)}{star(r.p_holm)}</td>"
        cells += f"<td class='num'>{int(tests[(tests.criterion == c) & (tests.model == 'PL') & (tests.judge == j)].n_positive.iloc[0])}/5</td>"
        if k == 0:
            cells += f"<td rowspan='3' class='num'>{pf(overid[overid.criterion == c].p.iloc[0])}</td>"
        abl.append(cells)
t_abl = table(["결과", "평가자", "① 1위 모형<br>1위 확률 차", "② 순위 로짓 모형(PL)<br>승률 차", "③ 쌍 고정효과 모형(LPM)<br>승률 차", "PL: 양수인 seed", "식별 가정 검정 p"], abl, num_from=2)
prow = []
for c in ["overall"] + CRIT:
    cells = f"<td>{CN[c]}</td>"
    for j in J:
        r = pos[(pos.criterion == c) & (pos.judge == j) & (pos.seed == "all")].iloc[0]
        oa = mpos[(mpos.criterion == c) & (mpos.judge == j) & (mpos.position == "A")].odds_ratio_vs_C.iloc[0]
        cells += f"<td class='num'>{r.top1_A:.1f}% · {oa:.2f}배 · {r.same_top1_groups / 5:.0f}%</td>"
    prow.append(cells)
t_pos = table(["결과", "GPT 평가자", "Claude 평가자", "Gemini 평가자"], prow)
prk = []
for j in J:
    for k, t in enumerate(J):
        r = prank[(prank.criterion == "overall") & (prank.judge == j) & (prank.teacher == t)].iloc[0]
        prk.append((f"<td rowspan='3'>{N[j]} 평가자</td>" if k == 0 else "") + f"<td>{N[t]} 학생</td><td class='num'>{r.A:.2f}</td><td class='num'>{r.B:.2f}</td><td class='num'>{r.C:.2f}</td><td class='num'>{sgn(r.C - r.A, 2)}</td>")
t_prank = table(["평가자", "학생", "A에 놓였을 때", "B", "C", "A→C"], prk, num_from=2)
srows = []
for j in J:
    for k, (m, lab) in enumerate([("PL_top1", "① 1위 모형"), ("PL", "② 순위 로짓 모형(PL)"), ("LPM", "③ 쌍 고정효과 모형(LPM)")]):
        vals = [bys[(bys.criterion == "overall") & (bys.model == m) & (bys.judge == j) & (bys.seed == s)].pp.iloc[0] for s in SEEDS]
        t = tests[(tests.criterion == "overall") & (tests.model == m) & (tests.judge == j)].iloc[0]
        srows.append((f"<td rowspan='3'>{N[j]}</td>" if k == 0 else "") + f"<td>{lab}</td>" + "".join(f"<td class='num'>{sgn(v)}</td>" for v in vals)
                     + f"<td class='num'>{int(t.n_positive)}/5</td><td class='num'>{pf(t.t_p_one_sided)} ({pf(t.t_p_holm)})</td>")
t_seed = table(["평가자", "모형", "seed 42", "43", "44", "45", "46", "양수", "t검정 p (Holm)"], srows, num_from=2)

g = lambda m, j: R("overall", m, j)
gm, gp, gl = g("PL_top1", "gemini"), g("PL", "gemini"), g("LPM", "gemini")
pm, pp_, pl = g("PL_top1", "gpt"), g("PL", "gpt"), g("LPM", "gpt")
cm, cp, cl_ = g("PL_top1", "claude"), g("PL", "claude"), g("LPM", "claude")
posA = {j: pos[(pos.criterion == "overall") & (pos.judge == j) & (pos.seed == "all")].iloc[0].top1_A for j in J}
orA = {j: mpos[(mpos.criterion == "overall") & (mpos.judge == j) & (mpos.position == "A")].odds_ratio_vs_C.iloc[0] for j in J}

ABSTRACT = (
    "대규모 언어모델(LLM)이 생성한 합성데이터로 소형 모델을 학습하고 그 성능을 다시 LLM 평가자로 측정하는 방식이 널리 쓰이지만, "
    "평가자가 자기 모델의 데이터로 학습한 모델을 더 높게 평가하는 선호 누출(preference leakage)이 평가 결과를 왜곡할 수 있다. "
    "본 연구는 한국어 합성데이터 증류 환경에서 이 현상을 추정하였다. KoAlpaca 질문 1,000개에 대해 GPT, Claude, Gemini가 생성한 답변으로 "
    "Qwen3-0.6B-Base를 교사별로 다섯 번씩 미세조정하여 학생 모델 15개를 만들었다. 먼저 세 교사의 답변이 답변만으로 구별됨을 확인하였고"
    f"(다항 로지스틱 회귀 {clf[0]['acc']:.1f}%, KLUE-RoBERTa {clf[2]['acc']:.1f}%, 서식 제거 후에도 {clf[1]['acc']:.1f}%, {clf[3]['acc']:.1f}%), "
    "학생 15개 모두 자기 교사의 답변에서 가장 낮은 교차 손실을 보였다. 이어 세 교사 모델을 평가자로 하여, 같은 질문에 대한 학생 세 개의 답변을 "
    "제시 순서를 바꿔 세 번씩 비교하는 listwise 평가 4,500회를 수행하고, 학생 품질과 답변 위치를 통제한 순위 로짓(Plackett-Luce) 모형과 "
    f"쌍 고정효과 선형확률모형으로 평가자별 자기 계열 효과를 추정하였다. 그 결과 Gemini 평가자는 품질이 같은 상대를 이길 확률을 "
    f"{gp.pp:.1f}%p(95% 신뢰구간 {gp.pp_ci_low:.1f}–{gp.pp_ci_high:.1f}) 높게 판정하였고, 이 효과는 다섯 번의 독립 학습 모두와 네 평가 기준 중 세 기준에서 일관되게 나타났다. "
    f"GPT 평가자는 약한 효과({pp_.pp:.1f}–{pm.pp:.1f}%p)를 보였고 Claude 평가자에서는 유의한 효과가 관찰되지 않았다. "
    f"또한 첫 번째 위치의 답변이 1위로 선택될 오즈가 마지막 위치보다 {min(orA.values()):.1f}–{max(orA.values()):.1f}배 높아 선호 누출보다 큰 위치 편향이 확인되었다. "
    "이는 선호 누출이 평가자 모델에 따라 결론을 바꿀 만큼 클 수 있으므로, LLM 평가에서 생성 모델과 평가 모델의 관계와 제시 순서를 설계와 분석 단계에서 통제해야 함을 시사한다."
)
KEYWORDS = "선호 누출, LLM 평가자, 합성데이터 증류, 순위 로짓 모형, 위치 편향"


# ------------------------------------------------------------------ 6.2 측정 방법 (수식)
gg, cg, mg = C("overall", "gpt", "gpt"), C("overall", "claude", "gpt"), C("overall", "gemini", "gpt")
lk = leak[(leak.criterion == "overall") & (leak.metric == "top1_rate") & (leak.judge == "gpt")].iloc[0]
lw = leak[(leak.criterion == "overall") & (leak.metric == "win_rate") & (leak.judge == "gpt")].iloc[0]
# ------------------------------------------------------------------ 6.2 (5) 용어 정리와 순위 로짓 숫자 예제
dA, dB, lamG = 0.6, 0.2, float(np.log(0.7 / 0.3))
ex_u = {"A": dA, "B": dB, "C": lamG}
ex_e = {k: float(np.exp(v)) for k, v in ex_u.items()}
ex_sum = sum(ex_e.values())
ex_p1 = {k: v / ex_sum for k, v in ex_e.items()}
ex_p2 = ex_e["A"] / (ex_e["A"] + ex_e["B"])
SEC_PL_EASY = f"""
  <div class="note"><b>용어 정리 (이 포스터 전체에서 같은 뜻으로 쓴다)</b>
  {table(["구분", "이름", "뜻", "단위"], [
      "<td>지표 (원자료 요약)</td><td><b>1위 비율</b></td><td>평가자가 그 학생을 1위로 놓은 비율</td><td>%, 기준 33.3%</td>",
      "<td>지표 (원자료 요약)</td><td><b>쌍대 승률</b></td><td>순위를 1:1 대결로 풀었을 때 그 학생이 이긴 비율</td><td>%, 기준 50%</td>",
      "<td>계산</td><td><b>단순 계산</b></td><td>같은 학생에 대해 [자기 계열 평가자의 지표] − [다른 두 평가자 지표의 평균]. (4)의 함정이 있다</td><td>%p</td>",
      "<td>통계 모형</td><td><b>① 1위 모형</b> (보조)</td><td>순위 중 '1위 선택'만 써서 자기 계열 효과 λ를 추정</td><td><b>1위 확률 차</b> %p (33.3% 대비)</td>",
      "<td>통계 모형</td><td><b>② 순위 로짓 모형(PL)</b> (주 모형)</td><td>1·2·3위 순위 전체로 λ를 추정</td><td><b>승률 차</b> %p (50% 대비)</td>",
      "<td>통계 모형</td><td><b>③ 쌍 고정효과 모형(LPM)</b> (강건성)</td><td>1:1 대결 자료에서 같은 두 답의 품질 차이를 지우고 λ를 추정</td><td><b>승률 차</b> %p</td>"], num_from=9)}
  <p style="margin-bottom:0"><b>쌍대 승률은 지표이고 LPM은 모형이다.</b> 쌍대 승률은 원자료를 요약한 숫자다. ②·③은 그 승률에서 품질·위치·다른 평가자의 편향을 걸러 내고 남는 "자기 계열 때문에 오른 승률(승률 차)"을 추정하는 모형이다.</p></div>

  <h3>순위 로짓 모형을 쉽게: 숫자로 따라가기</h3>
  <p><b>1단계, 점수.</b> 평가자가 답 세 개를 볼 때, 답마다 보이지 않는 점수 <i>u</i>를 매긴다고 본다. 점수는 세 부분의 합이다.</p>
<pre>점수 u = 학생 품질 θ  +  위치 가산점 δ  +  (평가자와 같은 계열이면) 자기 계열 가산점 λ</pre>
  <p><b>2단계, 점수 → 순위 확률.</b> 점수가 높을수록 1위로 뽑힐 확률이 크다. 1위로 뽑힐 확률은 e<sup>u</sup>를 세 답의 e<sup>u</sup> 합으로 나눈 값이고, 1위가 정해지면 남은 두 답 중에서 같은 방식으로 2위를 고른다.</p>
  <p><b>예.</b> Gemini 평가자에게 A = GPT 학생, B = Claude 학생, C = Gemini 학생의 답을 보여 줬다. 세 학생의 품질은 같고(θ = 0), 위치 가산점은 A {dA}, B {dB}, C 0, Gemini 평가자의 자기 계열 가산점은 λ = {lamG:.2f}(품질이 같은 상대를 70% 확률로 이기게 만드는 크기)라고 하자.</p>
  {table(["답", "학생", "품질 θ", "위치 δ", "자기 계열 λ", "점수 u", "e<sup>u</sup>", "1위로 뽑힐 확률"], [
      f"<td>A</td><td>GPT 학생</td><td class='num'>0</td><td class='num'>{dA}</td><td class='num'>0</td><td class='num'>{ex_u['A']:.2f}</td><td class='num'>{ex_e['A']:.2f}</td><td class='num'>{ex_e['A']:.2f} ÷ {ex_sum:.2f} = <b>{ex_p1['A'] * 100:.0f}%</b></td>",
      f"<td>B</td><td>Claude 학생</td><td class='num'>0</td><td class='num'>{dB}</td><td class='num'>0</td><td class='num'>{ex_u['B']:.2f}</td><td class='num'>{ex_e['B']:.2f}</td><td class='num'>{ex_e['B']:.2f} ÷ {ex_sum:.2f} = <b>{ex_p1['B'] * 100:.0f}%</b></td>",
      f"<td>C</td><td>Gemini 학생</td><td class='num'>0</td><td class='num'>0</td><td class='num'>{lamG:.2f}</td><td class='num'>{ex_u['C']:.2f}</td><td class='num'>{ex_e['C']:.2f}</td><td class='num'>{ex_e['C']:.2f} ÷ {ex_sum:.2f} = <b>{ex_p1['C'] * 100:.0f}%</b></td>"], num_from=2)}
<pre>순위 "C ≻ A ≻ B"가 나올 확률
  = P(C가 1위) × P(남은 A, B 중 A가 2위)
  = {ex_p1['C']:.2f} × {ex_e['A']:.2f} ÷ ({ex_e['A']:.2f} + {ex_e['B']:.2f})
  = {ex_p1['C']:.2f} × {ex_p2:.2f} = {ex_p1['C'] * ex_p2:.2f}</pre>
  <p>품질이 같은데도 Gemini 학생이 1위로 뽑힐 확률이 {ex_p1['C'] * 100:.0f}%로 가장 높다. λ 때문이다. A에 놓인 GPT 학생이 B의 Claude 학생보다 높은 것은 δ(위치) 때문이다.</p>
  <p><b>3단계, 추정.</b> 실제로는 θ, δ, λ를 모른다. 모형은 4,500개의 실제 순위를 보고, <b>그 순위들이 나올 확률이 가장 커지는</b> θ, δ, λ 값을 찾는다(최대우도법). 위 계산을 4,500번 하고 값을 조금씩 바꿔 가며 가장 잘 맞는 조합을 고르는 것이다.</p>
  <p><b>4단계, 왜 통제가 되는가.</b> (4)의 예시(Gemini 평가자만 편향)를 모형은 다음처럼 설명한다. 모든 순위를 <b>세 모수 중 가장 잘 맞는 것</b>에 배정하기 때문에, 한 평가자의 편향이 다른 학생이나 다른 평가자에게 넘어가지 않는다.</p>
  {table(["자료에서 보이는 패턴", "모형이 설명하는 방법", "결과"], [
      "<td>어느 평가자든, 어느 학생이든 A에 놓이면 1위가 잦다</td><td>모든 학생에게 똑같이 일어나므로 위치 가산점 δ<sub>A</sub>가 설명</td><td>위치 효과는 누구의 품질이나 편향에도 들어가지 않음</td>",
      "<td>세 평가자 모두 GPT 학생 vs Claude 학생은 50:50</td><td>품질 θ<sub>GPT</sub> = θ<sub>Claude</sub>, 이 쌍에서 GPT·Claude 평가자의 가산점 없음</td><td>λ<sub>GPT</sub> = λ<sub>Claude</sub> = 0</td>",
      "<td>Claude 평가자(중립)는 GPT 학생 vs Gemini 학생 50:50인데 Gemini 평가자는 30:70</td><td>품질은 같고(중립 심판 기준) Gemini 평가자만 Gemini 학생에게 가산점</td><td>λ<sub>Gemini</sub> = 0.85 (승률 차 +20%p)</td>",
      "<td>Gemini 평가자 아래에서 GPT·Claude 학생의 승률이 40%로 낮다</td><td>옆의 Gemini 학생이 λ<sub>Gemini</sub>만큼 올라가 밀려난 것으로 설명됨. GPT·Claude 학생의 품질 θ나 GPT·Claude 평가자의 λ로 돌리지 않음</td><td>가짜 누출이 생기지 않음</td>"], num_from=9)}
  <div class="note" style="margin-bottom:0"><b>핵심 차이.</b> 단순 계산은 "Gemini 평가자가 GPT 학생에게 준 40%"를 GPT 학생의 기준선에 그대로 넣는다. 그래서 공정한 GPT 평가자가 +5%p 편향된 것처럼 보인다. 순위 로짓 모형은 그 40%가 <b>Gemini 학생의 가산점 λ<sub>Gemini</sub> 때문에 밀려난 결과</b>라는 것을 같은 순위 안의 정보로 알아낸다. Gemini 평가자 아래에서도 GPT 학생 vs Claude 학생은 50:50이고, 낮아지는 것은 Gemini 학생과 함께 있을 때뿐이기 때문이다. 그래서 각 평가자의 λ만 깨끗하게 남는다.</div>
"""

# ------------------------------------------------------------------ 6.3~6.7 결과 본문 (표마다 읽는 법·해석)
cr = {(j, t): C("overall", j, t) for j in J for t in J}
lt = {j: leak[(leak.criterion == "overall") & (leak.metric == "top1_rate") & (leak.judge == j)].iloc[0] for j in J}
lw2 = {j: leak[(leak.criterion == "overall") & (leak.metric == "win_rate") & (leak.judge == j)].iloc[0] for j in J}
M = {(m, j): R("overall", m, j) for m in ["PL_top1", "PL", "LPM"] for j in J}
seedv = {j: [bys[(bys.criterion == "overall") & (bys.model == "PL") & (bys.judge == j) & (bys.seed == s_)].pp.iloc[0] for s_ in SEEDS] for j in J}
pr = {(j, t): prank[(prank.criterion == "overall") & (prank.judge == j) & (prank.teacher == t)].iloc[0] for j in J for t in J}
# ------------------------------------------------------------------ 6.4 길이 통제 민감도 분석
_o = pd.read_csv(ROOT / "judge/output/ranks_long.csv"); _o = _o[_o.criterion == "overall"]
_m = _o.merge(_o, on=["judge", "group_id", "rotation"], suffixes=("_a", "_b"))
_m = _m[(_m.teacher_a < _m.teacher_b) & (_m.n_tokens_a != _m.n_tokens_b)]
longer = (((_m.n_tokens_a > _m.n_tokens_b) == (_m.rank_a < _m.rank_b)).groupby(_m.judge).mean() * 100)
tok_mean = _o.drop_duplicates(["teacher", "seed", "question_id"]).groupby("teacher").n_tokens.mean()
len_rows = []
for j in J:
    a, b = R("overall", "PL", j), length[length.judge == j].iloc[0]
    len_rows.append(f"<td>{N[j]} 평가자</td><td class='num'>{longer[j]:.1f}%</td><td class='num'>{b.len_slope:+.2f}".replace("-", "−") + "</td>"
                    f"<td class='num'>{sgn(a.pp)} [{sgn(a.pp_ci_low)}, {sgn(a.pp_ci_high)}]</td>"
                    f"<td class='num{' sig' if b.p < 0.05 else ''}'><b>{sgn(b.pp)}</b> [{sgn(b.pp_ci_low)}, {sgn(b.pp_ci_high)}]<br><span class='m'>p {pb(b.p, 5)}</span></td>")
t_len = table(["평가자", "길이가 다른 두 답 중<br>긴 답이 이긴 비율", "길이 기울기 β<sub>j</sub>", "② 순위 로짓 모형(PL)<br>승률 차 (길이 통제 전)", "② + 평가자별 길이 항<br>승률 차 (길이 통제 후)"], len_rows)
SEC_LEN = f"""
  <h3>6.4.1 길이 통제: 자기 계열 선호가 답 길이 때문은 아닌가</h3>
  <p><b>왜 확인했나.</b> LLM 평가자는 긴 답을 선호하는 길이 편향이 있다고 알려져 있고, 학생 답의 길이는 교사 계열마다 다르다(평균 GPT 학생 {tok_mean['gpt']:.0f}, Gemini 학생 {tok_mean['gemini']:.0f}, Claude 학생 {tok_mean['claude']:.0f}토큰). 만약 평가자마다 길이를 좋아하는 정도가 다르다면, 그 차이가 자기 계열 선호(λ)처럼 보일 수 있다. 예를 들어 Gemini 평가자만 중간 길이 답을 좋아한다면 Gemini 학생이 유리해진다.</p>
  <p><b>어떻게 했나.</b> ② 순위 로짓 모형(PL)의 효용 식에 <b>평가자별 길이 항</b>을 추가해 다시 적합했다(<code>judge/analyze_models.py</code>, 결과 <code>model_length.csv</code>).</p>
  <div class="eq"><i>u<sub>jit</sub></i> = <i>θ</i><sub><i>s</i>(<i>i</i>),<i>t</i></sub> + <i>δ</i><sub><i>j</i>,<i>a</i>(<i>i</i>,<i>t</i>)</sub> + <i>λ<sub>j</sub></i> · 𝟙[<i>t</i> = <i>j</i>] + <i>β<sub>j</sub></i> · <i>z</i><sub>len</sub>(<i>i</i>,<i>t</i>)</div>
  <ul>
    <li><i>z</i><sub>len</sub>: 그 학생 답의 생성 토큰 수에 log를 씌우고 표준화한 값(평균 0, 표준편차 1). 질문마다 다른 실제 길이를 쓴다.</li>
    <li><i>β<sub>j</sub></i>: 평가자 <i>j</i>의 길이 기울기. 양수면 긴 답 선호, 음수면 짧은 답 선호. 평가자마다 따로 추정해 "평가자마다 길이 취향이 다르다"는 가능성까지 걸러 낸다.</li>
    <li>이 항을 넣은 뒤에도 <i>λ<sub>j</sub></i>가 남으면, 그 몫은 길이로 설명되지 않는 자기 계열 선호다. 신뢰구간은 같은 질문 단위 부트스트랩(계산량 때문에 200회, 그래서 p의 최솟값은 0.005)으로 구했다.</li>
  </ul>
  {t_len}
  <p class="m">표 6-3b. 첫 열은 원자료(최종 순위에서 길이가 다른 두 학생 답을 1:1로 비교, 50%면 길이 무관). β는 표준화 log 길이 1단위당 로그 오즈. 음영 = p &lt; 0.05.</p>
  <p><b>표 6-3b 읽는 법과 해석.</b></p>
  <ul>
    <li><b>세 평가자 모두 오히려 짧은 답을 약간 선호했다:</b> 긴 답이 이긴 비율이 {longer.min():.0f}~{longer.max():.0f}%로 50%보다 낮고, β도 모두 음수다. 그리고 세 평가자의 길이 취향이 비슷하다. 따라서 길이 차이가 특정 평가자의 자기 계열 선호를 만들어 냈을 가능성은 작다.</li>
    <li><b>Gemini:</b> 길이를 통제해도 {sgn(R('overall','PL','gemini').pp)} → {sgn(length[length.judge=='gemini'].pp.iloc[0])}%p로 거의 그대로이고 신뢰구간이 0에서 멀다. <b>Gemini의 자기 계열 선호는 길이로 설명되지 않는다.</b></li>
    <li><b>GPT:</b> {sgn(R('overall','PL','gpt').pp)} → {sgn(length[length.judge=='gpt'].pp.iloc[0])}%p로 오히려 커졌다. GPT 학생이 가장 짧고 세 평가자 모두 짧은 답을 좋아하는데, 짧은 답 선호가 GPT 평가자에게서 가장 약하다(β {length[length.judge=='gpt'].len_slope.iloc[0]:+.2f}). 길이를 통제하지 않으면 이 몫이 GPT 학생의 품질 θ로 들어가 λ<sub>GPT</sub>를 가린다. 다만 이 값은 길이 모형을 넣었을 때만 유의해 모형 설정에 민감하므로, 주 결과(② 순위 로짓 모형)에 따라 GPT는 계속 "약한 증거"로 둔다.</li>
    <li><b>Claude:</b> {sgn(R('overall','PL','claude').pp)} → {sgn(length[length.judge=='claude'].pp.iloc[0])}%p, 여전히 0 근처다.</li>
    <li>③ 쌍 고정효과 모형(LPM)은 같은 두 답 쌍마다 고정효과를 두므로 길이 차이가 처음부터 흡수되어 있고, 그 결과도 Gemini +10.7%p로 같은 결론이다.</li>
  </ul>
"""

SEC6R = f"""
  <h3>6.3 결과 ① 원자료: 평가자는 각 학생을 몇 번 1위·2위·3위로 놓았나 (최종 순위)</h3>
  <p><b>무엇을 보는가.</b> 모형에 넣기 전의 원자료다. 평가자 한 명이 학생 한 명에게 매긴 순위 1,500번(질문 100 × seed 5 × 순서 3)을 센다.</p>
  {t_counts}
  <p class="m">표 6-1. 각 칸 = 1위 / 2위 / 3위 횟수(1,500번 중). 굵은 칸 = 자기 계열 평가(예: GPT 평가자 × GPT 학생).</p>
  {t_rates}
  <p class="m">표 6-2. 표 6-1을 1위 비율(기준 33.3%)과 쌍대 승률(기준 50%)로 바꾼 것.</p>
  <div class="note"><b>표 6-1·6-2 읽는 법: 가로로 읽지 말고 세로로 읽는다.</b>
  <ul style="margin-bottom:0">
    <li><b>가로(한 평가자의 행):</b> 한 평가자가 세 학생을 어떻게 나눠 줬는지다. 예: GPT 평가자 행은 GPT 학생 1위 {cr[('gpt','gpt')].rank1}번, Gemini 학생 {cr[('gpt','gemini')].rank1}번이다. 하지만 이 차이에는 <b>학생의 실제 품질 차이와 그 평가자의 자기 계열 가산점이 섞여</b> 있어서, GPT 학생이 정말 나은지 GPT 평가자가 편든 것인지 가로만으로는 구분할 수 없다. 그래서 가로로는 편향이 보이지 않는다.</li>
    <li><b>세로(같은 학생의 열):</b> <b>같은 학생의 같은 답</b>을 세 평가자가 어떻게 봤는지다. 답이 같으므로 품질 차이가 없고, 차이는 평가자에서 나온다. 자기 계열 평가자의 값이 다른 평가자보다 높으면 자기 계열을 우대했다는 신호다.</li>
  </ul></div>
  <p><b>해석.</b></p>
  <ul>
    <li><b>학생 품질:</b> 세 평가자의 행 모두에서 GPT 학생이 가장 자주 1위다(1위 비율 {cr[('gemini','gpt')].top1_rate:.1f}~{cr[('gpt','gpt')].top1_rate:.1f}%). 평가자가 누구든 같은 방향이므로 GPT 학생의 답이 실제로 가장 낫다고 볼 수 있다.</li>
    <li><b>Gemini 학생 열:</b> Gemini 평가자 {cr[('gemini','gemini')].rank1}번({cr[('gemini','gemini')].top1_rate:.1f}%), GPT 평가자 {cr[('gpt','gemini')].rank1}번({cr[('gpt','gemini')].top1_rate:.1f}%), Claude 평가자 {cr[('claude','gemini')].rank1}번({cr[('claude','gemini')].top1_rate:.1f}%). 같은 답인데 Gemini 평가자만 100번 이상 더 1위로 놓았다. 세로 차이가 가장 크다.</li>
    <li><b>GPT 학생 열:</b> GPT 평가자 {cr[('gpt','gpt')].rank1}번, Claude {cr[('claude','gpt')].rank1}번, Gemini {cr[('gemini','gpt')].rank1}번. 자기 계열이 가장 높지만, 여기서 Gemini 평가자의 낮은 값({cr[('gemini','gpt')].rank1})은 Gemini 학생을 올리느라 밀려난 몫일 수 있다(6.2의 함정). 세로 비교도 다른 평가자의 편향에 오염될 수 있어서, 6.4에서 모형으로 분리한다.</li>
    <li><b>Claude 학생 열:</b> Claude {cr[('claude','claude')].rank1}번, GPT {cr[('gpt','claude')].rank1}번, Gemini {cr[('gemini','claude')].rank1}번으로 차이가 작다.</li>
  </ul>
  {fig_top1}
  <p><b>그림 6-1 읽는 법.</b> 한 줄이 학생 한 명(세로 비교를 가로로 눕힌 것)이고, 점 세 개가 세 평가자가 준 1위 비율이다. <b>큰 점(자기 계열 평가자)이 그 줄의 오른쪽 끝에 있으면</b> 자기 계열을 더 자주 1위로 놓았다는 뜻이다. 세 줄 모두 큰 점이 가장 오른쪽이지만, 간격은 Gemini 학생 줄에서 가장 넓다.</p>

  <h3>6.4 결과 ② 선호 누출 추정: 단순 계산과 세 통계 모형 (주 결과, 최종 순위)</h3>
  <p><b>무엇을 보는가.</b> 6.3의 세로 비교를 숫자 하나로 만든 것이 단순 계산이고, 다른 평가자의 편향과 품질·위치를 걸러 낸 것이 세 통계 모형이다(6.2). 주 결과는 <b>② 순위 로짓 모형(PL)의 승률 차</b>다.</p>
  {t_leak}
  <p class="m">표 6-3. 단순 계산: 같은 학생에 대한 자기 계열 평가자의 1위 비율 vs 다른 두 평가자 평균(괄호 = 차이). 모형: 굵은 숫자 = 추정치(%p), 대괄호 = 95% 신뢰구간, p<sub>Holm</sub> = 평가자 3명에 대한 Holm 보정 p값(7.2). 음영·별표 = Holm p &lt; 0.05. 마지막 열 = ② 순위 로짓 모형(PL)을 seed별로 따로 추정했을 때 양수인 seed 수.</p>
  <div class="note"><b>표 6-3 읽는 법.</b>
  <ul style="margin-bottom:0">
    <li><b>한 행 = 한 평가자.</b> 왼쪽에서 오른쪽으로 읽으면 "단순 계산 → 모형"으로 같은 평가자의 선호 누출이 어떻게 바뀌는지 보인다.</li>
    <li><b>숫자 뜻:</b> ① 1위 모형의 +3.5는 "품질과 위치가 같은 세 답이라면 자기 계열 답이 1위로 뽑힐 확률이 33.3%에서 36.8%로 3.5%p 오른다"는 뜻이다. ②·③의 숫자는 "품질과 위치가 같은 상대를 이길 확률이 50%에서 몇 %p 오르나"다.</li>
    <li><b>판단:</b> 신뢰구간이 0을 포함하지 않고 Holm p &lt; 0.05이면(음영) 우연으로 보기 어렵다. 0을 포함하면 이 자료로는 효과가 0이라는 가설을 버릴 수 없다.</li>
  </ul></div>
  <p><b>해석 (평가자별).</b></p>
  <ul>
    <li><b>Gemini 평가자:</b> 단순 계산 {sgn(lt['gemini'].leak)}%p가 모형에서도 그대로 크다(① {sgn(M[('PL_top1','gemini')].pp)}, ② {sgn(M[('PL','gemini')].pp)}, ③ {sgn(M[('LPM','gemini')].pp)}%p, 세 모형 모두 음영). 품질과 위치가 같다면 Gemini 학생이 이길 확률은 50%가 아니라 약 {50 + M[('PL','gemini')].pp:.0f}%다. <b>뚜렷한 선호 누출.</b></li>
    <li><b>GPT 평가자:</b> 단순 계산 {sgn(lt['gpt'].leak)}%p가 모형에서 줄어든다(① {sgn(M[('PL_top1','gpt')].pp)}, ② {sgn(M[('PL','gpt')].pp)}, ③ {sgn(M[('LPM','gpt')].pp)}%p). 줄어든 몫은 Gemini 평가자의 편향이 기준선을 낮춘 부분이다. 보조 지표인 ①에서만 음영이고 주 모형 ②와 ③에서는 신뢰구간이 0 근처까지 내려온다. <b>약한 증거.</b></li>
    <li><b>Claude 평가자:</b> 단순 계산 {sgn(lt['claude'].leak)}%p가 모형에서 사라진다(① {sgn(M[('PL_top1','claude')].pp)}, ② {sgn(M[('PL','claude')].pp)}, ③ {sgn(M[('LPM','claude')].pp)}%p, 신뢰구간이 모두 0 포함). <b>선호 누출이 관찰되지 않는다.</b></li>
  </ul>
  {fig_forest}
  <p><b>그림 6-2 읽는 법.</b> 표 6-3의 모형 결과를 그림으로 옮긴 것이다. 한 줄이 "평가자 × 모형" 하나다. 점이 추정치, 가로선이 95% 신뢰구간, 세로 0선이 "편향 없음"이다. <b>가로선 전체가 0선 오른쪽에 있으면</b> 자기 계열 우대가 우연이 아니라는 뜻이다. 채운 점은 Holm 보정 후에도 유의한 것이다. Gemini의 세 줄은 모두 0선에서 멀리 떨어져 있고, GPT는 0선에 걸치거나 겨우 떨어져 있으며, Claude는 0선을 가로지른다.</p>
  <p class="m">식별 가정 검정(6.2 (5)): p = {overid[overid.criterion == 'overall'].p.iloc[0]:.2f}로, 세 평가자를 분리하는 데 필요한 가정이 자료와 맞는다.</p>
{SEC_LEN}

  <h3>6.5 결과 ③ 다섯 번의 독립 학습(seed)에서도 반복되는가</h3>
  <p><b>무엇을 보는가.</b> 같은 교사 데이터로 학생을 다섯 번(seed 42~46) 따로 학습했다. 6.4의 분석을 seed마다 따로 해서(seed 하나 = 질문 100 × 순서 3), 결과가 한 번의 학습 운이 아닌지 본다.</p>
  {fig_seed}
  <p><b>그림 6-3 읽는 법.</b> 한 줄이 평가자 한 명, 점 하나가 seed 하나에서 추정한 ② 순위 로짓 모형(PL)의 승률 차다. <b>다섯 점이 모두 0선 오른쪽</b>이면 학생을 다섯 번 다시 학습해도 매번 자기 계열을 우대했다는 뜻이다. Gemini는 다섯 점이 모두 +{min(seedv['gemini']):.0f}~+{max(seedv['gemini']):.0f}%p에 있다. GPT는 넷이 오른쪽, 하나가 0 근처다. Claude는 대부분 0선 왼쪽이다.</p>
  {t_seed}
  <p class="m">표 6-4. 세 모형 각각의 seed별 추정치(%p). 양수 = 다섯 seed 중 0보다 큰 개수. t검정 = 다섯 값의 평균이 0보다 큰지(한쪽), 괄호는 평가자 3명에 대한 Holm 보정 p.</p>
  <p><b>표 6-4 읽는 법과 해석.</b> 한 행을 왼쪽에서 오른쪽으로 읽으면 seed마다의 값이고, 오른쪽 두 열이 반복성 판정이다. Gemini는 세 모형 모두 5/5 양수이고 t검정이 보정 후에도 유의하다 → <b>반복된다.</b> GPT는 대부분 4/5이고 주 모형의 t검정은 보정 후 0.032로 유의하지만 seed 45에서 0이다 → 방향은 일관되지만 작다. Claude는 주 모형 1/5 → 반복되는 효과가 없다.</p>

  <h3>6.6 결과 ④ 평가 기준별 분석 (ablation)</h3>
  <p><b>무엇을 보는가.</b> 6.4는 최종 순위로 계산했다. 평가자는 네 평가 기준(사실 정확성, 질문 충족도, 논리성, 가독성)의 순위도 매겼으므로 같은 분석을 기준마다 따로 해서, 선호 누출이 <b>어떤 판단</b>에서 나타나는지 본다.</p>
  {t_abl}
  <p class="m">표 6-5. 값 = 자기 계열 효과(%p). 음영·별표 = Holm 보정 p &lt; 0.05(최종 순위는 평가자 3개, 네 기준은 4 × 3 = 12개를 한 묶음으로 보정). 식별 가정 검정 p &lt; 0.05이면 그 기준에서는 모형의 가정이 흔들린다는 뜻이다.</p>
  <div class="note"><b>표 6-5 읽는 법.</b> <b>한 평가자를 정해 위에서 아래로</b>(기준을 따라) 읽는다. 예를 들어 Gemini 행만 따라가면 사실 정확성 → 질문 충족도 → 논리성 → 가독성에서 효과가 어떻게 달라지는지 보인다. 같은 기준 안에서 세 평가자를 비교하려면 세 행을 위아래로 비교한다.</div>
  <p><b>해석.</b> Gemini 평가자는 사실 정확성·질문 충족도·가독성에서 세 모형 모두 유의하고, 가장 큰 기준은 <b>사실 정확성</b>이다(문체가 아니라 내용 판단에서 가장 컸다). 논리성에서는 가장 작고 ② 순위 로짓 모형(PL)에서만 유의하다. GPT와 Claude 평가자는 어떤 기준에서도 보정 후 유의하지 않다. 논리성·가독성은 식별 가정 검정이 경계 수준(p &lt; 0.05)이므로 그 두 기준의 추정치는 조심해서 읽는다.</p>
  {fig_crit}
  <p><b>그림 6-4 읽는 법.</b> 표 6-5의 ② 순위 로짓 모형(PL) 값을 기준별로 묶어 그린 것이다. 기준마다 세 평가자의 점과 신뢰구간이 있고, 0선 오른쪽에 떨어져 있는 것이 자기 계열 우대다. Gemini(초록)만 네 기준 모두 0선 오른쪽에 있다.</p>

  <h3>6.7 결과 ⑤ 위치 편향: 순서를 바꿔 넣은 설계가 필요했나</h3>
  <p><b>무엇을 보는가.</b> 같은 세 답을 순서만 바꿔 3번 넣었으므로, 각 학생이 A·B·C 위치에 정확히 500번씩 놓였다. 편향이 없다면 1위로 뽑힌 답의 위치는 A·B·C 각각 33.3%여야 한다.</p>
  {fig_pos}
  <p><b>그림 6-5 읽는 법과 해석.</b> 가로축이 답이 놓인 위치, 막대가 평가자, 점선이 편향이 없을 때의 33.3%다. 세 평가자 모두 A(첫 번째)에서 막대가 점선보다 훨씬 높다(GPT {posA['gpt']:.1f}%, Claude {posA['claude']:.1f}%, Gemini {posA['gemini']:.1f}%). <b>먼저 보여 준 답을 1위로 고르는 경향이 실제로 있다.</b></p>
  {t_prank}
  <p class="m">표 6-6. 같은 학생의 같은 답이 A·B·C에 놓였을 때 받은 평균 순위(1~3, 작을수록 좋음, 위치마다 500번). A→C = C에 놓였을 때 − A에 놓였을 때.</p>
  <p><b>표 6-6 읽는 법과 해석.</b> 한 행을 왼쪽에서 오른쪽으로 읽으면 <b>답은 그대로 두고 위치만 바꿨을 때</b>의 순위 변화다. 모든 행에서 A → C로 갈수록 순위가 나빠진다. 예: GPT 평가자가 본 GPT 학생은 A에서 {pr[('gpt','gpt')].A:.2f}위, C에서 {pr[('gpt','gpt')].C:.2f}위로, 같은 답이 위치만으로 약 0.5등 차이가 난다. 이 위치 효과는 학생 간 품질 차이와 비슷하거나 크다. 각 학생이 세 위치에 똑같이 놓였기 때문에 6.4의 결과에는 이 효과가 섞이지 않는다.</p>
  {t_pos}
  <p class="m">표 6-7. 결과(최종 순위·네 기준)별, 평가자별: 1위가 A에 놓여 있던 비율 · 모형의 A 위치 오즈비(C 대비) · 순서 3가지에서 1위가 모두 같았던 묶음 비율.</p>
  <p><b>표 6-7 읽는 법과 해석.</b> 한 칸에 세 숫자가 있다. 첫째(33.3%보다 클수록 첫 번째 답 선호), 둘째(1보다 클수록 A 위치 가산점이 큼), 셋째(100%에 가까울수록 순서를 바꿔도 판정이 안정적). 거의 모든 기준에서 A 선호가 나타났고, 순서 3가지에서 1위가 같았던 묶음은 약 절반~3분의 2뿐이다. 학생 답이 모두 품질이 낮고 비슷해 위치가 판정을 쉽게 바꾸는 조건이다. <b>순서를 하나로만 정해 평가했다면 위치 효과가 선호 누출보다 결과를 더 크게 좌우했을 것이다.</b></p>
"""

SEC62 = f"""  <h3>6.2 선호 편향을 재는 방법</h3>
  <p>평가자 한 명이 매긴 순위 하나를 예로 든다. GPT 평가자가 어떤 질문에서 <b>GPT 학생 1위, Gemini 학생 2위, Claude 학생 3위</b>로 매겼다. 이런 순위 1,500개(평가자·학생마다)를 먼저 두 가지 지표로 요약한다.</p>
  <div class="cols"><div>
  <h3>(1) 1위 비율: "나머지 두 답보다 위에 놓았나?"</h3>
  <p>학생이 1위, 즉 나머지 두 학생보다 모두 위에 놓인 비율이다. 위 예에서 GPT 학생은 1번 성공이다.</p>
<pre>1위 비율(평가자 j, 학생 t)
  = 평가자 j가 학생 t를 1위로 놓은 횟수 ÷ 전체 횟수(1,500)

기준선: 세 학생의 품질이 같고 편향이 없으면 1/3 = 33.3%
예) GPT 평가자 → GPT 학생: {gg.rank1} ÷ 1,500 = {gg.top1_rate:.1f}%</pre>
  <p>"GPT 평가자가 GPT 학생을 나머지 둘보다 높게 매긴 경우가 많은가?"에 그대로 답하는 가장 직관적인 지표다. 다만 2위와 3위의 차이는 쓰지 않는다.</p>

  <h3>(2) 쌍대 승률: "상대 한 명과 붙었을 때 이겼나?"</h3>
  <p>순위 하나를 두 학생씩 짝지은 대결 3개로 푼다. 위 예에서는 GPT &gt; Gemini, GPT &gt; Claude, Gemini &gt; Claude. 학생 한 명은 순위 하나에서 대결 2개에 참여한다.</p>
<pre>1위 → 2승 0패,  2위 → 1승 1패,  3위 → 0승 2패

쌍대 승률(평가자 j, 학생 t)
  = 이긴 대결 수 ÷ 참여한 대결 수
  = (1위 횟수 × 2 + 2위 횟수) ÷ (전체 횟수 × 2)

기준선: 50%
예) GPT 평가자 → GPT 학생: ({gg.rank1} × 2 + {gg.rank2}) ÷ 3,000 = {gg.win_rate:.1f}%</pre>
  <p>1·2·3위 정보를 모두 쓰므로 1위 비율보다 정밀하다.</p>
  <div class="note"><b>주 지표는 쌍대 승률, 1위 비율은 보조.</b> 선호 누출의 최종 판단은 쌍대 승률(② 순위 로짓 모형(PL))로 한다. 순위 정보를 모두 써서 더 정밀하고(신뢰구간이 더 좁음), 분석 계획 단계에서 미리 정한 주 모형이기 때문이다. 1위 비율(① 1위 모형)은 "1등을 몰아주는가"라는 직관적인 질문으로도 같은 결론이 나오는지 확인하는 보조 지표다.</div>
  {table(["", "1위 비율", "쌍대 승률"], ["<td>묻는 것</td><td>나머지 둘보다 모두 위인가</td><td>상대 한 명보다 위인가</td>",
        "<td>쓰는 정보</td><td>1위인지만</td><td>1·2·3위 전부</td>", "<td>기준선</td><td>33.3%</td><td>50%</td>",
        "<td>장점</td><td>직관적</td><td>정밀함</td>", "<td>역할</td><td>보조 지표</td><td><b>주 지표</b></td>"], num_from=9)}
  </div><div>
  <h3>(3) 두 지표로 알고 싶은 것: 선호 누출</h3>
  <p>(1)·(2)는 "평가자가 그 학생을 얼마나 높게 매겼나"를 요약한 값이다. 이 연구가 알고 싶은 것은 그중에서 <b>학생의 실제 품질 때문이 아니라 평가자와 같은 계열이라서 더 받은 몫</b>이다.</p>
<pre>본 연구가 알고 싶은 값 (선호 누출, 평가자 j)
  = 평가자 j가 자기 계열 학생에게 준 1위 비율(또는 쌍대 승률)
  − 그 학생의 실제 품질만으로 받았어야 할 값

0이면 공정, 양수면 자기 계열을 더 높게 평가</pre>
  <p>문제는 "실제 품질만으로 받았어야 할 값"을 모른다는 것이다. 사람 평가가 없으므로 정답 순위가 없다. GPT 평가자가 GPT 학생을 1위로 자주 놓았다고 바로 편향이라 할 수도 없다. GPT 학생의 답이 실제로 좋아서일 수 있기 때문이다.</p>
  <p>가장 쉬운 방법은 <b>같은 학생의 답을 다른 평가자가 어떻게 봤는지</b>를 기준선으로 쓰는 것이다. 답이 같으므로 품질 차이는 지워지고 평가자의 차이만 남는다고 기대할 수 있다.</p>
<pre>단순 누출(GPT 평가자)
  = [GPT 평가자가 매긴 GPT 학생의 1위 비율]
  − [Claude·Gemini 평가자가 매긴 GPT 학생의 1위 비율 평균]

편향이 없으면 0, 양수면 자기 계열 학생을 더 높게 본 것

예) {gg.top1_rate:.1f}% − ({cg.top1_rate:.1f}% + {mg.top1_rate:.1f}%) ÷ 2 = {sgn(lk.leak)}%p
    (쌍대 승률로 재면 {gg.win_rate:.1f}% − {lw.others:.1f}% = {sgn(lw.leak)}%p)</pre>
  </div></div>

  <h3>(4) 단순 계산의 함정: 기준선인 "다른 평가자"도 편향될 수 있다</h3>
  <p>단순 계산은 "다른 평가자는 공정하다"고 가정한다. 그런데 다른 평가자도 각자 자기 계열 학생을 우대할 수 있다. 게다가 순위는 세 학생이 1·2·3위를 나눠 갖는 구조라, 한 학생을 올리면 나머지 학생은 <b>자동으로</b> 내려간다.</p>
  <div class="cols"><div>
  <p><b>예시: 편향이 Gemini 평가자에게만 있는 경우.</b> 세 학생의 실제 품질은 모두 같다고 하자. GPT·Claude 평가자는 완전히 공정하고, Gemini 평가자만 Gemini 학생을 우대한다(쌍대 승률, %).</p>
  {table(["평가자", "GPT 학생", "Claude 학생", "Gemini 학생"], [
      "<td>GPT 평가자 (공정)</td><td class='num'>50</td><td class='num'>50</td><td class='num'>50</td>",
      "<td>Claude 평가자 (공정)</td><td class='num'>50</td><td class='num'>50</td><td class='num'>50</td>",
      "<td>Gemini 평가자 (편향)</td><td class='num'><b>40</b></td><td class='num'><b>40</b></td><td class='num'>70</td>"])}
  <p>Gemini 평가자가 Gemini 학생을 70%로 올리자, 승률의 평균 50%를 맞추느라 GPT·Claude 학생이 <b>40%로 내려갔다</b>. 이 40은 두 학생의 품질 탓이 아니라 Gemini 평가자의 편향 탓이다. 이제 단순 계산을 해 보면</p>
<pre>GPT 누출    = 50 − (Claude 평가자 50 + Gemini 평가자 40) ÷ 2 = 50 − 45 = +5%p   ✗ 실제는 0
Claude 누출 = 50 − (GPT 평가자 50 + Gemini 평가자 40) ÷ 2    = 50 − 45 = +5%p   ✗ 실제는 0
Gemini 누출 = 70 − (GPT 평가자 50 + Claude 평가자 50) ÷ 2    = 70 − 50 = +20%p  ✓</pre>
  <p><b>공정한 GPT·Claude 평가자에게도 +5%p의 가짜 누출이 생긴다.</b> Gemini 평가자의 편향이 "다른 평가자 평균"이라는 기준선을 통해 다른 평가자에게 번진 것이다.</p>
  </div><div>
  <p><b>해결의 실마리: 쌍마다 "중립 심판"이 있다.</b> 학생 둘을 비교하면, 둘 중 어느 쪽과도 계열이 다른 평가자가 항상 한 명 있다. 그 평가자의 판정을 기준선으로 쓰면 기준선이 오염되지 않는다. 같은 예시를 쌍(1:1 대결)으로 풀면</p>
  {table(["학생 쌍", "중립 심판", "중립 심판의 판정", "나머지 두 평가자의 판정", "결론"], [
      "<td>GPT vs Claude</td><td>Gemini 평가자</td><td class='num'>50 : 50</td><td>GPT 평가자 50:50, Claude 평가자 50:50</td><td>GPT·Claude 평가자 편향 없음</td>",
      "<td>GPT vs Gemini</td><td>Claude 평가자</td><td class='num'>50 : 50</td><td>GPT 평가자 50:50, <b>Gemini 평가자 30:70</b></td><td>Gemini 평가자만 +20%p</td>",
      "<td>Claude vs Gemini</td><td>GPT 평가자</td><td class='num'>50 : 50</td><td>Claude 평가자 50:50, <b>Gemini 평가자 30:70</b></td><td>Gemini 평가자만 +20%p</td>"], num_from=9)}
<pre>중립 심판 기준 → GPT 0,  Claude 0,  Gemini +20%p   (정답)
단순 계산      → GPT +5, Claude +5, Gemini +20%p</pre>
  <p><b>실제 자료에서도 같은 일이 일어났다.</b> Gemini 평가자의 자기 우대가 실제로 컸기 때문에, 쌍대 승률로 단순 계산하면 GPT +{simple_vs[0][0]:.1f}, Claude +{simple_vs[1][0]:.1f}%p였지만, 중립 심판을 기준으로 한 통계 모형에서는 GPT {sgn(simple_vs[0][1])}, Claude {sgn(simple_vs[1][1])}%p였다(7.1).</p>
  </div></div>

  <h3>(5) 그래서 본 연구가 쓴 방법: 세 평가자의 편향을 동시에 추정하는 통계 모형</h3>
  <p>(3)의 값을 제대로 구하려면 세 가지가 필요하다. <b>① 세 평가자의 편향을 한꺼번에</b> 추정해 (4)처럼 서로의 기준선에 섞이지 않게 하고, <b>② 학생 품질과 답의 위치를 통제</b>하고, <b>③ 그 결과가 우연인지 판단할 불확실성</b>(신뢰구간, p값)을 계산해야 한다. 이를 위해 (4)의 "중립 심판" 아이디어를 순위 자료의 표준 모형에 넣었다.</p>
{SEC_PL_EASY}
  <h3>모형을 식으로 쓰면</h3>
  <div class="cols"><div>
  <p><b>주 모형: 순위 로짓 모형 (Plackett–Luce).</b> 평가자 <i>j</i>가 평가 항목 <i>i</i>(질문 × seed × 제시 순서)에서 학생 <i>t</i>의 답에 부여하는 효용(잠재 점수)을</p>
  <div class="eq"><i>u<sub>jit</sub></i> = <i>θ</i><sub><i>s</i>(<i>i</i>),<i>t</i></sub> + <i>δ</i><sub><i>j</i>,<i>a</i>(<i>i</i>,<i>t</i>)</sub> + <i>λ<sub>j</sub></i> · 𝟙[<i>t</i> = <i>j</i>]</div>
  <p>로 두고, 평가자가 매긴 순위 <i>π</i> = (<i>π</i><sub>1</sub> ≻ <i>π</i><sub>2</sub> ≻ <i>π</i><sub>3</sub>)의 확률을</p>
  <div class="eq">P(<i>π</i> | <i>j</i>, <i>i</i>) = <span class="frac"><span>exp(<i>u<sub>jiπ<sub>1</sub></sub></i>)</span><span>Σ<sub><i>k</i>∈{{<i>π</i><sub>1</sub>,<i>π</i><sub>2</sub>,<i>π</i><sub>3</sub>}}</sub> exp(<i>u<sub>jik</sub></i>)</span></span> × <span class="frac"><span>exp(<i>u<sub>jiπ<sub>2</sub></sub></i>)</span><span>exp(<i>u<sub>jiπ<sub>2</sub></sub></i>) + exp(<i>u<sub>jiπ<sub>3</sub></sub></i>)</span></span></div>
  <p>로 모형화한다. 순위를 "세 답 중 1위를 고르고, 남은 둘 중 2위를 고르는" 두 번의 선택으로 보는 것이다. 순위 4,500개의 로그우도 Σ log P를 최대로 하는 값으로 모수를 추정한다(최대우도법).</p>
  {table(["기호", "뜻"], [
      "<td><i>j</i></td><td>평가자 ∈ {GPT, Claude, Gemini}</td>",
      "<td><i>i</i></td><td>평가 항목 = 질문 × seed × 제시 순서 (평가자마다 1,500개)</td>",
      "<td><i>t</i></td><td>학생의 교사 계열 ∈ {GPT, Claude, Gemini}. 한 항목에는 같은 seed의 세 학생이 들어 있다</td>",
      "<td><i>s</i>(<i>i</i>)</td><td>항목 <i>i</i>의 seed (42~46)</td>",
      "<td><i>a</i>(<i>i</i>,<i>t</i>)</td><td>항목 <i>i</i>에서 학생 <i>t</i>의 답이 놓인 위치 ∈ {A, B, C}</td>",
      "<td><i>θ<sub>s,t</sub></i></td><td><b>학생 품질</b>: seed <i>s</i>의 <i>t</i> 학생이 세 평가자에게 공통으로 받는 강함 (15명; seed마다 GPT 학생 = 0 기준)</td>",
      "<td><i>δ<sub>j,a</sub></i></td><td><b>위치 효과</b>: 평가자 <i>j</i>가 위치 <i>a</i>의 답에 주는 가산점 (C = 0 기준, 평가자마다 따로)</td>",
      "<td><i>λ<sub>j</sub></i></td><td><b>선호 누출</b>: 평가자 <i>j</i>가 자기 계열 학생에게 주는 추가 점수(로그 오즈). 관심 모수. 0이면 공정</td>",
      "<td>𝟙[<i>t</i> = <i>j</i>]</td><td>학생이 평가자와 같은 계열이면 1, 아니면 0</td>",
      "<td><i>π</i><sub>1</sub>, <i>π</i><sub>2</sub>, <i>π</i><sub>3</sub></td><td>평가자가 1·2·3위로 매긴 학생</td>"], num_from=9)}
  <div class="note"><b>왜 λ<sub><i>j</i></sub> · 𝟙[<i>t</i> = <i>j</i>] 항을 넣었나: 연구 가설을 모형 안에 옮긴 것</b>
  <p>이 연구의 가설(H1)은 <b>"평가자는 품질이 같다면 자기와 같은 계열의 데이터로 학습한 학생을 더 높게 평가한다"</b>이다. 이 가설을 모형 안에 그대로 옮긴 것이 λ<sub><i>j</i></sub> · 𝟙[<i>t</i> = <i>j</i>] 항이다.</p>
  <ul>
    <li><b>𝟙[<i>t</i> = <i>j</i>] (같은 계열 지표):</b> 학생이 평가자와 같은 계열일 때만 1이 된다. 그래서 가산점 λ<sub><i>j</i></sub>는 <b>자기 계열 학생에게만</b> 붙는다. 예: GPT 평가자가 GPT 학생을 볼 때만 λ<sub>GPT</sub>가 더해지고, Claude·Gemini 학생을 볼 때는 더해지지 않는다.</li>
    <li><b>평가자마다 따로 λ<sub><i>j</i></sub>:</b> 세 평가자의 편향 크기가 다를 수 있으므로 λ<sub>GPT</sub>, λ<sub>Claude</sub>, λ<sub>Gemini</sub>를 각각 추정한다. 하나의 공통 λ로 두면 "어느 평가자에게 편향이 있는가"를 답할 수 없다.</li>
    <li><b>품질·위치와 분리:</b> 같은 식 안에 학생 품질 θ와 위치 효과 δ가 함께 있으므로, λ는 "품질과 위치를 같게 맞췄을 때 같은 계열이라서 더 받는 점수"만 나타낸다.</li>
    <li><b>이 항이 없으면:</b> 모형은 "평가자 계열과 무관하게 품질과 위치만으로 순위가 정해진다"(누출 없음)는 모형이 된다. 즉 λ<sub><i>j</i></sub> = 0이 귀무가설에 해당한다.</li>
  </ul>
  <p><b>이 항으로 검정하는 가설 (평가자 <i>j</i>마다):</b></p>
<pre>귀무가설 H0: λ_j = 0   평가자 j는 품질·위치가 같으면 자기 계열 학생을 다른 학생과 똑같이 평가한다 (선호 누출 없음)
대립가설 H1: λ_j ≠ 0   같은 조건에서 자기 계열 학생을 다르게 평가한다
                        (연구 가설의 예상 방향은 λ_j &gt; 0: 더 높게 평가)</pre>
  <p style="margin-bottom:0">양측 검정으로 둔 이유는 자기 계열을 오히려 낮게 보는 경우(λ<sub><i>j</i></sub> &lt; 0)도 확인하기 위해서다. 검정 방법(부트스트랩 p값, Holm 보정)과 판정은 7.2에 있다. 또 "자기 계열에만 가산점이 붙고, 자기 계열이 아닌 두 학생은 품질 외에는 똑같이 대한다"는 이 항의 형태 자체가 가정이므로, 모든 평가자 × 학생 조합에 별도 효과를 둔 일반 모형과 우도비 검정으로 비교해 확인했다(최종 순위 p = {overid[overid.criterion == 'overall'].p.iloc[0]:.2f}, 가정이 자료와 맞음).</p></div>
  <p><b>λ를 확률로 읽기.</b> 이 모형에서 학생 <i>t</i>가 <i>t′</i>보다 위에 놓일 확률은 정확히 σ(<i>u<sub>t</sub></i> − <i>u<sub>t′</sub></i>) = 1 / (1 + e<sup>−(<i>u<sub>t</sub></i> − <i>u<sub>t′</sub></i>)</sup>)이다. 따라서 품질과 위치가 같은 두 답이라면 자기 계열 답이 이길 확률은 σ(<i>λ<sub>j</sub></i>)이고, 50%에서 σ(<i>λ<sub>j</sub></i>) − 50%만큼 오른다(이 문서의 "승률 %p"). e<sup><i>λ<sub>j</sub></i></sup>는 오즈비다.</p>
  </div><div>
  <p><b>세 평가자의 λ가 분리되는 이유.</b> 학생 <i>t</i>와 <i>t′</i>의 효용 차이를 평가자별로 쓰면(위치 효과 생략)</p>
  <div class="eq" style="text-align:left">GPT 평가자: <i>θ</i><sub>GPT</sub> − <i>θ</i><sub>Claude</sub> + <i>λ</i><sub>GPT</sub><br>Claude 평가자: <i>θ</i><sub>GPT</sub> − <i>θ</i><sub>Claude</sub> − <i>λ</i><sub>Claude</sub><br>Gemini 평가자: <i>θ</i><sub>GPT</sub> − <i>θ</i><sub>Claude</sub> &nbsp;(중립 심판)</div>
  <p>중립 심판의 판정이 품질 차이(<i>θ</i><sub>GPT</sub> − <i>θ</i><sub>Claude</sub>)를 알려 주고, 나머지 두 평가자의 판정에서 그것을 빼면 <i>λ</i>가 남는다. 쌍 3개 × 평가자 3명의 정보로 품질 차 2개와 <i>λ</i> 3개를 함께 추정할 수 있다. (4)의 예시에 적용하면 <i>λ</i><sub>GPT</sub> = 0, <i>λ</i><sub>Claude</sub> = 0, <i>λ</i><sub>Gemini</sub> &gt; 0이 나온다.</p>
  <div class="note"><b>"학생 품질" θ는 무엇인가?</b> 평가자가 따로 매긴 점수나 사람이 정한 정답 품질이 아니다. 모형 안에서 λ, δ와 함께 추정되는 <b>숨은 값</b>으로, 자기 계열 가산점(λ)과 위치 효과(δ)를 빼고 나서 <b>세 평가자의 순위에 공통으로 남는 학생의 강함</b>이다. 즉 "세 평가자가 공통으로 인정한 품질"이다. 학생 15명(교사 × seed)마다 하나씩 있고, 같은 seed의 세 학생끼리만 비교되므로 seed마다 GPT 학생을 0으로 두고 나머지를 그 대비 값으로 추정한다. 질문마다 달라지는 품질은 θ에 들어가지 않아, 이 부분은 아래 ③ 쌍 고정효과 모형(LPM)이 따로 흡수한다.</div>
  </div></div>

  <h3>세 가지 모형: 무엇이고, 원래 있는 모형인가</h3>
  {table(["모형", "어떤 모형인가", "출처 (표준 모형)", "이 연구에서의 역할"], [
      "<td><b>② 순위 로짓<br>(Plackett–Luce, 주 모형)</b></td><td>순위를 '1위 선택 → 남은 것 중 2위 선택'의 연속된 선택으로 보고, 각 선택 확률을 softmax(로짓)로 두는 순위 자료 모형. 두 개만 비교하면 Bradley–Terry 모형(쌍대 비교 모형)과 같다. 위 식 그대로다.</td><td>Luce(1959)의 선택 공리, Plackett(1975). 계량경제학에서는 rank-ordered(exploded) logit으로 쓰인다(Beggs et al., 1981). 조건부 로짓으로 추정한다.</td><td>1·2·3위 정보를 모두 써서 λ를 추정. 결과를 쌍대 승률 %p로 읽는다.</td>",
      "<td><b>① 1위 모형<br>(조건부 로짓, 보조)</b></td><td>같은 효용 식에서 '세 답 중 1위 선택'만 쓰는 다항(조건부) 로짓. 위 확률식의 첫 번째 분수만 쓴다.</td><td>McFadden의 조건부 로짓(이산 선택 모형의 표준).</td><td>1위 비율에 대응. λ를 e<sup>λ</sup>/(e<sup>λ</sup>+2) − 33.3%로 읽는다.</td>",
      "<td><b>③ 쌍 고정효과 선형확률모형<br>(LPM, 강건성)</b></td><td>순위를 1:1 대결로 풀어 y = 1(앞 학생이 이김)을 종속변수로 한 선형 회귀. 같은 묶음의 같은 두 답(쌍)마다 고유 절편 α<sub>쌍</sub>(고정효과)을 둔다:<br><i>y<sub>jrp</sub></i> = <i>α<sub>p</sub></i> + Σ<sub><i>k</i></sub> <i>λ<sub>k</sub></i>(𝟙[<i>j</i>=<i>k</i>, 앞 학생=<i>k</i>] − 𝟙[<i>j</i>=<i>k</i>, 뒤 학생=<i>k</i>]) + 위치 항 + <i>ε</i></td><td>선형확률모형과 고정효과(within) 추정은 패널 자료 분석의 표준 방법. 표준오차는 질문 단위 군집 강건 표준오차(Cameron &amp; Miller, 2015).</td><td>α<sub>쌍</sub>가 그 질문에서 두 답의 품질 차이(내용, 길이, 난이도)를 통째로 흡수 → 가장 엄격한 확인. λ가 바로 %p로 읽힌다.</td>"], num_from=9)}
  <p class="m">세 모형 모두 통계학·계량경제학에서 널리 쓰이는 기존 모형이다. 이 연구가 새로 설계한 것은 모형 자체가 아니라, 평가자 × 같은 계열 지표 𝟙[<i>t</i> = <i>j</i>]를 넣고 "쌍마다 중립 심판이 있다"는 구조를 이용해 세 평가자의 선호 누출 λ를 동시에 식별한 점이다.</p>
  <ul>
    <li><b>통제:</b> 학생 품질(15명)과 평가자별 위치 효과를 모형에 넣는다. 각 학생이 각 위치에 같은 횟수로 놓였으므로 위치 효과가 λ에 섞이지 않는다.</li>
    <li><b>표준오차:</b> 질문 100개를 복원 추출하는 질문 단위 부트스트랩 1,000회(같은 질문이 다섯 seed에 다시 나오므로 질문을 단위로 묶는다).</li>
    <li><b>가설 검정:</b> 평가자마다 "자기 계열 효과 λ = 0"(귀무가설)을 검정하고, 평가자 3명을 함께 검정하므로 Holm 보정 p로 판단한다. 귀무가설·대립가설, p값의 뜻, Holm 보정의 절차는 7.2에 자세히 적었다.</li>
    <li><b>가정 확인:</b> "평가자는 자기 계열이 아닌 두 학생을 품질 외에는 똑같이 대한다"는 가정을 우도비 검정으로 확인했다(최종 순위 p = {overid[overid.criterion == 'overall'].p.iloc[0]:.2f}, 기각되지 않음). 정답을 아는 가짜 자료에서 모형이 누출이 없으면 0을, 있으면 같은 방향(크기는 다소 작게, 보수적)을 되찾는 것도 확인했다.</li>
  </ul>
"""


# ------------------------------------------------------------------ 7.1 단순 계산 → 모형, 7.2 가설 검정
def simple(metric, j):
    return leak[(leak.criterion == "overall") & (leak.metric == metric) & (leak.judge == j)].iloc[0]
cmp_rows = []
why = {"gemini": "자기 우대가 커서 다른 평가자 기준선의 왜곡보다 훨씬 크다. 두 방법 모두 같은 결론",
       "gpt": "단순 계산의 기준선(Claude·Gemini 평가자 평균)이 Gemini 평가자의 자기 우대 때문에 낮아져 있었다. 이를 빼면 효과가 작아진다",
       "claude": "Gemini 평가자가 Gemini 학생을 올리며 Claude 학생을 끌어내린 몫이 단순 계산에 섞여 있었다. 이를 빼면 남는 효과가 없다"}
for j in J:
    sw, st, r = simple("win_rate", j), simple("top1_rate", j), R("overall", "PL", j)
    cmp_rows.append(f"<td>{N[j]} 평가자</td><td class='num'>{sw.own:.1f}% − {sw.others:.1f}% = <b>{sgn(sw.leak)}</b></td>"
                    f"<td class='num{' sig' if r.p_holm < 0.05 else ''}'><b>{sgn(r.pp)}</b>{star(r.p_holm)}<br><span class='m'>[{sgn(r.pp_ci_low)}, {sgn(r.pp_ci_high)}]</span></td>"
                    f"<td class='num'>{sgn(r.pp - sw.leak)}</td><td>{why[j]}</td>")
t_cmp = table(["평가자", "단순 계산 (쌍대 승률)<br>자기 평가자 − 다른 평가자 평균", "② 순위 로짓 모형(PL)<br>승률 차 [95% CI]", "모형 − 단순", "왜 달라졌나"], cmp_rows)

pl = res[(res.criterion == "overall") & (res.model == "PL")].set_index("judge")
order = pl.p.sort_values().index.tolist()
m = len(order)
steps, running = [], 0
for k, j in enumerate(order, 1):
    adj = min(1, (m - k + 1) * pl.loc[j, "p"]); running = max(running, adj)
    steps.append(f"<td>{k}</td><td>{N[j]} 평가자</td><td class='num'>{pb(pl.loc[j, 'p'])}</td><td class='num'>× {m - k + 1}</td><td class='num'>{pb(running, m - k + 1)}</td>"
                 f"<td class='num'>0.05 / {m - k + 1} = {0.05 / (m - k + 1):.4f}</td><td>{'기각 (유의)' if running < 0.05 else '기각 못함'}</td>")
t_holm = table(["순서", "평가자", "보정 전 p", "곱하는 수", "Holm p", "같은 뜻: 보정 전 p와 비교할 기준", "판정"], steps, num_from=2)

def verdict(j):
    r, r1, r3 = R("overall", "PL", j), R("overall", "PL_top1", j), R("overall", "LPM", j)
    ci = f"[{sgn(r.pp_ci_low)}, {sgn(r.pp_ci_high)}]%p"
    if r.p_holm < 0.05:
        return ("<b>귀무가설 기각</b>",
                f"추정치 {sgn(r.pp)}%p, 95% 신뢰구간 {ci} 전체가 0보다 크다. 부트스트랩 1,000번 재추정에서 0 이하로 내려간 경우가 한 번도 없었고(보정 전 p {pb(r.p)}), "
                f"평가자 3명을 함께 검정한 것을 감안해도(Holm p {pb(r.p_holm, 3)}) 우연으로 보기 어렵다. 보조 지표(1위 모형 {sgn(r1.pp)}%p, Holm p {pb(r1.p_holm, 3)})와 "
                f"쌍 고정효과 모형({sgn(r3.pp)}%p, Holm p {pf(r3.p_holm)})에서도 같은 결론이다.")
    if r.p < 0.05:
        return ("<b>기각 못함</b> (보정 전에는 기각, 약한 증거)",
                f"추정치 {sgn(r.pp)}%p, 95% 신뢰구간 {ci}. 이 검정 하나만 보면 p {pf(r.p)} &lt; 0.05라 기각되지만, 평가자 3명을 함께 검정했으므로 Holm 보정을 하면 p {pf(r.p_holm)} &gt; 0.05가 되어 기각하지 못한다. "
                f"신뢰구간 하한({sgn(r.pp_ci_low)}%p)도 0에 매우 가깝다. 같은 자료를 다른 방식으로 본 쌍 고정효과 모형에서도 {sgn(r3.pp)}%p, p {pf(r3.p)}로 유의하지 않다. "
                f"다만 보조 지표인 1위 모형에서는 {sgn(r1.pp)}%p, Holm p {pf(r1.p_holm)}로 유의하다. 방향은 일관되게 양수지만 크기가 작아, 5% 기준을 안정적으로 넘지 못한다 → <b>약한 증거</b>로 해석한다.")
    return ("<b>기각 못함</b>",
            f"추정치 {sgn(r.pp)}%p, 95% 신뢰구간 {ci}이 0을 포함한다(보정 전 p {pf(r.p)}, Holm p {pf(r.p_holm)}). 즉 자료가 \"자기 계열 효과 0\"과 모순되지 않는다. "
            f"이것이 효과가 0임을 증명하는 것은 아니다. 신뢰구간의 상한이 {sgn(r.pp_ci_high)}%p이므로, 효과가 있더라도 이보다 작은 수준이라는 뜻이다. "
            f"보조 지표({sgn(r1.pp)}%p, Holm p {pf(r1.p_holm)})와 쌍 고정효과 모형({sgn(r3.pp)}%p, p {pf(r3.p)})도 같다.")
t_verdict = table(["평가자", "판정 (유의수준 5%, Holm 보정)", "이유"], [f"<td>{N[j]} 평가자</td><td>{verdict(j)[0]}</td><td>{verdict(j)[1]}</td>" for j in J], num_from=9)
fwer = (1 - 0.95 ** 3) * 100

SEC7A = f"""
  <h3>7.1 단순 계산에서 통계 모형으로: 무엇이 달라졌고 왜 달라졌나</h3>
  <div class="flow">
    <div><b>1단계: 단순 계산</b>같은 학생에 대해 [자기 계열 평가자의 값] − [다른 두 평가자 평균]. 원자료를 그대로 보여 준다.</div>
    <div><b>2단계: 한계</b>다른 평가자도 자기 계열을 우대해 기준선이 오염된다. 품질·위치 통제가 없고, 우연인지 판단할 불확실성(p값, 신뢰구간)이 없다.</div>
    <div><b>3단계: 통계 모형</b>쌍마다 중립 심판을 이용해 세 평가자의 효과를 동시에 추정. 학생 품질·위치 통제, 부트스트랩 신뢰구간, Holm 보정.</div>
    <div><b>4단계: 결과</b>Gemini는 그대로 크고 유의. GPT는 줄어 약한 증거. Claude는 사라짐.</div>
  </div>
  <div class="cols"><div>
  {t_cmp}
  <p class="m">주 지표(쌍대 승률) 기준, %p. 별표·음영 = Holm p &lt; 0.05. 1위 비율로 보아도 같은 방향이다(단순 GPT +5.3 → 모형 +3.5, Claude +3.0 → +0.6, Gemini +7.9 → +8.6).</p>
  <p><b>읽는 법.</b> 단순 계산은 "다른 평가자 평균"을 기준선으로 쓰는데, Gemini 평가자는 자기 계열 학생을 크게 올려 주므로 Gemini 평가자 아래에서 GPT·Claude 학생은 낮게 평가된다. 그 낮아진 값이 GPT·Claude 학생의 기준선에 들어가 GPT·Claude 평가자의 누출이 부풀려진다. 모형은 이 몫을 Gemini의 λ로 따로 떼어 내므로, 남는 GPT·Claude의 효과가 작아진다.</p>
  </div><div>
  {fig_simple}
  </div></div>

  <h3>7.2 가설 검정: 무엇을 검정했고 왜 "유의하다/유의하지 않다"고 판단했나</h3>
  <div class="cols"><div>
  <p><b>(1) 가설.</b> 평가자 j(GPT, Claude, Gemini) 각각에 대해 주 모형(② 순위 로짓 PL)의 자기 계열 효과 λ<sub>j</sub>를 검정한다.</p>
<pre>귀무가설 H0: λ_j = 0
  "평가자 j는 품질과 위치가 같다면 자기 계열 학생과 다른 학생을
   같은 확률(50%)로 이긴다고 판정한다" → 선호 누출 없음

대립가설 H1: λ_j ≠ 0  (양측 검정)
  "같은 조건에서 자기 계열 학생을 이긴다고 판정할 확률이 50%와 다르다"

유의수준 α = 0.05</pre>
  <p>양측 검정을 쓴 이유: 평가자가 자기 계열을 오히려 <b>낮게</b> 볼 가능성(음수)도 열어 두기 위해서다. Claude의 추정치가 실제로 음수였다.</p>
  <p><b>(2) p값과 신뢰구간을 구한 방법 (질문 단위 부트스트랩).</b> 질문 100개를 복원 추출해 새 자료를 만들고(뽑힌 질문의 seed 5 × 순서 3 × 평가자 3 자료를 통째로) 모형을 다시 적합하기를 1,000번 반복했다. 1,000개의 λ 추정치 분포에서</p>
  <ul>
    <li><b>95% 신뢰구간</b> = 하위 2.5% ~ 상위 2.5% 지점. "같은 방식으로 실험을 반복하면 효과의 참값이 이 범위에 있을 것"으로 해석한다. <b>0을 포함하지 않으면</b> 5% 수준에서 H0를 기각하는 것과 같은 뜻이다.</li>
    <li><b>p값</b> = 추정치가 0의 반대편으로 넘어간 비율 × 2. "선호 누출이 없는데도(H0) 이만큼 큰 효과가 우연히 나올 확률"에 해당한다. 1,000번 중 한 번도 넘어가지 않으면 p &lt; 0.001로 적는다.</li>
    <li>질문 단위로 묶은 이유: 같은 질문이 다섯 seed와 세 순서에 반복되어 서로 독립이 아니다. 개별 순위를 독립으로 취급하면 표준오차가 작게 계산되어 p값이 지나치게 작아진다.</li>
  </ul>
  </div><div>
  <p><b>(3) Holm 보정이란, 왜 썼나.</b> 검정을 한 번 하면 H0가 참인데도 우연히 기각할 확률(1종 오류)이 5%다. 그런데 평가자 3명을 각각 5%로 검정하면, 셋 중 <b>적어도 하나</b>가 우연히 유의하게 나올 확률(가족별 오류율, FWER)이 최대 1 − 0.95<sup>3</sup> = {fwer:.1f}%까지 커진다. "평가자 중 누군가는 누출이 있다"는 결론을 5% 오류 수준에서 말하려면 이 확률을 다시 5% 이하로 묶어야 한다.</p>
  <p>Holm(1979) 방법은 p값을 작은 순서로 정렬해, 가장 작은 것은 0.05/3, 다음은 0.05/2, 마지막은 0.05/1과 비교하고, 한 번 기각하지 못하면 그 뒤는 모두 기각하지 않는다. 같은 일을 "p값에 남은 검정 수를 곱한 값(Holm p)을 0.05와 비교"하는 것으로 표현할 수 있다. 모든 p값에 3을 곱하는 Bonferroni 보정보다 덜 보수적이면서 FWER를 5% 이하로 지킨다.</p>
  {t_holm}
  <p class="m">② 순위 로짓 모형(PL), 최종 순위. Gemini는 부트스트랩 1,000번 중 0이 한 번도 넘지 않아 p가 계산 하한(1/1,000) 이하이므로 "≤"로 적고, 보정에는 상한 0.001을 넣었다. Holm p는 앞 단계 값보다 작아지지 않도록 누적 최댓값을 쓴다. 기준별 분석(6.6)은 기준 4 × 평가자 3 = 12개를 한 묶음으로 같은 방식으로 보정했다.</p>
  </div></div>
  <p><b>(4) 평가자별 판정과 그 이유</b></p>
  {t_verdict}
  <div class="cols"><div>
  <p><b>(5) seed 반복 검정.</b> seed마다 따로 추정한 효과 5개로 두 가지를 검정했다(한쪽 검정, "누출이 양수인가"를 묻기 때문).</p>
  <ul>
    <li><b>부호 검정:</b> H0 "각 seed의 효과가 양수일 확률 = 1/2". 5개 모두 양수면 p = 0.5<sup>5</sup> = 0.031. 다만 seed가 5개뿐이라 이 값이 가능한 최소 p이고, 평가자 3명에 Holm 보정을 하면 0.094가 되어 5%를 넘을 수 없다. 그래서 t검정을 함께 본다.</li>
    <li><b>일표본 t검정:</b> H0 "5개 seed 효과의 평균 ≤ 0", H1 "평균 &gt; 0". 크기의 일관성까지 반영한다. Gemini: Holm p = {tests[(tests.criterion == 'overall') & (tests.model == 'PL') & (tests.judge == 'gemini')].t_p_holm.iloc[0]:.3f}로 기각, GPT: {tests[(tests.criterion == 'overall') & (tests.model == 'PL') & (tests.judge == 'gpt')].t_p_holm.iloc[0]:.3f}, Claude: {tests[(tests.criterion == 'overall') & (tests.model == 'PL') & (tests.judge == 'claude')].t_p_holm.iloc[0]:.3f}.</li>
  </ul>
  </div><div>
  <p><b>(6) "유의하지 않음"을 읽는 법.</b> 유의하지 않다는 것은 "효과가 없다"가 아니라 "<b>이 자료로는 효과가 0이라는 가설을 버릴 만큼 증거가 충분하지 않다</b>"는 뜻이다. 그래서 신뢰구간을 함께 보고, 효과가 있다면 최대 어느 정도인지(신뢰구간 상한)를 함께 말한다. 반대로 "유의하다"는 효과가 크다는 뜻이 아니므로, 효과 크기(%p)를 함께 보고한다.</p>
  </div></div>
"""


# ------------------------------------------------------------------ 5절 본문 (학습 → val loss → 교차 loss → 완결성)
ppl = {s_: {a: float(np.exp(cl[a][s_])) for a in J} for s_ in J}
bppl = {a: float(np.exp(base[a])) for a in J}
toy = [0.5, 0.2, 0.1]
toy_nll = [-np.log(q) for q in toy]
toy_loss = float(np.mean(toy_nll))
t_toy = table(["답변 토큰", "학생이 그 토큰에 준 확률 p", "−log p (그 토큰의 loss)"],
              [f"<td>토큰 {i + 1}</td><td class='num'>{q}</td><td class='num'>{v:.3f}</td>" for i, (q, v) in enumerate(zip(toy, toy_nll))]
              + [f"<td><b>평균</b></td><td class='num'>기하평균 {np.prod(toy) ** (1 / 3):.3f}</td><td class='num'><b>loss = {toy_loss:.3f}</b></td>"])
cross_rows = [f"<td>학습 전 Qwen</td>" + "".join(f"<td class='num'>{base[a]:.3f}<br><span class='m'>ppl {bppl[a]:.2f}</span></td>" for a in J)]
cross_rows += [f"<td>{N[s_]} 학생</td>" + "".join(f"<td{own(s_, a)}>{cl[a][s_]:.3f}<br><span class='m'>ppl {ppl[s_][a]:.2f}</span></td>" for a in J) for s_ in J]
t_cross2 = table(["학생 / 보여 준 답변", "GPT 답변", "Claude 답변", "Gemini 답변"], cross_rows)

seed_sd = pd.read_csv(ROOT / "sft/runs/cross_loss.csv").groupby(["student_teacher", "eval_teacher"]).loss.std().max()
gap = {s_: [(ppl[s_][a] / ppl[s_][s_] - 1) * 100 for a in J if a != s_] for s_ in J}
ex_tok = 300
ex_ratio = ex_tok * (cl["claude"]["gpt"] - cl["gpt"]["gpt"])
BOX_PPL = f"""  <div class="note"><b>숫자 감각: 이 정도 perplexity면 낮은 것인가?</b>
  <ul style="margin-bottom:0">
    <li><b>아무것도 모를 때와 비교:</b> Qwen3 토크나이저의 어휘는 약 15만 개다. 무작위로 고르면 perplexity가 약 150,000이다. 9~10은 15만 개 후보를 사실상 10개 정도로 좁혔다는 뜻이다(정답 토큰에 평균 약 0.1의 확률).</li>
    <li><b>1에 가까워질 수 없는 이유:</b> 같은 질문에도 맞게 답하는 방법(예시, 단어, 문장 연결)이 많다. 어미·조사는 쉽게 맞히지만 내용이 갈리는 지점은 어렵다. 그래서 자유 서술형 한국어 답변을 0.6B 모델이 읽을 때 10 안팎은 무난한 수준이다. 토크나이저와 텍스트가 다르면 perplexity를 직접 비교할 수 없으므로, 다른 연구의 값과 비교해 "낮다"고 말하지는 않는다.</li>
    <li><b>이 연구에서 중요한 것은 같은 조건 안의 상대 비교다:</b>
      학습으로 {bppl['gpt']:.1f} → {ppl['gpt']['gpt']:.1f}({(1 - ppl['gpt']['gpt'] / bppl['gpt']) * 100:.0f}% 감소, GPT 답변).
      자기 교사의 답은 다른 교사의 답보다 GPT 학생 {min(gap['gpt']):.0f}~{max(gap['gpt']):.0f}%, Claude 학생 {min(gap['claude']):.0f}~{max(gap['claude']):.0f}%, Gemini 학생 {min(gap['gemini']):.0f}~{max(gap['gemini']):.0f}% 낮다.
      반면 seed 간 흔들림은 loss 표준편차 최대 {seed_sd:.4f}, perplexity로 약 {seed_sd * 100:.1f}%다. <b>자기 교사 효과가 학습 무작위성보다 수십~수백 배 크다.</b></li>
    <li><b>토큰당 차이는 답 전체에서 곱으로 쌓인다:</b> perplexity는 토큰 하나당 평균이다. 예를 들어 {ex_tok}토큰짜리 답이라면, GPT 학생이 GPT 답을 "자기가 쓸 법한 글"로 볼 확률은 Claude 답보다 약 e<sup>{ex_ratio:.0f}</sup>배 크다(토큰당 loss 차 {cl['claude']['gpt'] - cl['gpt']['gpt']:.3f} × {ex_tok}).</li>
  </ul></div>
"""

SEC5 = f"""
  <h3>5.1 학습 과정</h3>
  <div class="cols"><div>
  <ul>
    <li><b>시작 모델:</b> Qwen3-0.6B-Base. 대화 스타일이 없는 Base 모델이라 학생의 스타일은 교사에게서 왔다고 말할 수 있다.</li>
    <li><b>학생 15명 = 교사 3 × seed 5 (42~46).</b> 같은 교사 안에서는 학습 무작위성(데이터 순서, dropout)만 바꿔 다섯 번 학습했다. 교사 간에는 답변 외의 모든 조건이 같다.</li>
    <li><b>형식:</b> ChatML(시스템 프롬프트 + 질문 + 교사 답 + 종료 토큰). 시스템 프롬프트와 질문은 손실에서 가리고(-100) <b>교사 답만 학습</b>한다.</li>
  </ul>
  </div><div>
  <ul>
    <li><b>설정(15명 동일):</b> full fine-tuning, AdamW lr 1e-5, cosine(warmup 5%), 실효 배치 16, 3 epoch = 189 step(1 epoch = 63 step), bf16, Colab A100. 15개 run_info의 하이퍼파라미터·데이터 해시가 한 종류임을 확인했다.</li>
    <li><b>응답 생성:</b> 학습이 끝난 학생이 평가 질문 400개에 greedy 디코딩(반복 억제 1.1, 최대 1,024토큰)으로 답한다. 이 답을 6절에서 평가한다.</li>
  </ul>
  </div></div>

  <h3>5.2 loss와 perplexity란 무엇인가</h3>
  <div class="cols"><div>
  <p>5절의 지표는 모두 <b>"학생이 이미 쓰여 있는 교사의 답을 얼마나 '자기가 쓸 법한 글'로 보는가"</b>를 잰다. 학생에게 답을 새로 쓰게 하지 않는다. 교사의 답을 한 토큰씩 보여 주면서, 매 위치에서 학생이 <b>실제로 나온 다음 토큰</b>에 준 확률 p를 읽는다.</p>
  <p><b>loss (교차 엔트로피).</b> 확률을 그대로 평균하지 않고, 각 토큰의 <b>−log p</b>를 평균한다.</p>
<pre>loss = −(1/N) × Σ log p(y_i | 질문, y₁ … y_(i−1))      N = 답변 토큰 수

p = 1   → −log p = 0      (확신하고 맞힘)
p = 0.5 → −log p = 0.69
p = 0.1 → −log p = 2.30   (드물다고 본 토큰일수록 크게 벌점)</pre>
  <ul>
    <li><b>왜 log인가:</b> 답 전체가 나올 확률은 토큰 확률의 곱 p₁ × p₂ × … 이다. log를 씌우면 곱이 합이 되고, 토큰 수 N으로 나누면 길이가 다른 답끼리도 비교할 수 있는 "토큰 하나당 평균"이 된다. 학습 때 줄이는 값과 같은 값이다.</li>
    <li><b>무엇을 계산에 넣나:</b> 시스템 프롬프트와 질문 부분은 빼고(-100), 교사 답변 토큰과 끝 토큰만 넣는다. 질문 199개의 모든 답변 토큰을 하나로 모아 평균한다(토큰 가중 평균).</li>
  </ul>
  </div><div>
  <p><b>작은 예.</b> 답변 토큰이 3개이고 학생이 각 토큰에 0.5, 0.2, 0.1의 확률을 줬다면</p>
  {t_toy}
<pre>perplexity = e^loss = e^{toy_loss:.3f} = {np.exp(toy_loss):.2f}
           = 1 ÷ (확률의 기하평균 {np.prod(toy) ** (1 / 3):.3f})</pre>
  <p><b>perplexity(혼란도).</b> loss를 다시 확률 크기로 되돌린 값이다(e<sup>loss</sup>). 뜻은 <b>"학생이 매 토큰마다 평균적으로 몇 개의 후보 중에서 고르는 것만큼 불확실한가"</b>이다.</p>
  <ul>
    <li>perplexity 1 = 매번 정답 토큰을 확률 1로 확신(완벽).</li>
    <li>perplexity 10 = 매번 동등한 후보 10개 중 하나를 고르는 것과 같은 불확실성. 정답 토큰에 평균(기하평균) 1/10 = 0.1의 확률을 줬다는 뜻이기도 하다.</li>
    <li>loss의 작은 차이가 perplexity에서는 체감되는 차이가 된다. loss 0.1 차이 = perplexity 약 10% 차이.</li>
  </ul>
  </div></div>

  <h3>5.3 학습이 잘 되었는가: val loss</h3>
  <div class="cols"><div>
  <ul>
    <li><b>무엇:</b> 학습에 쓰지 않은 질문 199개에서, <b>자기 교사의 답</b>에 대한 loss(5.2).</li>
    <li><b>언제:</b> 학습 시작 전(step 0), 학습 중 10 step마다(10, 20, …, 180), 마지막 step 189 → 학생마다 20개 시점.</li>
    <li><b>왜:</b> (1) 학습 전보다 교사의 답을 "자기가 쓸 법한 글"로 보게 되었는지, (2) 같은 데이터를 반복해 보며 외우기만 하는지(과적합)를 보기 위해서다. 모델 선택에는 쓰지 않았다. 15명 모두 같은 시점(마지막 step 189)의 모델을 써야 교사 간 조건이 같기 때문이다.</li>
  </ul>
  {t_sft}
  <p class="m">표 5-1. 교사별 학생 5명(seed)의 val loss. 감소량 = 최종 − 학습 전.</p>
  <p><b>그림 5-1 읽기.</b> 세 교사 모두 학습 전보다 0.26~0.37 낮아졌다(perplexity로 GPT {np.exp(sft['teachers']['gpt']['val_before']):.1f} → {np.exp(sft['teachers']['gpt']['val_final_mean']):.1f}, Claude {np.exp(sft['teachers']['claude']['val_before']):.1f} → {np.exp(sft['teachers']['claude']['val_final_mean']):.1f}, Gemini {np.exp(sft['teachers']['gemini']['val_before']):.1f} → {np.exp(sft['teachers']['gemini']['val_final_mean']):.1f}). step 120(2 epoch 끝 무렵)에서 가장 낮고 3 epoch째에 약간(1~2%) 오르는 가벼운 과적합이 있지만, 학습 전보다 훨씬 낮다. 다섯 seed의 띠가 선과 거의 겹칠 만큼 학습이 안정적이다. 교사 간 절대값은 학습 전부터 다르므로 비교하지 않는다(5.4에서 다룬다).</p>
  </div><div>
  {fig_val}
  </div></div>

  <h3>5.4 학생이 자기 교사의 스타일을 흡수했는가: 교차 loss</h3>
  <div class="cols"><div>
  <p><b>무엇을 쟀나.</b> 5.3의 val loss는 자기 교사의 답만 본다. 교차 loss는 학생 3종(GPT·Claude·Gemini 학생)에게 <b>세 교사의 검증 답변을 모두</b> 보여 주고(3 × 3) 같은 loss를 잰다. "교차"는 다른 교사의 답에도 같은 계산을 한다는 뜻이다.</p>
  <p><b>왜 했나.</b></p>
  <ul>
    <li><b>선호 누출의 전제 확인:</b> 평가자가 자기 계열 학생을 알아보려면, 학생이 "한국어 어시스턴트 답변" 일반이 아니라 <b>그 교사 고유의 말투·형식</b>을 물려받았어야 한다. val loss가 줄었다는 것만으로는 알 수 없다. 세 학생이 공통 형식만 배워도 val loss는 줄기 때문이다.</li>
    <li><b>val loss끼리는 비교할 수 없다:</b> 학습 전 Qwen부터 교사 답변마다 loss가 다르다(GPT {base['gpt']:.3f}, Claude {base['claude']:.3f}, Gemini {base['gemini']:.3f}). 그래서 <b>같은 답변을 서로 다른 학생에게</b> 보여 주는 비교가 필요하다.</li>
    <li><b>API가 필요 없고 결정적이다:</b> 생성하지 않고 확률만 계산하므로 무작위성이 없다. 4절의 "세 교사의 답변 분포가 다르다"는 결과를 학생 쪽에서 같은 관점(확률 분포)으로 확인하는 방법이다.</li>
  </ul>
  <p class="m">대상: 각 교사의 검증 답변 199개(답변 토큰 GPT 52,402 · Claude 80,021 · Gemini 65,608개). 학생 15명 × 교사 답변 3종 = 45개 값. 학습 전 값은 step 0의 val loss.</p>
  </div><div>
  {t_cross2}
  <p class="m">표 5-2. 교차 loss(위)와 perplexity(아래, = e<sup>loss</sup>). 행 = 학생, 열 = 보여 준 교사의 답변. 작을수록 학생이 그 답을 "자기가 쓸 법한 글"로 본다. seed 5개 평균, 굵은 값 = 자기 교사의 답. seed 간 loss 표준편차 0.0005~0.0021.</p>
  <p><b>표 5-2를 두 방향으로 읽는다.</b></p>
  <ul>
    <li><b>가로(행, 같은 학생):</b> 한 학생이 세 교사의 답 중 <b>자기 교사의 답에서 가장 낮은가</b>. 예: GPT 학생 행은 GPT 답 {cl['gpt']['gpt']:.3f}(ppl {ppl['gpt']['gpt']:.1f}) &lt; Claude 답 {cl['claude']['gpt']:.3f}({ppl['gpt']['claude']:.1f}) &lt; Gemini 답 {cl['gemini']['gpt']:.3f}({ppl['gpt']['gemini']:.1f}). → 학생 15명 중 15명 성립.</li>
    <li><b>세로(열, 같은 답변):</b> 같은 교사의 답을 세 학생에게 보였을 때 <b>그 교사의 학생이 가장 낮은가</b>. 예: Gemini 답 열은 Gemini 학생 {cl['gemini']['gemini']:.3f}(ppl {ppl['gemini']['gemini']:.1f}) &lt; Claude 학생 {cl['gemini']['claude']:.3f}({ppl['claude']['gemini']:.1f}) &lt; GPT 학생 {cl['gemini']['gpt']:.3f}({ppl['gpt']['gemini']:.1f}). 같은 글끼리 비교하므로 답 자체가 쉽거나 어려운 정도가 상쇄된다(6절의 "같은 학생을 평가자끼리 비교"와 같은 논리). → seed 5 × 교사 3 = 15번 중 15번 성립.</li>
  </ul>
  </div></div>
{BOX_PPL}
  <div class="cols"><div>
  {fig_cross}
  </div><div>
  <p><b>그림 5-2 읽기: 학습 전보다 얼마나 줄었나.</b> 표 5-2의 각 칸에서 첫 행(학습 전 Qwen)을 뺀 값이다. 모든 학생이 세 교사의 답 모두를 학습 전보다 "자기가 쓸 법한 글"로 보게 되었다. 세 교사에 공통인 한국어 어시스턴트 답변 형식을 배웠기 때문이다. 그러나 <b>줄어든 폭은 언제나 자기 교사의 답에서 가장 컸다</b>.</p>
  <ul>
    <li>GPT 학생: GPT 답 −{base['gpt'] - cl['gpt']['gpt']:.2f}, Claude 답 −{base['claude'] - cl['claude']['gpt']:.2f}, Gemini 답 −{base['gemini'] - cl['gemini']['gpt']:.2f}</li>
    <li>Claude 학생: Claude 답 −{base['claude'] - cl['claude']['claude']:.2f}, GPT 답 −{base['gpt'] - cl['gpt']['claude']:.2f}, Gemini 답 −{base['gemini'] - cl['gemini']['claude']:.2f}</li>
    <li>Gemini 학생: Gemini 답 −{base['gemini'] - cl['gemini']['gemini']:.2f}, GPT 답 −{base['gpt'] - cl['gpt']['gemini']:.2f}, Claude 답 −{base['claude'] - cl['claude']['gemini']:.2f}</li>
  </ul>
  <div class="note"><b>5.4 결론.</b> 학생은 "한국어로 답하는 법" 일반을 넘어 <b>자기 교사의 고유한 스타일</b>을 흡수했다(행·열 비교 모두 15/15). 평가자가 학생의 답에서 자기 계열의 흔적을 알아볼 수 있는 조건이 갖춰졌다.</div>
  </div></div>

  <h3>5.5 학생 응답의 완결성</h3>
  <p>평가 질문 400개 중 끝 토큰으로 스스로 끝맺고 1,024토큰 미만인 "완전한 답"의 수. 15명 모두 완전한 질문 144개 중 100개를 6절의 LLM 평가에 썼다.</p>
  {t_complete}
  <p class="m">표 5-3. 학생별 완전한 답의 수(400개 중).</p>
  <div class="note"><b>5절 결론.</b> (1) 학습에 안 쓴 답의 loss가 0.26~0.37 줄어 학습이 잘 되었다. (2) seed 간 최종 val loss 차이가 0.001 수준으로 학습이 안정적이다. (3) <b>학생 15명 모두 자기 교사의 답에서 loss가 가장 낮고 감소량도 가장 컸다(15/15)</b>. 교사의 스타일이 학생에게 옮겨졌다.</div>
"""

# ------------------------------------------------------------------ 문서
CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--line:#e4e3df;
--accent:#2a78d6;--note:#f4f7fc;--warn:#fdf6ec;--c-gpt:#2a78d6;--c-claude:#eb6834;--c-gemini:#1baf7a;--c-neutral:#4a3aa7;--c-neutral2:#a8a69e;--sig:#eef4fc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--surface:#1a1a19;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--line:#383835;
--accent:#3987e5;--note:#1f2a38;--warn:#33291a;--c-gpt:#3987e5;--c-claude:#d95926;--c-gemini:#199e70;--c-neutral:#9085e9;--c-neutral2:#6f6e68;--sig:#1f2a38}}
:root[data-theme="dark"]{--bg:#0d0d0d;--surface:#1a1a19;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--line:#383835;
--accent:#3987e5;--note:#1f2a38;--warn:#33291a;--c-gpt:#3987e5;--c-claude:#d95926;--c-gemini:#199e70;--c-neutral:#9085e9;--c-neutral2:#6f6e68;--sig:#1f2a38}
*{box-sizing:border-box;-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Apple SD Gothic Neo","Pretendard","Noto Sans KR",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}
header.top{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px 28px;margin-bottom:18px}
header.top h1{font-size:28px;line-height:1.3;margin:0 0 6px}
header.top .sub{color:var(--ink2);margin:0}
.hero{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:18px}
.hero div{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.hero b{display:block;font-size:24px;line-height:1.2}
.hero span{font-size:13px;color:var(--ink2)}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 26px;margin:16px 0}
section h2{font-size:21px;margin:0 0 12px;display:flex;gap:10px;align-items:center}
section h2 .no{display:inline-flex;width:30px;height:30px;border-radius:50%;background:var(--ink);color:var(--card);font-size:16px;align-items:center;justify-content:center;flex:none}
h3{font-size:16.5px;margin:22px 0 8px}
p{margin:8px 0}
.cols{display:block}
.note,.warn{border-radius:10px;padding:12px 16px;margin:12px 0}
.note{background:var(--note);border-left:4px solid var(--accent)}
.warn{background:var(--warn);border-left:4px solid #eda100}
figure{margin:12px 0;padding:0}
figcaption{font-size:13px;color:var(--ink2);margin-top:4px}
.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .axis{stroke:var(--axis);stroke-width:1}
.chart .zero{stroke:var(--ink2);stroke-width:1.2}
.chart .ref{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 4}
.chart .refl{fill:var(--muted);font-size:12px}
.chart .tick{fill:var(--muted);font-size:12px}
.chart .lab{fill:var(--ink2);font-size:13px}
.chart .val{fill:var(--ink2);font-size:12px}
.chart .b{font-weight:700;fill:var(--ink)}
.chart .ci{stroke:var(--ink2);stroke-width:1.5}
.chart rect.hl{stroke:var(--ink);stroke-width:2}
.chart .dot{stroke:var(--surface);stroke-width:2}
.chart .dot.own{stroke:var(--ink);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--ink2);margin-bottom:4px}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.tw{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card)}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{border-bottom:2px solid var(--ink);font-weight:700}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.own{font-weight:700}
td.sig{background:var(--sig)}
.m{color:var(--muted);font-size:12px}
.flow{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.flow div{flex:1 1 150px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:13.5px}
.flow b{display:block;color:var(--accent);font-size:12.5px}
.eq{font-family:"Times New Roman","Apple SD Gothic Neo",serif;font-size:17px;text-align:center;margin:12px 0;padding:10px 12px;background:var(--surface);border:1px solid var(--line);border-radius:8px;line-height:2.1;overflow-x:auto}
.eq sub{font-size:0.7em}.frac{display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;margin:0 4px}.frac span:first-child{border-bottom:1px solid var(--ink);padding:0 4px}.frac span{line-height:1.6}
pre{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;font:12.5px/1.55 "SF Mono",Menlo,monospace;white-space:pre-wrap;overflow-x:auto}
.abstract{font-size:14.5px;line-height:1.8;text-align:justify}
.kw{color:var(--ink2);font-size:14px}
ul{margin:6px 0;padding-left:20px}
li{margin:3px 0}
@media (max-width:820px){.cols{grid-template-columns:1fr}.hero{grid-template-columns:1fr 1fr}section{padding:18px 16px}}
@page{size:A4;margin:12mm 11mm}
@media print{body{background:#fff;font-size:12.5px}main{padding:0;max-width:none}section,header.top{break-inside:auto;border-color:#ddd}
figure,.tw,.note,.warn,.hero,.flow{break-inside:avoid}h2,h3{break-after:avoid}table{font-size:11.5px}th,td{padding:5px 6px}}
"""

doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM 평가자 선호 누출 포스터</title><style>{CSS}</style></head><body><main>

<header class="top">
  <h1>한국어 합성데이터 증류에서 LLM 평가자의 선호 누출 추정</h1>
  <p class="sub">GPT·Claude·Gemini가 만든 답으로 학습한 학생 모델을, 같은 세 모델이 평가할 때 자기 계열 학생을 더 좋게 보는가?</p>
  <div class="hero">
    <div><b>{clf[3]['acc']:.1f}%</b><span>서식을 지워도 답변만으로 생성 모델을 구별 (KLUE-RoBERTa, 우연 33.3%)</span></div>
    <div><b>15 / 15</b><span>학생 15명 모두 자기 교사의 답에서 loss가 가장 낮음 (교차 loss)</span></div>
    <div><b>{sgn(gp.pp)}%p</b><span>Gemini 평가자의 자기 계열 효과: 품질이 같은 상대를 이길 확률 (Holm p {pf(gp.p_holm)})</span></div>
    <div><b>{posA['gpt']:.0f}%</b><span>첫 번째(A) 답을 1위로 뽑은 비율, GPT 평가자 (기대 33.3%)</span></div>
  </div>
</header>

<section id="s1"><h2><span class="no">1</span>연구 배경</h2>
  <div class="cols"><div>
  <p><b>합성데이터 증류.</b> 작은 모델을 학습시킬 사람이 쓴 데이터가 부족하면, GPT·Claude·Gemini 같은 큰 모델(교사)에게 답을 쓰게 해 그 답으로 작은 모델(학생)을 학습한다. 학생은 교사의 지식뿐 아니라 <b>말투, 형식, 어휘</b>까지 물려받는다.</p>
  <p><b>LLM-as-a-Judge.</b> 학생의 성능은 사람이 채점하기에 비싸서, 다시 큰 모델에게 채점을 맡긴다(Zheng et al., 2023). 평가가 빠르고 사람 판단과도 꽤 일치하지만, LLM 평가자에게는 <b>위치 편향</b>(먼저 나온 답 선호), <b>길이 편향</b>, <b>자기 선호</b>(자기 출력 선호; Panickssery et al., 2024) 같은 체계적 편향이 있다고 보고되었다.</p>
  </div><div>
  <p><b>선호 누출(preference leakage).</b> 데이터를 만든 모델과 채점하는 모델이 같거나 관련되어 있으면, 평가자가 <b>자기 문체를 물려받은 학생</b>을 품질과 무관하게 더 좋게 볼 수 있다(Li et al., 2025). 이 현상은 영어 벤치마크에서 처음 보고되었다.</p>
  <div class="note"><b>왜 중요한가.</b> 같은 계열의 모델로 데이터를 만들고 평가까지 하는 일은 흔하다. 누출이 있으면 "어느 학생 모델이 더 좋은가"라는 결론이 실제 품질이 아니라 <b>평가자와의 계보</b>로 정해질 수 있다. 한국어 합성데이터에서 이 현상이 나타나는지, 얼마나 큰지는 아직 확인되지 않았다.</div>
  </div></div>
</section>

<section id="s2"><h2><span class="no">2</span>연구 질문</h2>
{rq}
</section>

<section id="s3"><h2><span class="no">3</span>데이터</h2>
  <div class="flow">
    <div><b>① 원천</b>beomi/KoAlpaca-v1.1a 21,155개 (네이버 지식iN 질문 기반 한국어 지시 데이터)</div>
    <div><b>② 추출</b>빈 질문·중복 제거 후 seed 42로 학습용 1,200개 + 평가용 400개 (서로 겹치지 않음)</div>
    <div><b>③ 교사 생성</b>GPT·Claude·Gemini가 같은 시스템 프롬프트로 1,200개에 답</div>
    <div><b>④ 학습 데이터</b>세 교사 모두 답한 1,199개 중 1,000개 학습 + 199개 검증</div>
    <div><b>⑤ 평가 데이터</b>학생 15명이 평가 질문 400개에 답 → 모두 끝까지 쓴 144개 중 100개를 평가</div>
  </div>
  <div class="cols"><div>
  <h3>교사 답변 생성</h3>
  {table(["항목", "GPT", "Claude", "Gemini"], [
      "<td>모델</td><td>gpt-5.6-terra</td><td>claude-sonnet-5-5</td><td>gemini-3.8-flash</td>",
      "<td>추론 설정</td><td>effort medium</td><td>adaptive, effort medium</td><td>thinking medium</td>",
      "<td>공통</td><td colspan='3'>같은 시스템 프롬프트(한국어, 600자 이내, 단정 금지 등), temperature 기본값, 출력 한도 4,096토큰</td>",
      "<td>실패</td><td>0</td><td>1 (안전 필터 거절 → 세 교사 모두에서 제외)</td><td>0</td>"], num_from=9)}
  <p>마크다운(굵게, 목록) 사용 여부는 일부러 지정하지 않아, 형식 차이도 모델 고유 스타일로 남겼다.</p>
  </div><div>
  <h3>데이터 구성</h3>
  {table(["용도", "질문 수", "설명"], [
      "<td>학생 학습</td><td class='num'>1,000</td><td>세 교사 답 각 1,000개, 같은 질문 id</td>",
      "<td>학습 중 검증</td><td class='num'>199</td><td>학습에 안 쓴 질문. val loss, 교차 loss</td>",
      "<td>분류 실험</td><td class='num'>1,199 × 3</td><td>질문 단위 70/15/15 분할 (train 839, val 179, test 181)</td>",
      "<td>학생 응답</td><td class='num'>400</td><td>학습 질문과 겹치지 않는 평가 질문</td>",
      "<td>LLM 평가</td><td class='num'>100</td><td>학생 15명 모두 끝까지 답한 144개 중 추첨</td>"], num_from=9)}
  <p class="m">길이 검사: 학습 형식 전체를 Qwen3 토크나이저로 세면 최대 945토큰으로 1,024를 넘는 예제는 없어, 길이 때문에 빠진 질문은 없다.</p>
  </div></div>
</section>

<section id="s4"><h2><span class="no">4</span>LLM의 답변만으로 어떤 모델인지 구별할 수 있는가?</h2>
  <p><b>왜 필요한가.</b> 평가자가 자기 계열을 선호하려면, 먼저 <b>세 교사의 답변이 통계적으로 서로 달라야</b> 한다. 각 LLM은 학습 데이터와 정렬 과정이 달라 서로 다른 조건부 분포 P<sub>모델</sub>(답변 | 질문)을 학습했을 것이다. 분포가 정말 다르다면, 답변만 보고 어느 모델에서 나왔는지 맞히는 분류기를 만들 수 있다. 분포가 같다면 어떤 분류기도 우연(33.3%)을 넘을 수 없다. 분류 정확도는 분포 차이의 <b>보수적 하한</b>이다.</p>
  <div class="cols"><div>
  <h3>실험 설계</h3>
  <ul>
    <li><b>재료:</b> 세 교사가 모두 답한 질문 1,199개의 답변 3,597개. 레이블 = 생성 모델.</li>
    <li><b>질문 단위 분할:</b> 같은 질문의 세 답은 항상 같은 split → test 질문은 처음 보는 질문이므로 질문 내용이 아니라 <b>답변 스타일</b>로만 맞혀야 한다. 세 클래스가 정확히 1/3씩.</li>
    <li><b>분류기 ① 다항 로지스틱 회귀:</b> 글자 2~4-gram TF-IDF, 규제 강도 C를 val로 선택. 단순한 빈도 통계만 본다.</li>
    <li><b>분류기 ② KLUE-RoBERTa:</b> 한국어 사전학습 인코더(1.1억 파라미터)를 미세조정(4 epoch, lr 2e-5, 실효 배치 16), val 정확도 최고 체크포인트. 문맥과 뉘앙스를 본다.</li>
    <li><b>통제 실험(정규화):</b> 굵게·목록 기호·따옴표·줄바꿈을 모두 지운 텍스트로 다시 실험. 겉모양을 지워도 구별되면 차이가 <b>어휘·어미·문장 구성</b>에 있다는 뜻이다.</li>
  </ul>
  {t_clf}
  </div><div>
  {fig_clf}
  {fig_f1}
  </div></div>
  <div class="cols"><div>
  {fig_style}
  </div><div>
  <h3>결과</h3>
  <ul>
    <li><b>답변만으로 생성 모델이 거의 완벽하게 구별된다:</b> 원문 97.2%(로지스틱), 97.1%(RoBERTa). 우연 33.3%의 약 3배.</li>
    <li><b>서식을 지워도 구별된다:</b> 정규화 텍스트에서도 92.1%, 96.5%. 정규화에서 RoBERTa가 로지스틱보다 높아, 단어 빈도를 넘어선 <b>문장 수준의 뉘앙스 차이</b>도 있다.</li>
    <li><b>세 모델의 프로필:</b> GPT는 짧고 완곡한 문단형(중앙값 380자, 완곡 표현 65%), Claude는 길고 구조화된 목록형(587자, 글머리 목록 76%, "정리하면" 마무리), Gemini는 번호로 나눈 설명형(482자, 번호 목록 49%, 영문 괄호 병기 67%).</li>
  </ul>
  <div class="note"><b>결론.</b> 세 LLM의 한국어 답변은 형식뿐 아니라 어휘와 뉘앙스에서도 통계적으로 구별되는 서로 다른 분포를 가진다. 학생이 이 차이를 물려받는다면, 평가자가 자기 계열 답을 알아볼 단서가 존재한다 → <b>선호 누출의 전제 조건이 충족</b>된다.</div>
  </div></div>
</section>

<section id="s5"><h2><span class="no">5</span>학생 모델 학습 (SFT)</h2>
{SEC5}
</section>

<section id="s6"><h2><span class="no">6</span>LLM-as-a-Judge: 설계, 과정, 결과</h2>
  <h3>6.1 설계</h3>
  <div class="cols"><div>
  <ul>
    <li><b>평가자 3명:</b> 교사와 같은 GPT·Claude·Gemini. 같은 프롬프트, 추론 강도 medium, 출력 한도 12,288토큰. 답의 출처는 보여 주지 않는다(블라인드).</li>
    <li><b>listwise 순위 평가:</b> 같은 질문에 대한 <b>같은 seed의 GPT·Claude·Gemini 학생 답 3개</b>를 한 번에 보여 주고 1~3위를 매긴다(동점 없음). 처음 시도한 1~5점 채점은 0.6B 학생 답이 대부분 사실을 지어내 사실 정확성이 120개 모두 1점(바닥 효과)이어서 순위 방식으로 바꿨다.</li>
    <li><b>출력 5개:</b> FLASK(Ye et al., 2024) 기반 평가 기준 4개(사실 정확성, 질문 충족도, 논리성, 가독성) 각각의 근거와 순위 + 네 기준을 종합한 <b>최종 순위(overall)</b>. 주 결과는 최종 순위다.</li>
    <li><b>위치 편향 통제:</b> 같은 세 답을 <b>순서만 바꿔 3번</b> 넣는다(각 학생이 A·B·C에 한 번씩, 라틴 방격). 시작 순서는 6가지 순열을 돌아가며 쓰고, 처리 순서는 무작위로 섞었다.</li>
  </ul>
  </div><div>
<pre>같은 질문, 같은 seed의 세 답      답변 A        답변 B        답변 C
순서 1                             GPT 학생      Claude 학생   Gemini 학생
순서 2                             Claude 학생   Gemini 학생   GPT 학생
순서 3                             Gemini 학생   GPT 학생      Claude 학생</pre>
  {table(["단위", "개수"], ["<td>질문</td><td class='num'>100</td>", "<td>묶음 (질문 × seed 5)</td><td class='num'>500</td>",
       "<td>평가 항목 (묶음 × 순서 3)</td><td class='num'>1,500</td>", "<td>API 평가 (× 평가자 3)</td><td class='num'>4,500</td>",
       "<td>순위 (× 기준 4 + 최종 1)</td><td class='num'>22,500</td>", "<td>평가자 1명이 학생 1명에게 매긴 순위</td><td class='num'>1,500</td>"])}
  <p class="m">실패 0건(Gemini 형식 오류 11건은 재요청으로 해결). 소요: GPT 53분, Claude 25분, Gemini 37분.</p>
  </div></div>

{SEC62}

{SEC6R}
</section>

<section id="s7"><h2><span class="no">7</span>결과 해석</h2>
  <div class="note" style="font-size:15.5px"><b>연구 질문: "한국어 합성데이터에서도 LLM 평가자는 자기 모델이 만든 데이터로 학습한 모델의 답을 더 선호하는가?"</b><br>
  <b>답: 그렇다. 다만 모든 평가자가 아니라 평가자 모델에 따라 다르다.</b> Gemini 평가자는 학생 품질과 위치를 통제한 뒤에도 자기 계열 학생을 분명히 더 높게 평가했고, GPT 평가자는 같은 방향의 약한 효과, Claude 평가자는 효과가 없었다.</div>
{SEC7A}
  <h3>7.3 종합: 연구 질문에 대한 답과 근거</h3>
  <div class="cols"><div>
  <h3>왜 이렇게 결론 내릴 수 있는가</h3>
  <ol>
    <li><b>전제가 성립한다:</b> 세 교사의 답은 서식을 지워도 96.5%로 구별되고(4절), 학생 15명 모두 자기 교사의 스타일을 흡수했다(5절). 평가자가 자기 계열을 알아볼 단서가 있다.</li>
    <li><b>품질이 아니라 평가자 차이다:</b> 같은 학생의 같은 답을 평가자끼리 비교했고, 학생 품질(15명)과 평가자별 위치 효과를 모형에 넣어 통제했다. 각 쌍에 중립인 제3의 평가자를 기준으로 삼아 세 평가자의 효과를 분리했고, 이를 위한 가정은 자료와 맞았다(p = {overid[overid.criterion == 'overall'].p.iloc[0]:.2f}).</li>
    <li><b>Gemini의 효과는 크고 강건하다:</b> 1위 확률 {sgn(gm.pp)}%p(33.3% → {33.3 + gm.pp:.1f}%), 승률 {sgn(gp.pp)}%p(50% → {50 + gp.pp:.1f}%), 같은 두 답을 직접 비교한 가장 엄격한 모형에서 {sgn(gl.pp)}%p. 세 모형 모두 Holm p ≤ {max(gm.p_holm, gp.p_holm, gl.p_holm):.3f}.</li>
    <li><b>우연이 아니다:</b> Gemini의 효과는 학생을 다섯 번 따로 학습한 모든 경우에서 양수였다(5/5, t검정 Holm p = {tests[(tests.criterion == 'overall') & (tests.model == 'PL') & (tests.judge == 'gemini')].t_p_holm.iloc[0]:.3f}).</li>
    <li><b>최종 순위만의 현상이 아니다:</b> Gemini는 사실 정확성·질문 충족도·가독성에서 세 모형 모두 유의했고, 누출이 가장 큰 기준은 <b>사실 정확성</b>이었다. 문체 취향을 넘어 자기 계열의 서술로 쓰인 내용을 더 정확하다고 판단했다.</li>
    <li><b>길이로 설명되지 않는다:</b> 순위 로짓 모형에 평가자별 길이 항을 추가해 다시 적합해도 Gemini의 승률 차는 {sgn(length[length.judge == 'gemini'].pp.iloc[0])}%p로 유지되었다(6.4.1). 세 평가자 모두 오히려 짧은 답을 약간 선호했다.</li>
  </ol>
  </div><div>
  <h3>가설별 판정</h3>
  {table(["가설", "판정", "근거"], [
      "<td>H1 자기 계열 선호</td><td><b>부분 지지</b></td><td>Gemini 강함, GPT 약함, Claude 없음</td>",
      "<td>H2 기준별 차이</td><td><b>지지 (예상과 다른 방향)</b></td><td>Gemini: 사실 정확성 최대, 논리성 최소</td>",
      "<td>H3 seed 반복</td><td><b>Gemini에서 지지</b></td><td>5/5 seed 양수, t검정 유의</td>"], num_from=9)}
  </div></div>
  <div class="cols"><div>
  <h3>방법론적 함의</h3>
  <ul>
    <li><b>단순 계산은 편향을 퍼뜨린다:</b> 다른 평가자를 기준선으로 쓰는 단순 계산으로는 세 평가자 모두 누출이 있어 보였다(승률 기준 GPT +4.3, Claude +1.8%p). 효과를 동시에 추정하면 Claude는 사라지고 GPT는 작아진다(7.1, 그림 7-1).</li>
    <li><b>위치 편향이 누출보다 크다:</b> 첫 번째 답이 1위로 뽑힐 오즈가 마지막보다 {min(orA.values()):.1f}~{max(orA.values()):.1f}배(Gemini 누출 오즈비 {gp.odds_ratio:.2f}배). 순서 균형 설계 없이는 누출을 잴 수 없다.</li>
  </ul>
  </div><div>
  <h3>한계</h3>
  <ul>
    <li>사람 평가가 없어 어느 평가자의 순위가 "옳은지"는 말할 수 없다. 누출은 평가자 간 체계적 차이로 정의했다.</li>
    <li>0.6B 학생의 답은 대부분 품질이 낮다. 높은 품질의 답 사이에서도 같은 크기인지는 알 수 없다.</li>
    <li>특정 모델 버전, 한국어 일반 질문 100개의 결과다. 중립 평가자(HyperCLOVA X)는 실행하지 않았다.</li>
  </ul>
  </div></div>
  <div class="note"><b>결론.</b> 한국어 합성데이터 증류에서도 LLM 평가자의 선호 누출은 존재할 수 있으며, 평가자에 따라 승부를 50:50에서 약 60:40으로 기울일 만큼 크다. LLM-as-a-Judge로 학생 모델을 비교할 때는 생성 모델과 다른 계열의 평가자를 쓰거나, 여러 계열의 평가자와 순서 균형 설계를 함께 쓰고 평가자 효과를 모형으로 분리해 보고해야 한다.</div>
</section>

<section id="s8"><h2><span class="no">8</span>초록</h2>
  <p class="abstract">{ABSTRACT}</p>
  <p class="kw"><b>주요용어:</b> {KEYWORDS}.</p>
</section>

<section id="refs"><h2>참고문헌</h2>
  <ol style="font-size:13px">
    <li>Li, D. et al. (2025). Preference Leakage: A Contamination Problem in LLM-as-a-judge. arXiv:2502.01534.</li>
    <li>Luce, R. D. (1959). Individual Choice Behavior. Wiley. · Plackett, R. L. (1975). The analysis of permutations. Applied Statistics, 24(2), 193–202.</li>
    <li>Panickssery, A., Bowman, S. R., &amp; Feng, S. (2024). LLM Evaluators Recognize and Favor Their Own Generations. NeurIPS 2024.</li>
    <li>Shi, L. et al. (2025). Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. arXiv:2406.07791.</li>
    <li>Wang, P. et al. (2024). Large Language Models are not Fair Evaluators. ACL 2024.</li>
    <li>Ye, S. et al. (2024). FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets. ICLR 2024.</li>
    <li>Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.</li>
    <li>Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70.</li>
    <li>Bradley, R. A., &amp; Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39(3/4), 324–345.</li>
    <li>McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), Frontiers in Econometrics, 105–142. Academic Press.</li>
    <li>Beggs, S., Cardell, S., &amp; Hausman, J. (1981). Assessing the potential demand for electric cars. Journal of Econometrics, 17(1), 1–19.</li>
    <li>Cameron, A. C., &amp; Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. Journal of Human Resources, 50(2), 317–372.</li>
  </ol>
  <p class="m">상세 방법과 전체 수치: description/result.html, teacher_style_analysis.html, sft_guide.html. 그림과 표는 results/poster.py가 분석 결과 파일에서 직접 생성했다.</p>
</section>
</main></body></html>"""

OUT.write_text(doc, encoding="utf-8")
print(f"저장: {OUT} ({len(doc):,}자)")
print("ABSTRACT", len(ABSTRACT), "자")
