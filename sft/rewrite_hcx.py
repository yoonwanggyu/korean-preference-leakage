"""8.2 문체 제거 실험: 세 교사의 학습 답 1,000개씩을 CLOVA HCX-005로 "내용은 그대로, 문체는 중립 규칙으로" 다시 쓴다.

왜 세 교사 모두인가: 시험에서 HCX-005가 쓴 중립 문체가 분류기 눈에 Gemini처럼 보였다(GPT·Claude 답을 다시 써도
대부분 Gemini로 판정). Gemini 답만 다시 쓰면 "내용"과 "Gemini와 닮은 중립 문체"가 섞이므로, 세 교사 답을 모두 같은
과정으로 다시 써서 학생 세 명이 같은 문체를 갖고 내용만 다르게 만든다.

답 하나를 다시 쓰는 과정 (원문 문장이 새 답에 넘어가지 않도록 두 단계로 나눈다)
  ① 내용 추출: 원문에서 정보(사실·이유·예시·조언 등)만 명제 목록으로 뽑는다. 원문의 숫자 목록을 함께 주고,
     목록에 숫자가 빠지면 다시 요청한다(최대 MAX_FIX번).
  ② 새로 쓰기: 질문과 ①의 목록만 보고(원문은 주지 않음) 중립 규칙대로 쓴다. 자동 검사(길이, 서식, 영문 괄호,
     정리 문장, 숫자 누락, 메타 문구)에 걸리면 문제를 알려 주고 다시 요청한다(최대 MAX_FIX번).
     끝까지 걸리면 문제가 가장 적은 시도를 고른다.
  ③ 기계적 정리: 남은 굵게·제목·목록 기호와 영문만 든 괄호를 정규식으로 지우고, 남은 문제를 기록한다.
호출 오류: 429(분당 한도)는 5초 쉬고 다시 보낸다(횟수를 센다). 5xx·시간 초과는 10초 쉬고 최대 5번. 그 밖의 오류는
그 답을 실패로 남기고(raw에 쓰지 않음) 다음 실행 때 다시 시도한다.

사용법
  python sft/rewrite_hcx.py              전체 실행(이어하기). 끝나면 학습 파일까지 만든다
  python sft/rewrite_hcx.py --limit 30   앞의 30개만(시험)
  python sft/rewrite_hcx.py --build      API 없이 raw 기록으로 학습 파일과 요약만 다시 만든다
출력
  sft/data/rewrite/raw_{teacher}.jsonl   답마다 추출 목록, 시도별 결과·문제, 최종 답, 남은 문제, 호출·429 수
  sft/data/rewrite/failures.log          실패한 답(오류 종류와 상태 코드만)
  sft/data/{teacher}_rw_train_1000.json  학습 데이터 [{id, category, instruction, output, source}] (train.py --teacher {teacher}_rw)
  sft/data/{teacher}_rw_val.json         원래 교사의 검증 답 199개 그대로(학습 중 val loss용, 다시 쓰지 않음)
  sft/data/rewrite/rewrite_summary.json  교사별 길이 비율, 검사 통과율, 숫자 누락, 호출·429 수, 설정과 프롬프트
API 키는 .env의 NEUTRAL_API_KEY를 쓰며 출력하지 않는다.
"""

import argparse
import json
import os
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(__file__).parent / "data"
RW_DIR = DATA_DIR / "rewrite"
TEACHERS = ["gpt", "claude", "gemini"]
MODEL = "HCX-005"
BASE_URL = os.environ.get("NEUTRAL_BASE_URL", "https://clovastudio.stream.ntruss.com/v1/openai/")
WORKERS = 4            # 동시 요청 수. 분당 약 30회 한도에 걸리므로 늘려도 빨라지지 않는다
TEMPERATURE = 0.3
MAX_FIX = 2            # 추출·새로 쓰기 각각 문제를 알려 주고 다시 요청하는 횟수
LEN_LO, LEN_HI = 0.85, 1.2   # 허용 길이(원문 글자 수 대비)
ORDER_SEED = 42        # 처리 순서(질문을 섞고 교사 셋을 번갈아). 중간에 멈춰도 세 교사가 비슷하게 진행된다
REPORT_EVERY = 100
SUMMARY_WORDS = ("요약하면", "정리하면", "결론적으로", "요컨대", "한마디로", "종합하면", "이처럼", "이와 같이", "이렇듯", "이렇게")

EXTRACT = """다음 질문과 답변을 읽고, 답변에 들어 있는 정보를 빠짐없이 한 줄에 하나씩 짧은 명제로 적으십시오.
규칙:
1. 사실, 주장, 이유, 예시, 조건, 방법, 조언, 주의사항을 모두 적습니다. 답변의 정보가 하나라도 빠지면 안 됩니다.
2. 숫자, 단위, 날짜, 비율, 고유명사(사람·지명·제품·학술 용어)는 원문 그대로 옮깁니다.
3. 원문의 문장 표현이나 말투는 따라 하지 말고 정보만 간결하게 적습니다.
4. 원문에 없는 정보를 더하지 않습니다.
5. 각 줄은 '- '로 시작하고, 목록 외의 말은 쓰지 않습니다.{num_rule}

[질문]
{q}

[답변]
{a}"""

WRITE = """다음 질문에 대한 답변을, 아래 정보 목록의 내용만 사용해 한국어로 작성하십시오.
작성 규칙:
1. 목록의 정보를 하나도 빠뜨리지 말고 모두 담습니다. 목록에 없는 정보, 예시, 수식어는 더하지 않습니다.
2. 숫자, 단위, 고유명사는 목록에 적힌 그대로 씁니다.
3. 굵게, 제목, 글머리 기호(-, *), 번호 목록(1.) 같은 서식을 쓰지 않고 문단 글로만 씁니다. 문단은 1~3개로 나눕니다.
4. 모든 문장을 '~습니다/~입니다' 체로 끝냅니다.
5. 괄호 안에 영어를 병기하지 않습니다.
6. '요약하면', '정리하면', '결론적으로', '이처럼', '이렇게' 같은 정리 문장을 쓰지 않고, 마지막 정보를 전달하는 문장으로 끝냅니다.
7. 분량은 {n}자 안팎(최소 {lo}자, 최대 {hi}자)으로 씁니다. 짧아지지 않도록 정보마다 한두 문장으로 충분히 설명합니다.
8. '정보 목록', '원문' 같은 말이나 머리말 없이 답변 본문만 출력합니다.

[질문]
{q}

[정보 목록]
{facts}"""

FIX = "\n\n(직전 결과의 문제: {p}. 이 문제를 고쳐 다시 작성하십시오.)"

client = OpenAI(base_url=BASE_URL, api_key=os.environ.get("NEUTRAL_API_KEY", ""), max_retries=0, timeout=120)
lock = threading.Lock()
stats = {"calls": 0, "n429": 0, "done": 0, "failed": 0}


def call(prompt, max_tokens):
    """한 번 호출. 429는 계속 기다리고, 5xx·연결 오류는 최대 5번 다시 보낸다."""
    other = 0
    while True:
        try:
            r = client.chat.completions.create(model=MODEL, max_tokens=max_tokens, temperature=TEMPERATURE,
                                               messages=[{"role": "user", "content": prompt}])
            with lock:
                stats["calls"] += 1
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            code = getattr(e, "status_code", None)
            if code == 429:
                with lock:
                    stats["n429"] += 1
                time.sleep(5)
            elif (code is None or code >= 500) and other < 5:
                other += 1
                time.sleep(10)
            else:
                raise RuntimeError(f"{type(e).__name__} {code or ''}") from None


def numbers(text):
    # 번호 목록 표시(1. 2.)는 내용 숫자가 아니다. "**1. 제목**", "### 1." 처럼 앞에 서식 기호가 붙은 경우도 뺀다
    text = re.sub(r"^[\s*_#>]*\d+[.)]\s", "", text, flags=re.M)
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def last_sentence(text):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return sents[-1] if sents else ""


def problems(text, orig):
    p, n = [], len(orig)
    if not text:
        return ["빈 답변입니다"]
    if len(text) < LEN_LO * n:
        p.append(f"분량이 {len(text)}자로 너무 짧습니다(최소 {n}자)")
    if len(text) > LEN_HI * n:
        p.append(f"분량이 {len(text)}자로 너무 깁니다(최대 {int(1.1 * n)}자)")
    if re.search(r"\*\*|__|^\s*#|^\s*[-*•]\s|^\s*\d+[.)]\s", text, re.M):
        p.append("굵게·제목·목록 서식을 썼습니다")
    if re.search(r"\([^)]*[A-Za-z][^)]*\)", text):
        p.append("괄호 안에 영어를 병기했습니다")
    if any(w in last_sentence(text) for w in SUMMARY_WORDS):
        p.append("마지막 문장을 정리 문장으로 끝냈습니다")
    miss = numbers(orig) - numbers(text)
    if miss:
        p.append(f"원문의 숫자 {sorted(miss)}가 빠졌습니다")
    if re.search(r"정보 목록|원문|^답변\s*:", text, re.M):
        p.append("'정보 목록'·'원문' 같은 메타 문구를 썼습니다")
    return p


def clean(text):
    """남은 서식 기호와 영문만 든 괄호를 지운다(숫자·한글이 든 괄호는 내용이므로 남긴다)."""
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)
    text = re.sub(r"\s*\((?=[^)]*[A-Za-z])[A-Za-z\s.,'’&/+:-]*\)", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def rewrite(row):
    q, o = row["instruction"], row["output"]
    nums = sorted(numbers(o))
    num_rule = f"\n6. 답변에 나오는 다음 숫자는 모두 목록에 그대로 들어가야 합니다: {', '.join(nums)}" if nums else ""
    base = EXTRACT.format(q=q, a=o, num_rule=num_rule)
    prompt, facts, ext_tries = base, "", 0
    for _ in range(MAX_FIX + 1):
        facts = call(prompt, 1024)
        ext_tries += 1
        lines = [l for l in facts.splitlines() if l.strip().startswith("-")]
        miss = set(nums) - numbers(facts)
        p = (["목록이 비었습니다"] if not lines else []) + ([f"목록에 숫자 {sorted(miss)}가 빠졌습니다"] if miss else [])
        if not p:
            break
        prompt = base + FIX.format(p="; ".join(p))

    base = WRITE.format(q=q, facts=facts, n=int(1.05 * len(o)), lo=len(o), hi=int(1.2 * len(o)))
    prompt, tries = base, []
    for _ in range(MAX_FIX + 1):
        text = call(prompt, 1536)
        p = problems(text, o)
        tries.append({"text": text, "problems": p})
        if not p:
            break
        prompt = base + FIX.format(p="; ".join(p))
    best = min(tries, key=lambda t: len(t["problems"]))
    final = clean(best["text"])
    return {"facts": facts, "extract_tries": ext_tries, "tries": tries, "final": final,
            "remaining": problems(final, o), "len_ratio": round(len(final) / len(o), 3)}


def load_raw(t):
    path = RW_DIR / f"raw_{t}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def run(limit):
    RW_DIR.mkdir(parents=True, exist_ok=True)
    src = {t: {x["id"]: x for x in json.loads((DATA_DIR / f"{t}_train_1000.json").read_text(encoding="utf-8"))} for t in TEACHERS}
    ids = sorted(src["gemini"])
    random.Random(ORDER_SEED).shuffle(ids)
    jobs = [(t, i) for i in ids for t in TEACHERS][:limit]
    done = {t: set(load_raw(t)) for t in TEACHERS}
    todo = [(t, i) for t, i in jobs if i not in done[t]]
    total = len(jobs)
    stats["done"] = total - len(todo)
    print(f"{time.strftime('%H:%M')} 모델 {MODEL}, 동시 요청 {WORKERS}개: 전체 {total}개 중 {len(todo)}개 재작성 "
          f"(이미 끝난 것 {stats['done']}개)", flush=True)
    t0, n0 = time.time(), stats["done"]

    def job(ti):
        t, i = ti
        row = src[t][i]
        try:
            res = rewrite(row)
        except RuntimeError as e:
            with lock:
                stats["failed"] += 1
                with (RW_DIR / "failures.log").open("a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {t} {i} {e}\n")
                print(f"{time.strftime('%H:%M')} [실패] {t} id {i}: {e}", flush=True)
            return
        rec = {"id": i, "teacher": t, "model": MODEL, **res}
        with lock:
            with (RW_DIR / f"raw_{t}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["done"] += 1
            d = stats["done"]
            if d % REPORT_EVERY == 0 or d == total:
                rate = (time.time() - t0) / max(d - n0, 1)
                per = {t: len(load_raw(t)) for t in TEACHERS}
                print(f"{time.strftime('%H:%M')} {d}/{total} 완료 (GPT {per['gpt']} · Claude {per['claude']} · Gemini {per['gemini']}), "
                      f"호출 {stats['calls']}회, 429 대기 {stats['n429']}번, 실패 {stats['failed']}개, "
                      f"남은 예상 {rate * (total - d) / 60:.0f}분", flush=True)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(job, todo))
    print(f"{time.strftime('%H:%M')} 이번 실행 끝: 완료 {stats['done']}/{total}, 실패 {stats['failed']}개 "
          f"(다시 실행하면 실패한 것만 재시도), {(time.time() - t0) / 60:.0f}분", flush=True)


def build():
    """raw 기록으로 학습 파일과 요약을 만든다. 1,000개가 다 있는 교사만 학습 파일을 쓴다."""
    summary = {"model": MODEL, "temperature": TEMPERATURE, "workers": WORKERS, "max_fix": MAX_FIX,
               "length_range": [LEN_LO, LEN_HI], "prompts": {"extract": EXTRACT, "write": WRITE, "fix": FIX}, "teachers": {}}
    for t in TEACHERS:
        src = json.loads((DATA_DIR / f"{t}_train_1000.json").read_text(encoding="utf-8"))
        raw = load_raw(t)
        recs = [raw[x["id"]] for x in src if x["id"] in raw]
        n = len(recs)
        if not n:
            continue
        kinds = {}
        for r in recs:
            for p in r["remaining"]:
                k = re.sub(r"\(.*|\[.*|\d+자.*", "", p).strip()
                kinds[k] = kinds.get(k, 0) + 1
        ratios = sorted(r["len_ratio"] for r in recs)
        summary["teachers"][t] = {
            "n_done": n, "n_total": len(src),
            "pass_first_try": sum(len(r["tries"]) == 1 and not r["tries"][0]["problems"] for r in recs) / n,
            "pass_final": sum(not r["remaining"] for r in recs) / n,
            "remaining_problem_counts": kinds,
            "len_ratio_mean": sum(ratios) / n, "len_ratio_median": ratios[n // 2],
            "len_ratio_in_range": sum(LEN_LO <= x <= LEN_HI for x in ratios) / n,
            "numbers_missing": sum(any("숫자" in p for p in r["remaining"]) for r in recs),
            "mean_write_tries": sum(len(r["tries"]) for r in recs) / n,
            "mean_extract_tries": sum(r["extract_tries"] for r in recs) / n,
        }
        if n == len(src):
            rows = [{"id": x["id"], "category": x["category"], "instruction": x["instruction"],
                     "output": raw[x["id"]]["final"], "source": t} for x in src]
            (DATA_DIR / f"{t}_rw_train_1000.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            shutil.copy(DATA_DIR / f"{t}_val.json", DATA_DIR / f"{t}_rw_val.json")
            print(f"저장: {t}_rw_train_1000.json, {t}_rw_val.json")
        else:
            print(f"[대기] {t}: {n}/{len(src)}개만 끝나 학습 파일은 아직 만들지 않음")
    (RW_DIR / "rewrite_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for t, s in summary["teachers"].items():
        print(f"{t}: {s['n_done']}/{s['n_total']}, 최종 검사 통과 {s['pass_final']:.0%}, 길이 비율 평균 {s['len_ratio_mean']:.2f}, "
              f"숫자 누락 {s['numbers_missing']}개, 남은 문제 {s['remaining_problem_counts']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="처리 순서의 앞 N개만 (시험)")
    parser.add_argument("--build", action="store_true", help="API 없이 학습 파일과 요약만 다시 만든다")
    cli = parser.parse_args()
    if not cli.build:
        if not os.environ.get("NEUTRAL_API_KEY"):
            raise SystemExit(".env에 NEUTRAL_API_KEY가 없습니다")
        run(cli.limit)
    build()


if __name__ == "__main__":
    main()
