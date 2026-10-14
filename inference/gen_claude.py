"""학습용 instruction 1,200개에 대해 Claude로 답을 생성해 JSON으로 저장한다.

기존 키(id, category, instruction, answer)는 유지하고 claude_output을 추가한다.
중간에 끊겨도 다시 실행하면 이미 생성된 항목은 건너뛰고, 실패한 항목(null)만 다시 생성한다.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from prompt import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.environ["ANTHROPIC_MODEL"]
MAX_TOKENS = int(os.environ["GEN_MAX_TOKENS"])
THINKING = os.environ.get("ANTHROPIC_THINKING") or None
EFFORT = os.environ.get("ANTHROPIC_EFFORT") or None
TEMPERATURE = os.environ.get("GEN_TEMPERATURE") or None
SEND_TEMPERATURE = os.environ.get("ANTHROPIC_SEND_TEMPERATURE") == "1"

IN_PATH = ROOT / os.environ.get("DATA_DIR", "data") / "koalpaca_instructions_1200.json"
OUT_PATH = Path(__file__).parent / "output" / "claude_train_1200.json"
OUTPUT_KEY = "claude_output"
MAX_WORKERS = 8
MAX_RETRIES = 5  # rate limit, 5xx, 타임아웃은 SDK가 지수 백오프로 재시도한다
SAVE_EVERY = 20


def generate(client, instruction):
    kwargs = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": instruction}],
        "max_tokens": MAX_TOKENS,
    }
    if THINKING:
        kwargs["thinking"] = {"type": THINKING}
    if EFFORT:
        kwargs["output_config"] = {"effort": EFFORT}
    if SEND_TEMPERATURE and TEMPERATURE:
        kwargs["extra_body"] = {"temperature": float(TEMPERATURE)}

    # 생성자 정체성이 섞이지 않도록 거절 시 다른 모델로 넘기는 fallback은 쓰지 않는다.
    resp = client.messages.create(**kwargs)
    # 출력 한도에 걸려 잘린 답이나 거절된 답은 학습에 쓰지 않도록 실패로 처리한다.
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"incomplete: {resp.stop_reason}")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise RuntimeError("empty output")
    return text


def save(results):
    OUT_PATH.write_text(json.dumps(list(results.values()), ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    items = json.loads(IN_PATH.read_text(encoding="utf-8"))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if OUT_PATH.exists():
        done = {x["id"]: x[OUTPUT_KEY] for x in json.loads(OUT_PATH.read_text(encoding="utf-8")) if x.get(OUTPUT_KEY)}
    results = {x["id"]: {**x, OUTPUT_KEY: done.get(x["id"])} for x in items}
    todo = [x for x in items if not done.get(x["id"])]
    print(f"모델: {MODEL}, 전체 {len(items)}개 중 {len(todo)}개 생성")

    client = Anthropic(max_retries=MAX_RETRIES)  # ANTHROPIC_API_KEY는 환경변수에서 읽는다
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(generate, client, x["instruction"]): x["id"] for x in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                results[i][OUTPUT_KEY] = fut.result()
            except Exception as e:
                # 에러 메시지에 요청 정보가 섞일 수 있어 종류와 상태 코드만 남긴다.
                print(f"[실패] id={i}: {type(e).__name__} {getattr(e, 'status_code', '')} {e if isinstance(e, RuntimeError) else ''}")
            if n % SAVE_EVERY == 0:
                save(results)
                print(f"{n}/{len(todo)} 진행")
    save(results)

    n_fail = sum(1 for x in results.values() if not x[OUTPUT_KEY])
    print(f"저장: {OUT_PATH} (실패 {n_fail}개, 다시 실행하면 실패한 것만 재시도)")


if __name__ == "__main__":
    main()
