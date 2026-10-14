"""학습용 instruction 1,200개에 대해 Gemini로 답을 생성해 JSON으로 저장한다.

기존 키(id, category, instruction, answer)는 유지하고 gemini_output을 추가한다.
중간에 끊겨도 다시 실행하면 이미 생성된 항목은 건너뛰고, 실패한 항목(null)만 다시 생성한다.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompt import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.environ["GEMINI_MODEL"]
MAX_TOKENS = int(os.environ["GEN_MAX_TOKENS"])
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL") or None
THINKING_BUDGET = os.environ.get("GEMINI_THINKING_BUDGET") or None
TEMPERATURE = os.environ.get("GEN_TEMPERATURE") or None
SEND_TEMPERATURE = os.environ.get("GEMINI_SEND_TEMPERATURE") == "1"

IN_PATH = ROOT / os.environ.get("DATA_DIR", "data") / "koalpaca_instructions_1200.json"
OUT_PATH = Path(__file__).parent / "output" / "gemini_train_1200.json"
OUTPUT_KEY = "gemini_output"
MAX_WORKERS = 8
MAX_RETRIES = 5  # rate limit, 5xx는 SDK가 지수 백오프로 재시도한다
SAVE_EVERY = 20


def generate(client, instruction):
    # thinking_level이 있으면 우선 쓰고, 없을 때만 thinking_budget을 쓴다.
    if THINKING_LEVEL:
        thinking = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
    elif THINKING_BUDGET:
        thinking = types.ThinkingConfig(thinking_budget=int(THINKING_BUDGET))
    else:
        thinking = None
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_TOKENS,
        thinking_config=thinking,
        temperature=float(TEMPERATURE) if SEND_TEMPERATURE and TEMPERATURE else None,
    )

    resp = client.models.generate_content(model=MODEL, contents=instruction, config=config)
    # 출력 한도에 걸려 잘린 답이나 안전 필터로 막힌 답은 학습에 쓰지 않도록 실패로 처리한다.
    if not resp.candidates:
        raise RuntimeError(f"blocked: {resp.prompt_feedback}")
    reason = resp.candidates[0].finish_reason
    if reason != types.FinishReason.STOP:
        raise RuntimeError(f"incomplete: {reason}")
    text = (resp.text or "").strip()
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

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=MAX_RETRIES)),
    )
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(generate, client, x["instruction"]): x["id"] for x in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                results[i][OUTPUT_KEY] = fut.result()
            except Exception as e:
                # 에러 메시지에 요청 정보가 섞일 수 있어 종류와 상태 코드만 남긴다.
                print(f"[실패] id={i}: {type(e).__name__} {getattr(e, 'code', '')} {e if isinstance(e, RuntimeError) else ''}")
            if n % SAVE_EVERY == 0:
                save(results)
                print(f"{n}/{len(todo)} 진행")
    save(results)

    n_fail = sum(1 for x in results.values() if not x[OUTPUT_KEY])
    print(f"저장: {OUT_PATH} (실패 {n_fail}개, 다시 실행하면 실패한 것만 재시도)")


if __name__ == "__main__":
    main()
