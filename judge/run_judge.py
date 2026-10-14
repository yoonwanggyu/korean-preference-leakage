"""평가자 한 명으로 listwise 평가 항목을 채점한다: 같은 질문·같은 seed의 학생 3명 답을 한 번에 보고 기준별 1~3위.

평가 항목 = 묶음(질문 100 × seed 5 = 500) × 제시 순서(N_ROTATIONS, 기본 3) = 1,500개. 순서는 select_items.py가 정한다.

평가자 4명 (본 분석은 gpt·claude·gemini 3명의 채점으로 했고, hyperclova는 코드만 준비해 두고 아직 실행하지 않았다)
  gpt / claude / gemini : 교사와 같은 모델(.env의 OPENAI_MODEL, ANTHROPIC_MODEL, GEMINI_MODEL).
                          같은 계열 선호(선호 누출)를 보려면 교사와 평가자가 같은 모델이어야 한다.
  hyperclova            : 중립 평가자. 세 계열과 무관한 NAVER HyperCLOVA X(HCX-007)를 CLOVA Studio의
                          OpenAI 호환 API로 부른다(.env의 NEUTRAL_API_KEY, NEUTRAL_MODEL).
추론 강도는 네 평가자 모두 JUDGE_EFFORT(기본 medium), 출력 한도는 모두 JUDGE_MAX_TOKENS(기본 12,288, 사고 토큰 포함)로
맞추고, temperature는 보내지 않는다. 출력 한도는 HCX-007이 medium 추론에 기본 10,240토큰을 쓰는 것에 맞춰 정했다.
한도는 상한일 뿐 추론 깊이는 effort가 정하므로, 같은 값으로 두어도 다른 평가자의 채점에는 영향이 없다.
프롬프트는 네 평가자 모두 judge/test_prompt.py의 같은 문장을 쓴다.

사용법
  python judge/run_judge.py --judge gpt --dry-run          API를 부르지 않고 첫 항목의 프롬프트만 출력
  python judge/run_judge.py --judge hyperclova --limit 20  파일럿: 앞의 20개만 채점
  python judge/run_judge.py --judge claude                 전체 채점 (끊기면 다시 실행하면 이어서 진행)
  python judge/run_judge.py --judge gpt --pilot            파일럿: seed 42 묶음 10개 × 순서 3개 = 30개 (judge/output/pilot/에 따로 저장)

입력: judge/data/eval_items.json (select_items.py가 만든다)
출력: judge/output/raw/{judge}.jsonl  한 줄에 한 항목 (채점 원본 기록, 이어하기에 쓴다)
  {item_id, group_id, rotation, order, judge, judge_model, question_id, seed,
   factuality: {reason, ranking: ["B","A","C"], ranks: {"gpt": 2, "claude": 1, "gemini": 3}}, ... 4개 기준 + overall}
  ranking은 평가자가 쓴 라벨(위치) 순위이고, ranks는 이를 교사별 순위로 풀어 쓴 것이다(order로 변환).
  채점이 끝나면 judge/merge_results.py가 학생 응답 JSON에 평가 결과를 붙여 judge/output/judged/에 저장한다.
형식이 틀린 응답은 최대 MAX_PARSE_TRIES번 다시 요청하고, 그래도 실패하면 기록하지 않는다(다시 실행하면 재시도).
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from test_prompt import JUDGE_SYSTEM, LABELS, OUTPUT_KEYS, build_user_prompt, parse_judgement

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

JUDGES = ["gpt", "claude", "gemini", "hyperclova"]
ITEMS_PATH = Path(__file__).parent / "data" / "eval_items.json"
RAW_DIR = Path(__file__).parent / "output" / "raw"
PILOT_DIR = Path(__file__).parent / "output" / "pilot"
PILOT_SEED, PILOT_N_QUESTIONS = 42, 10   # 파일럿 범위: seed 42 묶음 중 선정 질문 앞 10개 × 순서 3개
EFFORT = os.environ.get("JUDGE_EFFORT", "medium")                       # 네 평가자 공통 추론 강도
MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", 12288))            # 네 평가자 공통 출력 한도(사고 토큰 포함)
HCX_BASE_URL = os.environ.get("NEUTRAL_BASE_URL", "https://clovastudio.stream.ntruss.com/v1/openai/")
MAX_WORKERS = {"gpt": 8, "claude": 8, "gemini": 8, "hyperclova": 1}   # 동시 요청 수. CLOVA Studio 테스트 키는 호출 한도가 낮아(동시 8개에서 429) 1개씩 보낸다
MAX_RETRIES = 5        # rate limit, 5xx는 SDK가 지수 백오프로 재시도
MAX_PARSE_TRIES = 3    # JSON 형식이 틀리면 다시 요청하는 횟수


def make_client(judge):
    """평가자별 (모델 이름, 호출 함수)를 돌려준다. 호출 함수는 (system, user) → 응답 텍스트."""
    if judge == "gpt":
        from openai import OpenAI
        client, model = OpenAI(max_retries=MAX_RETRIES), os.environ["OPENAI_MODEL"]

        def call(system, user):
            resp = client.responses.create(model=model, instructions=system, input=user,
                                           max_output_tokens=MAX_TOKENS, reasoning={"effort": EFFORT})
            if resp.status != "completed":
                raise RuntimeError(f"incomplete: {resp.incomplete_details}")
            return resp.output_text
        return model, call

    if judge == "claude":
        from anthropic import Anthropic
        client, model = Anthropic(max_retries=MAX_RETRIES), os.environ["ANTHROPIC_MODEL"]

        def call(system, user):
            # 거절 시 다른 모델로 넘기는 fallback은 쓰지 않는다(평가자 정체성이 섞이지 않게).
            resp = client.messages.create(model=model, system=system, max_tokens=MAX_TOKENS,
                                          messages=[{"role": "user", "content": user}],
                                          thinking={"type": "adaptive"}, output_config={"effort": EFFORT})
            if resp.stop_reason != "end_turn":
                raise RuntimeError(f"incomplete: {resp.stop_reason}")
            return "".join(b.text for b in resp.content if b.type == "text")
        return model, call

    if judge == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=MAX_RETRIES)))
        model = os.environ["GEMINI_MODEL"]

        def call(system, user):
            config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=MAX_TOKENS,
                                                 thinking_config=types.ThinkingConfig(thinking_level=EFFORT))
            resp = client.models.generate_content(model=model, contents=user, config=config)
            if not resp.candidates or resp.candidates[0].finish_reason != types.FinishReason.STOP:
                raise RuntimeError(f"incomplete: {resp.candidates[0].finish_reason if resp.candidates else 'blocked'}")
            return resp.text
        return model, call

    if judge == "hyperclova":
        from openai import OpenAI
        # HCX-007(추론 모델)은 max_tokens를 거부하고 max_completion_tokens만 받는다(실제 호출로 확인).
        client = OpenAI(base_url=HCX_BASE_URL, api_key=os.environ["NEUTRAL_API_KEY"], max_retries=MAX_RETRIES)
        model = os.environ.get("NEUTRAL_MODEL", "HCX-007")

        def call(system, user):
            resp = client.chat.completions.create(model=model, max_completion_tokens=MAX_TOKENS, reasoning_effort=EFFORT,
                                                  messages=[{"role": "system", "content": system},
                                                            {"role": "user", "content": user}])
            if resp.choices[0].finish_reason != "stop":
                raise RuntimeError(f"incomplete: {resp.choices[0].finish_reason}")
            return resp.choices[0].message.content
        return model, call

    raise ValueError(judge)


def judge_one(call, item):
    base = build_user_prompt(item["instruction"], item["answers"])
    user, last_error = base, None
    for _ in range(MAX_PARSE_TRIES):
        text = call(JUDGE_SYSTEM, user)
        try:
            return parse_judgement(text)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            # Gemini가 가끔 overall에 근거만 쓰고 ranking을 빠뜨리며, 같은 프롬프트로는 같은 항목에서 반복된다.
            # 재요청에만 형식 오류를 알려 주는 문장을 덧붙인다(형식 오류가 난 항목에만, 모든 평가자에게 같게 적용).
            user = base + (f"\n\n(주의: 직전 응답에 형식 오류가 있었습니다: {e}. "
                           "다섯 항목 모두 reason과 ranking을 포함한 JSON 전체를 다시 출력하십시오.)")
    raise RuntimeError(f"형식 오류 {MAX_PARSE_TRIES}회: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True, choices=JUDGES)
    parser.add_argument("--limit", type=int, help="파일럿: 앞의 N개 항목만 채점")
    parser.add_argument("--dry-run", action="store_true", help="API를 부르지 않고 첫 항목의 프롬프트만 출력")
    parser.add_argument("--pilot", action="store_true", help="seed 42 묶음 10개 × 순서 3개만 채점해 pilot 폴더에 저장")
    parser.add_argument("--workers", type=int, help="동시 요청 수 (기본: 평가자별 MAX_WORKERS)")
    cli = parser.parse_args()

    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    raw_dir = RAW_DIR
    if cli.pilot:
        qids = sorted({it["question_id"] for it in items})[:PILOT_N_QUESTIONS]
        items = [it for it in items if it["seed"] == PILOT_SEED and it["question_id"] in qids]
        raw_dir = PILOT_DIR / "raw"
    items = items[:cli.limit]
    if cli.dry_run:
        print("===== system =====\n" + JUDGE_SYSTEM)
        print("\n===== user =====\n" + build_user_prompt(items[0]["instruction"], items[0]["answers"]))
        print(f"\n(A·B·C = {items[0]['order']} 학생, seed {items[0]['seed']} — 평가자에게는 보이지 않음)")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{cli.judge}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["item_id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line}
    todo = [it for it in items if it["item_id"] not in done]
    model, call = make_client(cli.judge)
    print(f"평가자 {cli.judge} ({model}, effort {EFFORT}, 동시 {cli.workers or MAX_WORKERS[cli.judge]}개): "
          f"전체 {len(items)}개 중 {len(todo)}개 채점")

    n_fail = 0
    workers = cli.workers or MAX_WORKERS[cli.judge]
    with ThreadPoolExecutor(workers) as pool, out_path.open("a", encoding="utf-8") as f:
        futures = {pool.submit(judge_one, call, it): it for it in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            it = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                # 에러 메시지에 요청 정보가 섞일 수 있어 종류와 상태 코드만 남긴다.
                n_fail += 1
                print(f"[실패] item {it['item_id']}: {type(e).__name__} {getattr(e, 'status_code', '')} "
                      f"{e if isinstance(e, RuntimeError) else ''}")
                continue
            row = {"item_id": it["item_id"], "group_id": it["group_id"], "rotation": it["rotation"],
                   "order": it["order"], "judge": cli.judge, "judge_model": model,
                   "question_id": it["question_id"], "seed": it["seed"]}
            for k in OUTPUT_KEYS:
                ranking = result[k]["ranking"]
                # 라벨(위치) 순위를 교사별 순위로 바꾼다: order[i]가 라벨 LABELS[i]에 놓인 교사
                ranks = {it["order"][LABELS.index(lab)]: rank for rank, lab in enumerate(ranking, 1)}
                row[k] = {"reason": result[k]["reason"], "ranking": ranking, "ranks": ranks}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()   # 한 항목마다 바로 기록해 끊겨도 잃지 않는다
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} 진행", flush=True)
    print(f"저장: {out_path} (이번 실행 실패 {n_fail}개, 다시 실행하면 실패한 것만 재시도)")


if __name__ == "__main__":
    main()
