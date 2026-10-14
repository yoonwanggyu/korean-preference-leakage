"""학습된 학생 모델들로 평가 질문 400개에 답을 생성해 JSON으로 저장한다.

사용법: python inference/gen_student.py              (sft/checkpoints/ 안의 모든 학생, 2차 설정이 기본값)
        python inference/gen_student.py --teacher gpt --seed 42
        python inference/gen_student.py --repetition-penalty 1.0 --max-new-tokens 590 --batch-size 16 \
            --out-dir inference/output/student                                                  (1차 설정 재현)
        이미 생성한 학생은 건너뛴다. 다시 생성하려면 --force.

학습 때와 같은 build_prompt()로 질문을 넣고, greedy 디코딩(항상 가장 확률이 높은 토큰)으로 답을 만든다.
그래서 같은 학생은 언제 돌려도 같은 답을 내고, 학생 간 차이는 학습된 모델의 차이만 반영한다.
<|im_end|> 또는 <|endoftext|>가 나오면 멈춘다. 최대 생성 길이는 모든 질문에 같다.
  기본(1차): 프롬프트+답변이 STUDENT_MAX_LEN(1024)을 넘지 않게 1024 - (가장 긴 프롬프트) = 590토큰, 반복 억제 없음
  2차: --repetition-penalty 1.1로 같은 문장 반복을 줄이고, --max-new-tokens 1024로 답변 길이 상한을 따로 준다.
  1차 결과에서 같은 문장 반복과 긴 답의 잘림이 많아(특히 Claude 학생 37.5%) 2차 설정을 추가했다.
  2차 생성은 Colab A100에서 학생 모델 15개를 불러와 이 스크립트와 같은 로직으로 실행했다(배치 64).

출력: inference/output_2nd/{teacher}_seed{seed}_eval400.json (평가에 사용하는 2차 결과)
      1차 결과는 2차와 비교한 뒤 삭제했다(통계는 description/sft_guide에 기록).
  [{id, category, instruction, reference, teacher, seed, output, n_new_tokens, finished}]
  finished=false는 최대 길이에 걸려 잘린 답이다.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
from prompt import END_TOKEN, build_prompt  # noqa: E402

load_dotenv(ROOT / ".env")

MAX_LEN = int(os.environ.get("STUDENT_MAX_LEN", 1024))
EVAL_PATH = ROOT / os.environ.get("DATA_DIR", "data") / "koalpaca_eval_400.json"
CKPT_DIR = Path(os.environ.get("SFT_CKPT_DIR", ROOT / "sft" / "checkpoints"))  # train.py와 같은 위치
OUT_DIR = Path(__file__).parent / "output_2nd"
BATCH_SIZE = 64          # 2차 설정 (1차는 16)
REPETITION_PENALTY = 1.1 # 2차 설정 (1차는 1.0)
MAX_NEW_TOKENS = 1024    # 2차 설정 (1차는 1024 - 가장 긴 프롬프트 = 590)


def generate_all(model, tok, questions, max_new, repetition_penalty, batch_size):
    stop_ids = [tok.convert_tokens_to_ids(END_TOKEN), tok.convert_tokens_to_ids("<|endoftext|>")]
    results = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        enc = tok([build_prompt(q["instruction"]) for q in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stop_ids,
                                 pad_token_id=tok.pad_token_id, repetition_penalty=repetition_penalty)
        for q, seq in zip(batch, out[:, enc["input_ids"].shape[1]:].tolist()):
            finished = any(t in stop_ids for t in seq)
            if finished:
                seq = seq[:min(seq.index(t) for t in stop_ids if t in seq)]
            results.append({**q, "output": tok.decode(seq, skip_special_tokens=True).strip(),
                            "n_new_tokens": len(seq), "finished": finished})
        print(f"  {min(start + batch_size, len(questions))}/{len(questions)}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", help="gpt | claude | gemini (코드 점검 때는 smoke)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, help="코드 점검·파일럿용: 앞의 N개 질문만 생성한다")
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY, help="1.0이면 반복 억제 없음")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="답변 최대 토큰")
    parser.add_argument("--tag", default="", help="출력 파일 이름 끝에 붙일 이름 (예: rp → ..._eval400_rp.json)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="저장 폴더")
    cli = parser.parse_args()

    eval_rows = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    questions = [{"id": x["id"], "category": x["category"], "instruction": x["instruction"],
                  "reference": x["answer"]} for x in eval_rows][:cli.limit]
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    students = sorted(p for p in CKPT_DIR.glob("*/seed*") if (p / "config.json").exists())
    students = [p for p in students if p.parent.name != "smoke" or cli.teacher == "smoke"]
    if cli.teacher:
        students = [p for p in students if p.parent.name == cli.teacher]
    if cli.seed is not None:
        students = [p for p in students if p.name == f"seed{cli.seed}"]
    print(f"학생 {len(students)}명: {[f'{p.parent.name}/{p.name}' for p in students]}")

    for path in students:
        teacher, seed = path.parent.name, int(path.name.removeprefix("seed"))
        out_path = out_dir / f"{teacher}_seed{seed}_eval{len(questions)}{'_' + cli.tag if cli.tag else ''}.json"
        if out_path.exists() and not cli.force:
            print(f"[건너뜀] {out_path.name}: 이미 생성됨")
            continue

        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = "left"   # 생성할 때는 왼쪽을 패딩해야 답이 프롬프트 바로 뒤에 이어진다
        # 모든 질문에 같은 최대 생성 길이: 가장 긴 프롬프트 + 답변이 MAX_LEN을 넘지 않게 한다.
        max_prompt = max(len(tok(build_prompt(q["instruction"]), add_special_tokens=False)["input_ids"]) for q in questions)
        max_new = cli.max_new_tokens or MAX_LEN - max_prompt
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()

        print(f"{teacher} seed {seed}: 최대 생성 {max_new}토큰, 반복 억제 {cli.repetition_penalty}, {device}/{dtype}")
        started = time.time()
        rows = generate_all(model, tok, questions, max_new, cli.repetition_penalty, cli.batch_size)
        for r in rows:
            r.update({"teacher": teacher, "seed": seed})
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {"teacher": teacher, "seed": seed, "decoding": "greedy", "max_new_tokens": max_new,
                "repetition_penalty": cli.repetition_penalty, "tag": cli.tag,
                "max_prompt_tokens": max_prompt, "batch_size": cli.batch_size, "dtype": str(dtype),
                "n_unfinished": sum(not r["finished"] for r in rows), "runtime_sec": round(time.time() - started, 1)}
        out_path.with_name(out_path.stem + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"저장: {out_path} (잘린 답 {meta['n_unfinished']}개, {meta['runtime_sec']}초)")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
