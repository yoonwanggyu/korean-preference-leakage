# 프로젝트 개요
한국어 합성데이터 증류에서 LLM 평가자의 preference leakage(자기 계열 학생 우대)를 추정한다. KoAlpaca 질문에 GPT·Claude·Gemini가 답을 생성하고(분류 실험으로 교사 스타일이 구별됨을 확인), 각 교사 답 1,000개로 Qwen3-0.6B-Base 학생 15명(교사 3 × seed 42~46)을 SFT했다(교차 loss로 스타일 흡수 확인).
같은 seed 학생 3명의 답을 세 교사 모델이 listwise로 순위 평가하고(질문 100 × seed 5, 순서 3가지, FLASK 기준 4개 + 최종 순위), 순위 로짓 모형(주)·1위 모형·쌍 고정효과 LPM으로 학생 품질과 위치 편향을 통제한 누출 λ를 추정한다(주 지표는 쌍대 승률, 부트스트랩·Holm 보정).
결과: Gemini는 승률 +9.8%p로 뚜렷한 누출(seed 5/5, 길이 통제 후에도 유지), GPT는 약한 증거, Claude는 없음, 모든 평가자에 A 위치 편향이 있다. 중립 평가자(HyperCLOVA)는 교수 피드백 후 추가한다.

# 코드 스타일
- 단순한 함수와 스크립트로 작성한다.
- 불필요한 class로 함수를 묶지 않는다.
- 한 번만 쓰는 로직을 잘게 쪼개지 않는다.
- 설정은 .env와 상단 상수로 관리한다.

# 아키텍처
실행 순서: data → inference(교사 답) → classify / sft → inference(학생 답) → judge → results → description(포스터는 직접 작성)

- data/: 질문 준비
  - load_koalpaca.py: KoAlpaca-v1.1a에서 학습용 instruction 1,200개 → koalpaca_instructions_1200.json
  - split_eval.py: 학습용과 겹치지 않는 평가 질문 400개 → koalpaca_eval_400.json
- inference/: 답변 생성
  - prompt.py: 교사 생성·학생 학습·추론 공통 프롬프트
  - gen_gpt.py, gen_claude.py, gen_gemini.py: 교사 답 생성 → output/{teacher}_train_1200.json
  - gen_student.py: 학생 15명의 평가 질문 400개 답 → output_2nd/{teacher}_seed{n}_eval400.json(+ .meta.json)
- classify/: 교사 답만으로 생성 모델을 맞히는 분류 실험(raw·norm 두 조건)
  - common.py: 질문 단위 train/val/test 분할, 정규화, 그림 함수
  - logistic.py: 문자 n-gram TF-IDF 로지스틱 회귀 → output/logistic/
  - roberta.py: KLUE-RoBERTa 미세조정 → output/roberta/, checkpoints/(커밋 안 함)
  - compare.py: 두 분류기 비교 → output/compare_*.png, compare.md
- sft/: 학생 학습(Qwen3-0.6B-Base, 교사 3 × seed 42~46)
  - build_data.py: 세 교사 공통 학습 1,000개와 검증 199개 → data/
  - train.py: 교사별 full SFT → checkpoints/(커밋 안 함), runs/{teacher}_seed{n}/(loss 기록)
  - cross_loss.py: 학생 × 교사 검증 답변의 교차 loss·perplexity → runs/cross_loss.csv
  - models/: 원본 모델(커밋 안 함)
- judge/: LLM-as-a-Judge(listwise, FLASK 기준 4개 + 최종 순위, 순서 3가지)
  - select_items.py: 평가 질문 100개 × seed 5 묶음과 제시 순서 → data/eval_items.json, selection.json
  - test_prompt.py: 평가 프롬프트·기준·응답 파싱
  - run_judge.py: 평가자별 채점 → output/raw/{judge}.jsonl, output/logs/ (pilot/은 파일럿 결과)
  - merge_results.py: 채점을 학생 응답에 붙임 → output/judged/, output/ranks_long.csv
  - descriptive.py: 1위 비율·쌍대 승률·단순 누출·위치 편향 → output/analysis/desc_*.csv
  - analyze_models.py: 순위 로짓(PL, 주 모형)·1위 모형·쌍 고정효과 LPM, 부트스트랩, Holm, 길이 통제 → output/analysis/model_*.csv, models_summary.json
- results/: 요약 표와 그림
  - style_analysis.py: 교사 답의 길이·형식·문체 비교 → output/style_*, fig_style_*.png
  - sft_curves.py: SFT loss 곡선과 교차 loss 그림 → output/sft_*.png, sft_summary.json
- description/: 연구 기록 문서 하나로 통합
  - poster.html: 모든 실험·설계 근거·결과·실제 LLM 입출력 예시·예상 질문을 담은 단일 문서(목차 링크 포함). 직접 편집해 관리한다(생성 스크립트 없음)
  - poster.pdf: poster.html을 Chrome으로 인쇄한 PDF
- 신청서_SampleTex/: 학술대회 신청서(hwp)와 초록(git 제외)
  - abstract/tex/KoreanAbstract_preference_leakage.tex(+ 스타일 파일), abstract/pdf/: 컴파일한 초록 PDF
- 루트: requirements.txt(.venv 라이브러리), .env(API 키, 커밋 안 함)

# 중요사항
- venv 만들어서 지금 프로젝트 코드를 돌릴 수 있는 환경 만들어서 거기서만 코드를 돌려줘.
- requirements.txt 파일에 필요한 라이브러리 추가해줘.
- .env는 절대 커밋하지 않는다.
- API 키를 코드와 로그에 남기지 않는다.
- 무조건 사용자의 요청 해당하는 폴더와 파일들만 들어가서 확인하고 그 이외에 폴더와 파일은 건드리지 않는다.