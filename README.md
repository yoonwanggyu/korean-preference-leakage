# 한국어 합성데이터 증류에서의 LLM-as-a-Judge 선호 누출

> Preference Leakage in LLM-as-a-Judge for Korean Synthetic-Data Distillation

## Abstract

LLM이 생성한 합성데이터로 학습한 모델을 다시 LLM이 채점하는 LLM-as-a-Judge 평가에서는 채점 LLM이 자기 데이터로 학습한 모델을 우대하는 선호 누출(preference leakage)이 보고되었으나, 관련 연구는 주로 영어로 이루어져 한국어 데이터에서의 양상은 충분히 다뤄지지 않았다. 본 연구는 GPT, Claude, Gemini의 한국어 답변으로 각각 학습한 학생 모델 15개를 같은 세 모델이 평가하도록 하였다. 평가자는 한 질문에 대한 세 학생의 답변을 FLASK 기반 네 기준(사실 정확성, 질문 충족도, 논리성, 가독성)과 이를 종합한 최종 순위에서 각각 1~3위로 매겼다(listwise 평가 4,500회). 이를 세 평가자가 공통으로 인정한 학생별 답변 품질과 답변이 프롬프트에 놓인 순서를 통제하며 평가자별 자기 계열 우대를 동시에 추정하는 순위 로짓 모형으로 분석하였다. 그 결과 최종 순위에서 Gemini 평가자는 품질이 같은 상대에 대해 자기 계열 학생의 승률을 9.8%p(95% 신뢰구간 7.1~12.5) 높게 판정하였고, 이는 5회 반복 학습과 답변의 생성 토큰 수를 통제한 뒤에도 유지되었다. 예를 들어 세 학생이 모두 핵심인 삼투압 조절을 설명하지 못한 '민물고기는 왜 바다에서 살 수 없는가' 질문에서 GPT와 Claude 평가자는 제시 순서를 바꾼 세 번 모두 GPT 학생을 1위로 판정하였으나, Gemini 평가자만 세 번 모두 Gemini 학생을 1위로 판정하였다. 반면 GPT 학생만 핵심 기전(IgE 면역 반응)을 맞힌 메밀 알레르기 질문에서는 세 평가자가 아홉 번 모두 GPT 학생을 1위로 판정하여, 평가자들이 내용의 우열은 공통으로 가려내며 자기 계열 우대는 우열이 분명하지 않은 답변 사이에서 나타날 수 있음을 시사하였다. 자기 계열 평가자의 승률에서 다른 두 평가자의 평균을 뺀 단순 비교에서는 GPT(4.3%p)와 Claude(1.8%p)도 선호를 보였으나, 이는 Gemini 평가자가 자기 학생을 올리며 다른 학생의 승률을 함께 낮춘 몫이 기준선에 섞인 결과로, 이를 분리한 모형에서는 유의하지 않았다. 이는 한국어 데이터로 학습한 모델의 평가에서 평가자 선택만으로 학생 모델 간 성능 순위가 바뀔 수 있음을 보여 주며, 생성 모델과 다른 계열의 평가자를 쓰고 답변 순서를 바꿔 가며 반복 평가하는 설계가 필요함을 시사한다.

**주요용어:** 선호 누출(preference leakage), LLM-as-a-Judge, 합성데이터, 순위 로짓 모형

---

## 1. Introduction

작은 모델은 큰 모델(교사)이 만든 합성데이터로 학습하고, 그 성능도 다시 큰 모델에게 채점을 맡기는 일이 흔하다(LLM-as-a-Judge). 이때 데이터를 만든 모델과 채점하는 모델이 같은 계열이면, 평가자가 자기 문체를 물려받은 학생을 품질과 무관하게 더 좋게 보는 **선호 누출**이 생길 수 있다(Li et al., 2026). 이 현상은 영어 벤치마크에서 보고되었지만, 한국어 합성데이터에서의 양상은 확인되지 않았다.

본 연구는 한국어 합성데이터를 생성한 모델이 그 데이터로 학습한 학생 모델을 다시 평가하는 환경에서 선호 누출을 추정하고, 이를 완화할 수 있는 방법을 탐구한다.

- **RQ1 (선호 누출의 존재)** : 한국어 합성데이터를 생성한 LLM은 해당 데이터로 학습한 학생 모델을 평가할 때, 답변 품질과 제시 위치를 통제한 후에도 우대하는가?
- **RQ2 (선호 누출의 이질성)** : 선호 누출의 크기는 평가자와 평가 기준에 따라 어떻게 달라지는가?
- **RQ3 (선호 누출의 강건성)** : 관측된 선호 누출은 학습 무작위성과 답변 길이를 고려해도 유지되는가?

## 2. Related Work

| 주제 | 참고 논문 |
|---|---|
| 선호 누출 | Li et al. (2026). *Preference Leakage: A Contamination Problem in LLM-as-a-judge.* ICLR 2026 |
| LLM-as-a-Judge | Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 |
| 자기 선호 | Panickssery et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS 2024 |
| 위치 편향 | Wang et al. (2024). *Large Language Models are not Fair Evaluators.* ACL 2024 · Shi et al. (2025). *Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge* |
| 평가 기준 | Ye et al. (2024). *FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets.* ICLR 2024 |
| 순위 모형 | Luce (1959); Plackett (1975); Bradley & Terry (1952); McFadden (1974); Beggs et al. (1981) |
| 통계 추론 | Efron (1979) 부트스트랩; Holm (1979) 다중 비교 보정; Cameron & Miller (2015) 군집 강건 추론 |

전체 참고문헌은 [description/poster.html](description/poster.html)의 참고문헌 절에 있다.

## 3. Experimental Setup

**① 교사 답변 생성.** KoAlpaca-v1.1a 질문 1,200개에 세 교사(gpt-5.6-terra, claude-sonnet-5-5, gemini-3.8-flash)가 같은 시스템 프롬프트로 답했다. 서식은 지정하지 않아 모델 고유의 스타일이 남게 했다.

<p align="center"><img src="description/figures/01_%EA%B7%B8%EB%A6%BC3-1_3%20%EC%8B%A4%EC%A0%9C%EB%A1%9C%20%EB%B3%B4%EB%82%B8%20%EC%9A%94%EC%B2%AD%EA%B3%BC%20%EB%B0%9B%EC%9D%80%20%EB%8B%B5%20%28%EA%B5%90%EC%82%AC%20%EC%83%9D%EC%84%B1%20%ED%95%9C%20%EA%B1%B4%29.png" width="100%"></p>
<p align="center"><sub>그림 1. 같은 요청에 대한 세 교사의 실제 답. 길이와 형식(굵은 소제목, 목록, 예문)이 모델마다 다르다.</sub></p>

**② 교사 스타일 구별 확인.** 답변만 보고 어느 교사가 썼는지 맞히는 분류기는 97% 정확도를 보였고, 서식을 지운 텍스트에서도 92~97%였다. 평가자가 자기 계열 문체를 알아볼 수 있는 조건이다.

<p align="center"><img src="description/figures/05_%EA%B7%B8%EB%A6%BC4-7_4.8%20%EA%B2%B0%EA%B3%BC%20%ED%95%B4%EC%84%9D.png" width="100%"></p>
<p align="center"><sub>그림 2. ① 교사별 형식 특징 ② 답변만으로 교사를 맞히는 분류 정확도(로지스틱 회귀, KLUE-RoBERTa).</sub></p>

**③ 학생 학습(SFT).** 각 교사의 답 1,000개로 Qwen3-0.6B-Base를 full fine-tuning했다(교사 3 × seed 42~46 = 학생 15명, lr 1e-5, 3 epoch, 실효 배치 16). 학생 15명 모두 자기 교사의 검증 답에서 loss가 가장 낮아, 교사의 문체를 흡수했다(15/15).

<p align="center"><img src="description/figures/13_%EA%B7%B8%EB%A6%BC5-C_5.4%20%ED%95%99%EC%83%9D%EC%9D%B4%20%EC%9E%90%EA%B8%B0%20%EA%B5%90%EC%82%AC%EC%9D%98%20%EC%8A%A4%ED%83%80%EC%9D%BC%EC%9D%84%20%ED%9D%A1%EC%88%98%ED%96%88%EB%8A%94%EA%B0%80%20-%20%EA%B5%90%EC%B0%A8%20loss.png" width="100%"></p>
<p align="center"><sub>그림 3. ① 학습 곡선 ② 학생 × 교사 검증 답의 교차 loss. 대각선(★)이 가장 낮다.</sub></p>

**④ LLM-as-a-Judge.** 같은 seed의 세 학생 답을 이름 없이 A·B·C로 보여 주고, 세 교사 모델이 FLASK 기반 네 기준(사실 정확성, 질문 충족도, 논리성, 가독성)과 최종 순위로 1~3위를 매겼다. 같은 세 답을 순서만 바꿔 3번 평가해 모든 학생이 각 위치에 한 번씩 놓이게 했다(질문 100 × seed 5 × 순서 3 × 평가자 3 = 4,500회).

<p align="center"><img src="description/figures/16_%EA%B7%B8%EB%A6%BC6-D_6.1.0%20%ED%8F%89%EA%B0%80%20%EC%84%A4%EA%B3%84%EB%A5%BC%20%EC%9D%B4%EB%A0%87%EA%B2%8C%20%EC%A0%95%ED%95%9C%20%EC%9D%B4%EC%9C%A0.png" width="100%"></p>
<p align="center"><sub>그림 4. 평가 기준과 listwise 평가 방식(순서 순환).</sub></p>

## 4. Statistical Model

평가자의 순위를 **순위 로짓 모형**(Plackett–Luce)으로 분석했다. 평가자 $j$가 학생 $t$의 답에 주는 점수를

$$u_{jt} = \theta_{s,t} + \delta_{j,\text{위치}} + \lambda_j \cdot \mathbb{1}[t = j]$$

로 두고, 1위 선택 → 남은 둘 중 2위 선택의 확률로 순위를 설명한다.

- $\theta_{s,t}$ : 학생 품질(세 평가자가 공통으로 인정한 몫)
- $\delta_{j,\text{위치}}$ : 평가자별 위치 편향
- $\lambda_j$ : **선호 누출**. 품질과 위치가 같은 상대를 이길 확률의 증가 $\sigma(\lambda_j) - 0.5$ (%p)로 읽는다.

쌍마다 두 학생 모두와 계열이 다른 평가자가 있어 세 평가자의 $\lambda$를 동시에 식별할 수 있다. 신뢰구간은 질문 단위 부트스트랩 1,000회, 세 평가자 검정은 Holm 보정으로 판단했다. 강건성 확인으로 1위 모형(1위만 사용)과 쌍 고정효과 선형확률모형(같은 두 답끼리 비교)을 함께 적합했다.

<p align="center"><img src="description/figures/21_%EA%B7%B8%EB%A6%BC6-H_6.2%285%29%20%EC%84%B8%20%EA%B0%80%EC%A7%80%20%EB%AA%A8%ED%98%95%20-%20%EB%AC%B4%EC%97%87%EC%9D%B4%EA%B3%A0%2C%20%EC%9B%90%EB%9E%98%20%EC%9E%88%EB%8A%94%20%EB%AA%A8%ED%98%95%EC%9D%B8%EA%B0%80.png" width="100%"></p>
<p align="center"><sub>그림 5. 단순 계산에 섞이는 네 가지 함정(학생 품질, 위치 편향, 다른 평가자의 편애, 우연)과 이를 통제하는 방법.</sub></p>

## 5. Results

**RQ1.** Gemini 평가자는 품질과 위치를 통제한 뒤에도 자기 계열 학생의 승률을 **+9.8%p**(95% CI 7.1~12.5, Holm p ≤ 0.003) 높게 판정했다. 품질이 같다면 이길 확률이 50%가 아니라 약 60%라는 뜻이다. GPT 평가자는 약한 증거(+2.8%p, 보정 전에만 유의), Claude 평가자는 누출이 없었다(−2.0%p).

**RQ3.** Gemini 평가자의 누출은 독립 학습 5회 모두에서 양수였고(5/5), 답변 길이를 통제해도 +8.8%p로 유지되었으며, 세 통계 모형에서도 +8.6~+10.7%p로 일관되었다.

<p align="center"><img src="description/figures/29_%EA%B7%B8%EB%A6%BC6-I_6.5%20%EA%B2%B0%EA%B3%BC%20%E2%91%A2%20%EB%8B%A4%EC%84%AF%20%EB%B2%88%EC%9D%98%20%EB%8F%85%EB%A6%BD%20%ED%95%99%EC%8A%B5%28seed%29%EC%97%90%EC%84%9C%EB%8F%84%20%EB%B0%98%EB%B3%B5%EB%90%98%EB%8A%94%EA%B0%80.png" width="100%"></p>
<p align="center"><sub>그림 6. ① 세 모형으로 추정한 평가자별 선호 누출 ② seed별 추정치. * Holm 보정 후 유의, † 보정 전에만 유의.</sub></p>

**RQ2.** Gemini 평가자의 누출은 **사실 정확성**(+10.9%p)에서 가장 컸고, 가독성(+6.9), 질문 충족도(+4.3), 논리성(+3.6%p) 순이었다. 문체뿐 아니라 내용 판단에서도 우대가 나타났다. GPT·Claude 평가자는 모든 기준에서 보정 후 유의하지 않았다.

<p align="center"><img src="description/figures/30_%EA%B7%B8%EB%A6%BC6-4_6.6%20%EA%B2%B0%EA%B3%BC%20%E2%91%A3%20%ED%8F%89%EA%B0%80%20%EA%B8%B0%EC%A4%80%EB%B3%84%20%EB%B6%84%EC%84%9D%20%28ablation%29.png" width="100%"></p>
<p align="center"><sub>그림 7. 평가 기준별 선호 누출(순위 로짓 모형, 95% 신뢰구간).</sub></p>

**위치 편향.** 세 평가자 모두 첫 번째(A) 답을 1위로 고르는 경향이 있었다(GPT 49.6%, Claude 45.6%, Gemini 42.3%; 편향이 없으면 33.3%). 순서 순환 설계로 이 효과가 누출 추정에 섞이지 않게 했다.

<p align="center"><img src="description/figures/31_%EA%B7%B8%EB%A6%BC6-5_6.7%20%EA%B2%B0%EA%B3%BC%20%E2%91%A4%20%EC%9C%84%EC%B9%98%20%ED%8E%B8%ED%96%A5%20-%20%EC%88%9C%EC%84%9C%EB%A5%BC%20%EB%B0%94%EA%BF%94%20%EB%84%A3%EC%9D%80%20%EC%84%A4%EA%B3%84%EA%B0%80%20%ED%95%84%EC%9A%94%ED%96%88%EB%82%98.png" width="100%"></p>
<p align="center"><sub>그림 8. 평가자가 1위로 뽑은 답이 놓여 있던 위치.</sub></p>

## 6. Mitigating Preference Leakage

가장 뚜렷한 Gemini 평가자의 누출을 두 방법으로 줄일 수 있는지 확인했다(학생은 seed 42~44, 평가와 통계 모형은 본 실험과 동일).

- **① 데이터 비율 조정** : Gemini 학생의 학습 데이터 중 Gemini 답 비율을 100% → 75 / 50 / 33 / 25 / 0%로 줄이고 나머지를 GPT·Claude 답으로 채웠다. 비율이 낮을수록 누출이 줄었고, 50% 이하에서는 0과 구별되지 않았다.
- **② 문체 재작성** : 세 교사의 답을 모두 HCX-005로 같은 중립 문체로 다시 써서 학생을 학습했다. 내용은 그대로 두고 문체만 지운 조건이다. 누출은 +9.8 → **+4.7%p**로 절반 이상 줄었지만(같은 질문 비교 감소 −5.6%p, Holm p = 0.018) 0이 되지는 않았다.

<p align="center"><img src="description/figures/65_%EA%B7%B8%EB%A6%BC8-R_%ED%8F%AC%EC%8A%A4%ED%84%B0%EC%9A%A9%20-%20%EC%84%A0%ED%98%B8%20%EB%88%84%EC%B6%9C%20%EC%96%B5%EC%A0%9C%20%EB%91%90%20%EB%B0%A9%EB%B2%95.png" width="100%"></p>
<p align="center"><sub>그림 9. 선호 누출 억제 두 방법. ① 학습 데이터의 Gemini 답 비율 ② 문체 재작성 전후의 평가자별 누출.</sub></p>

평가 기준별로 보면, 문체 재작성 후 **가독성 누출은 사라졌고**(+8.0 → −0.5%p) **사실 정확성 누출은 절반이 남았다**(+11.4 → +5.7%p). 누출의 약 절반은 문체, 나머지는 내용에서 온다.

<p align="center"><img src="description/figures/63_%EA%B7%B8%EB%A6%BC8-26_8.2%20%ED%8F%89%EA%B0%80%20%EA%B2%B0%EA%B3%BC%20-%20Gemini%20%ED%8F%89%EA%B0%80%EC%9E%90%EC%9D%98%20%EA%B8%B0%EC%A4%80%EB%B3%84%20%EC%84%A0%ED%98%B8%20%EB%88%84%EC%B6%9C.png" width="100%"></p>
<p align="center"><sub>그림 10. 문체 재작성 전후 Gemini 평가자의 기준별 선호 누출.</sub></p>

## 7. Conclusion

- **RQ1 (존재)** : 선호 누출은 존재하지만 평가자에 따라 다르다. Gemini 평가자 +9.8%p, GPT 약한 증거, Claude 없음.
- **RQ2 (이질성)** : 평가자 간 차이가 가장 크고, 기준 중에서는 내용 판단(사실 정확성)에서 가장 크다.
- **RQ3 (강건성)** : 학생을 5번 따로 학습시켜도 매번 나타났고(5/5), 답변 길이를 통제하거나(+8.8%p) 분석 방법을 바꿔도(+8.6~+10.7%p) 유지되었다.
- **누출 억제** : 데이터 비율 조정과 문체 재작성 모두 누출을 줄였지만 완전히 없애지는 못했다. 누출의 약 절반은 문체, 나머지는 내용에서 온다.

**시사점.** 합성데이터를 생성한 모델과 같은 계열의 평가자는 결과를 왜곡할 수 있고 문체 통일만으로는 누출을 없앨 수 없으므로, 다른 계열의 평가자를 쓰고 데이터 출처를 다양화하는 평가 설계가 필요하다.

**한계.** 0.6B 크기의 학생 모델로만 실험해, 답변 품질이 높은 더 큰 학생 모델에서도 같은 누출이 나타나는지는 추가 검증이 필요하다.

---

## Repository

실행 순서: `data` → `inference`(교사 답) → `classify` / `sft` → `inference`(학생 답) → `judge` → `results`

| 폴더 | 내용 |
|---|---|
| [data/](data/) | KoAlpaca 학습 질문 1,200개, 평가 질문 400개 |
| [inference/](inference/) | 교사 답 생성(GPT·Claude·Gemini), 학생 답 생성 |
| [classify/](classify/) | 답변만으로 교사를 맞히는 분류 실험(로지스틱 회귀, KLUE-RoBERTa) |
| [sft/](sft/) | 학생 학습(Qwen3-0.6B-Base), 교차 loss, 비율 조정·문체 재작성 데이터 |
| [judge/](judge/) | LLM-as-a-Judge 채점과 통계 모형(순위 로짓, 1위 모형, 쌍 고정효과, 부트스트랩, Holm) |
| [results/](results/) | 요약 표와 그림 |
| [description/](description/) | 전체 연구 기록 [poster.html](description/poster.html), 그림 [figures/](description/figures/) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# API 키는 .env에 둔다(커밋하지 않음)
```

모든 실험의 설계 근거, 실제 입출력 예시, 세부 결과는 [description/poster.html](description/poster.html)에 있다.
