"""LLM-as-a-Judge 평가 프롬프트와 평가 기준 (listwise 순위 평가).

평가 차원은 LLM 응답의 세부 기술 평가 체계인 FLASK(Ye et al., 2024)를 따랐다.
FLASK의 네 가지 상위 능력인 논리적 사고, 배경지식, 문제 처리, 사용자 정렬에서 본 연구의 질문 유형에 적용 가능한 대표 기술을 하나씩 선택하였다.
선택한 기술은 논리성(Logical Robustness), 사실 정확성(Factuality), 질문 충족도(Comprehension), 가독성(Readability)이다.
정답이 단일한 추론 문항을 전제하는 기술과 일상 질문에서 적용 빈도가 낮은 기술은 제외하였다.
응답 길이와 강하게 얽힌 기술도 제외하였으며, 길이는 별도 공변량으로 통제하였다.
각 기준의 1, 3, 5점 기준 문장은 FLASK의 정의를 한국어 일반 질문에 맞게 번안하였다.
가독성에는 한국어 평가자가 외국어 혼입을 잘 감점하지 못한다는 선행 보고(Son et al., 2024)에 따라 한국어 자연스러움을 포함하였다.
네 평가자에게 동일한 프롬프트를 사용하였고, 평가자는 각 기준마다 근거를 먼저 서술한 뒤 판단하였다.

평가 방식: 처음에는 답변 하나에 1~5점을 매기는 pointwise 방식이었으나, 파일럿에서 0.6B 학생의 답이 대부분 사실을
지어내 사실 정확성 120개 전부, 질문 충족도·논리성 대부분이 1점에 몰렸다(바닥 효과). 그래서 같은 질문·같은 seed의 학생
3명(교사만 다름)의 답을 한 번에 보여 주고 기준별로 1~3위 순위를 매기는 listwise 방식으로 바꿨다. 셋 다 품질이 낮아도
상대적으로 나은 답을 고르게 되므로 바닥 효과가 사라지고, 품질이 비슷한 답 사이의 선호(선호 누출이 드러나는 지점)를 잴 수 있다.
1, 3, 5점 기준 문장은 각 기준에서 좋은 답과 나쁜 답을 구별하는 판단 기준으로 그대로 제시한다.
출력은 네 기준별 순위와, 네 기준을 종합한 최종 순위(overall)다. 최종 순위로 승률(자기 계열 학생을 다른 학생보다
위에 놓는 비율)을 계산해 주 결과로 쓰고, 기준별 순위로 어떤 판단에서 편향이 드러나는지 본다.

위치 편향 대응: 답변 순서는 select_items.py가 정한다(같은 묶음을 순서를 돌려 3번 평가, 각 학생이 각 위치에 한 번씩).
프롬프트에는 "제시된 순서는 품질과 무관하다"는 지시를 넣었고, 근거를 먼저 쓰게 하는 것도 위치 편향을 줄이는
방법으로 보고되었다(Wang et al., 2023, evidence calibration).
"""

import json
import re

# (JSON 키, 한국어 이름, FLASK 상위 능력)
CRITERIA = [
    ("factuality", "사실 정확성", "배경지식"),
    ("comprehension", "질문 충족도", "문제 처리"),
    ("logical_robustness", "논리성", "논리적 사고"),
    ("readability", "가독성과 한국어 자연스러움", "사용자 정렬"),
]
CRITERIA_KEYS = [c[0] for c in CRITERIA]
OVERALL = "overall"                       # 네 기준을 종합한 최종 순위 (승률 계산의 주 결과)
OUTPUT_KEYS = CRITERIA_KEYS + [OVERALL]
LABELS = ["A", "B", "C"]   # 답변 표시. 위치를 그대로 나타내며 출처(교사)와는 무관하다.

JUDGE_SYSTEM = """당신은 한국어 답변의 품질을 평가하는 엄격하고 공정한 평가자입니다.
하나의 질문에 대한 세 개의 답변(답변 A, 답변 B, 답변 C)을 읽고, 네 가지 기준 각각에 대해 세 답변의 순위를 1위부터 3위까지 매기십시오.

규칙
- 채점 전에 질문이 요구하는 핵심 내용과 사실관계를 스스로 확인하십시오.
- 각 기준마다 세 답변을 비교한 근거를 먼저 한두 문장으로 쓰고, 그 다음 순위를 정하십시오.
- 기준끼리 섞지 마십시오. 예를 들어 사실이 틀렸어도 문장이 자연스러우면 가독성 순위는 높을 수 있습니다.
- 답변의 길이 자체를 순위에 반영하지 마십시오. 길다고 좋은 답이 아닙니다.
- 답변을 쓴 모델이 누구인지 추측하지 마십시오.
- 답변이 제시된 순서(A, B, C)는 품질과 무관합니다. 순서에 영향을 받지 말고 내용만 비교하십시오.
- 명백히 틀린 정보는 감점하십시오. 자신의 지식으로 확인할 수 없는 정보는 근거에 "확인 불가"라고 적고, 사실 정확성의 3점 기준(확인되지 않는 세부 정보)에 따라 판단하십시오.
- 세 답변이 모두 부족하더라도 기준에 비추어 상대적으로 나은 순서대로 순위를 매기십시오. 동점은 허용하지 않습니다.
- 네 기준의 순위를 모두 매긴 뒤, 마지막으로 네 기준을 종합해 질문한 사람에게 가장 좋은 답변부터 최종 순위(overall)를 매기십시오.

아래는 각 기준에서 좋은 답(5점)과 나쁜 답(1점)을 구별하는 판단 기준입니다. 이 기준에 더 가까운 답변에 높은 순위를 주십시오.

[기준 1] 사실 정확성 (Factuality)
답변에 담긴 사실 정보가 정확한가.
5점: 핵심 정보와 세부 정보가 모두 정확하다.
3점: 핵심은 맞지만 부정확하거나 확인되지 않는 세부 정보가 있다.
1점: 핵심 정보가 틀렸거나 지어낸 내용이 중심이다.

[기준 2] 질문 충족도 (Comprehension)
질문의 명시적 요구와 암묵적 의도를 이해하고 그에 맞게 답했는가.
5점: 질문이 묻는 모든 부분에 직접 답하고 의도에 맞는 정보를 준다.
3점: 질문의 일부에만 답하거나 의도를 부분적으로 놓친다.
1점: 질문과 무관하거나 질문을 잘못 이해했다.

[기준 3] 논리성 (Logical Robustness)
설명의 전개가 일관되고 모순이나 비약이 없는가.
5점: 근거와 결론이 자연스럽게 이어지고 모순이 없다.
3점: 대체로 이어지지만 비약이나 근거 없는 주장이 일부 있다.
1점: 자기모순이 있거나 전개를 따라갈 수 없다.

[기준 4] 가독성과 한국어 자연스러움 (Readability)
읽기 쉽게 구성되었고 문장이 자연스러운 한국어인가.
5점: 구성이 명확하고 문법과 어휘가 자연스럽다. 불필요한 외국어 혼입이 없다.
3점: 이해는 되지만 어색한 문장, 반복, 번역투가 눈에 띈다.
1점: 문장이 깨지거나 반복이 심하거나 다른 언어가 섞여 읽기 어렵다."""

USER_TEMPLATE = """[질문]
{question}

[답변 A]
{answer_a}

[답변 B]
{answer_b}

[답변 C]
{answer_c}

다음 JSON 형식으로만 출력하십시오. ranking에는 가장 나은 답변부터 순서대로 "A", "B", "C"를 한 번씩만 쓰십시오.
{"factuality": {"reason": "세 답변을 비교한 근거 한두 문장", "ranking": ["1위", "2위", "3위"]},
 "comprehension": {"reason": "세 답변을 비교한 근거 한두 문장", "ranking": ["1위", "2위", "3위"]},
 "logical_robustness": {"reason": "세 답변을 비교한 근거 한두 문장", "ranking": ["1위", "2위", "3위"]},
 "readability": {"reason": "세 답변을 비교한 근거 한두 문장", "ranking": ["1위", "2위", "3위"]},
 "overall": {"reason": "네 기준을 종합한 근거 한두 문장", "ranking": ["1위", "2위", "3위"]}}"""


def build_user_prompt(question, answers):
    """answers: 제시 순서대로 답변 3개 [A, B, C]. JSON 예시의 중괄호 때문에 자리표시자만 바꾼다."""
    text = USER_TEMPLATE.replace("{question}", question)
    for label, answer in zip(["a", "b", "c"], answers):
        text = text.replace("{answer_" + label + "}", answer)
    return text


def parse_judgement(text):
    """평가자 응답에서 JSON을 꺼내 네 기준의 근거와 순위를 검증한다. 형식이 틀리면 ValueError.

    반환: {"factuality": {"reason": str, "ranking": ["B", "A", "C"]}, ... 4개 기준 + "overall"}
    ranking은 A·B·C를 한 번씩 담은 목록이어야 한다(동점·누락·중복 불가).
    """
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("JSON을 찾지 못함")
    data = json.loads(match.group(0))
    parsed = {}
    for key in OUTPUT_KEYS:
        item = data.get(key)
        if not isinstance(item, dict):
            raise ValueError(f"{key} 항목 없음")
        ranking = item.get("ranking")
        if isinstance(ranking, str):   # 가끔 "B > C > A"처럼 문자열로 쓴다. 아래에서 A·B·C가 한 번씩인지 검증한다.
            ranking = re.findall(r"[ABC]", ranking.upper())
        if not isinstance(ranking, list):
            raise ValueError(f"{key} ranking 없음")
        ranking = [str(x).strip().upper().replace("답변", "").strip() for x in ranking]
        if sorted(ranking) != LABELS:
            raise ValueError(f"{key} 순위 이상: {ranking}")
        parsed[key] = {"reason": str(item.get("reason", "")), "ranking": ranking}
    return parsed
