"""GPT/Claude/Gemini 답변 생성과 학생(Qwen) 학습·추론에 공통으로 쓰는 프롬프트."""

SYSTEM_PROMPT = """당신은 한국어 사용자의 질문에 답하는 어시스턴트입니다.
다음 규칙을 지켜 답변하십시오.

1. 한국어로 답합니다. 전문용어는 필요하면 영어를 함께 적을 수 있습니다.
2. 질문에 바로 답합니다. 되묻거나 추가 정보를 요청하지 않습니다. 질문이 모호하면 가장 일반적인 해석으로 답합니다.
3. 답변은 600자 이내에서 완결되게 작성합니다.
4. 자신이 AI라는 점, 모델 이름, 개발사를 언급하지 않습니다.
5. 확실하지 않은 내용을 단정하지 않습니다. 존재하지 않는 출처나 링크를 만들지 않습니다.
6. 의료, 법률, 금융처럼 전문가 확인이 필요한 질문에는 일반적인 정보를 제공하고, 필요하면 전문가 상담을 한 문장으로 권합니다."""

# 학생 모델은 Qwen의 ChatML 형식으로 교사와 같은 시스템 프롬프트를 받는다.
# 학습 때는 build_prompt(질문) + 답변 + END_TOKEN을, 추론 때는 build_prompt(질문)만 넣고 END_TOKEN에서 멈춘다.
END_TOKEN = "<|im_end|>"


def build_prompt(instruction):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}{END_TOKEN}\n"
        f"<|im_start|>user\n{instruction}{END_TOKEN}\n"
        f"<|im_start|>assistant\n"
    )
