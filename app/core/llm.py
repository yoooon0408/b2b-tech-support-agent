"""Claude(Anthropic) API 호출 래퍼."""
from anthropic import Anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL, CLAUDE_TEMPERATURE

_client = Anthropic(api_key=ANTHROPIC_API_KEY)


def generate_answer(system_prompt: str, question: str) -> str:
    """시스템 프롬프트와 사용자 질문으로 Claude를 호출해 답변 텍스트를 반환한다."""
    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        temperature=CLAUDE_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text
