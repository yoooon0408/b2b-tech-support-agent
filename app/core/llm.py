"""Claude(Anthropic) API 호출 래퍼."""
import json
import re

from anthropic import Anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MAX_TOKENS, CLAUDE_MODEL

_client = Anthropic(api_key=ANTHROPIC_API_KEY)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_text(response) -> str:
    """응답 content 블록 중 첫 번째 텍스트 블록을 반환한다.
    확장 사고(extended thinking)가 켜진 모델은 ThinkingBlock을 content[0]으로 먼저 반환할 수 있어
    content[0]을 텍스트로 가정하면 안 된다."""
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("응답에 텍스트 블록이 없습니다.")


def generate_answer(system_prompt: str, question: str) -> str:
    """시스템 프롬프트와 사용자 질문으로 Claude를 호출해 답변 텍스트를 반환한다."""
    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return _extract_text(response)


def generate_json(system_prompt: str, user_content: str, max_tokens: int = CLAUDE_MAX_TOKENS) -> dict:
    """Claude를 호출해 JSON 객체로 파싱 가능한 구조화된 응답을 받는다."""
    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    raw_text = _extract_text(response).strip()
    fenced = _JSON_FENCE_RE.search(raw_text)
    candidate = fenced.group(1) if fenced else raw_text
    return json.loads(candidate)
