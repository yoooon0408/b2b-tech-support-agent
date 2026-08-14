"""생성된 답변이 검색된 Context(문서)에 실제로 근거하고 있는지 검증하는 Faithfulness 평가 로직.

LLM-as-judge 방식: 답변을 원자적 주장(claim) 단위로 분해한 뒤, 각 주장이 검색된 Context로부터
직접 뒷받침되는지 Claude에게 판정시켜 0.0~1.0 사이의 신뢰도 점수를 계산한다.
"""
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from app.config import FAITHFULNESS_THRESHOLD
from app.core.llm import generate_json
from app.core.prompts import FAITHFULNESS_JUDGE_SYSTEM_PROMPT, build_faithfulness_judge_prompt

_JUDGE_MAX_TOKENS = 1536


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    evidence: str = ""


@dataclass
class FaithfulnessResult:
    score: float  # 0.0 (전혀 근거 없음) ~ 1.0 (완전히 근거함)
    claims: List[ClaimVerdict] = field(default_factory=list)
    reasoning: str = ""
    # 판정 자체(LLM 호출/파싱)가 실패한 경우 True. 이 경우 score는 0.0으로 fail-closed 처리된다.
    evaluation_failed: bool = False

    @property
    def unsupported_claims(self) -> List[ClaimVerdict]:
        return [c for c in self.claims if not c.supported]

    @property
    def is_faithful(self) -> bool:
        return not self.evaluation_failed and self.score >= FAITHFULNESS_THRESHOLD


def evaluate_faithfulness(question: str, answer: str, chunks: List[Document]) -> FaithfulnessResult:
    """LLM-as-judge로 답변의 각 주장이 Context에 근거하는지 검증하고 신뢰도 점수를 계산한다.

    claim이 하나도 없는 답변(예: 순수 인사말)은 검증할 대상이 없으므로 score=1.0으로 처리한다.
    판정 자체가 실패하면(모델 호출/JSON 파싱 오류 등) 안전하게 신뢰 불가로 처리해
    담당자 전달(티켓 생성) 경로로 넘어가도록 한다 (fail-closed).
    """
    judge_prompt = build_faithfulness_judge_prompt(question, answer, chunks)
    try:
        result = generate_json(
            system_prompt=FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
            user_content=judge_prompt,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
        claims = [
            ClaimVerdict(
                claim=c.get("claim", ""),
                supported=bool(c.get("supported", False)),
                evidence=c.get("evidence", ""),
            )
            for c in result.get("claims", [])
        ]
        reasoning = result.get("reasoning", "")
    except Exception as exc:  # 모델 호출 실패, JSON 파싱 실패 등
        return FaithfulnessResult(
            score=0.0,
            claims=[],
            reasoning=f"Faithfulness evaluation failed: {exc}",
            evaluation_failed=True,
        )

    if not claims:
        return FaithfulnessResult(score=1.0, claims=[], reasoning=reasoning)

    supported_count = sum(1 for c in claims if c.supported)
    score = supported_count / len(claims)
    return FaithfulnessResult(score=score, claims=claims, reasoning=reasoning)
