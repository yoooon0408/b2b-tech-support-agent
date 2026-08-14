"""사용자 질문 -> ChromaDB 검색 -> 시스템 프롬프트 결합 -> Claude 호출 -> 답변 생성 ->
Faithfulness 검증 -> (필요 시) 담당자 전달용 티켓 리포트 생성 으로 이어지는 RAG 파이프라인."""
from dataclasses import dataclass
from typing import List, Optional

from app.core.faithfulness import FaithfulnessResult, evaluate_faithfulness
from app.core.llm import generate_answer
from app.core.prompts import build_system_prompt
from app.core.ticket import TicketReport, build_ticket_report, is_unresolvable_answer
from app.retrieval.vectorstore import DEFAULT_TOP_K, VectorStore


@dataclass
class Source:
    source: str
    section: Optional[str]
    page: Optional[int]


@dataclass
class RAGResponse:
    answer: str
    sources: List[Source]
    # 답변이 "문서에서 확인 불가" 정형 문구인 경우 faithfulness 평가는 건너뛰므로 None일 수 있다.
    faithfulness: Optional[FaithfulnessResult] = None
    # 답변 품질이 기준 미달이거나(low faithfulness) 해결 불가능한 질문일 경우에만 생성된다.
    ticket: Optional[TicketReport] = None


class RAGChain:
    """질문에 대해 관련 문서를 검색하고 Claude로 답변을 생성하는 RAG 파이프라인."""

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self.vector_store = vector_store or VectorStore()

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> RAGResponse:
        chunks = self.vector_store.search(question, top_k=top_k)
        system_prompt = build_system_prompt(chunks)
        answer_text = generate_answer(system_prompt, question)

        sources = [
            Source(
                source=chunk.metadata.get("source", "unknown"),
                section=chunk.metadata.get("h1") or chunk.metadata.get("h2") or chunk.metadata.get("h3"),
                page=chunk.metadata.get("page"),
            )
            for chunk in chunks
        ]

        faithfulness_result: Optional[FaithfulnessResult] = None
        ticket: Optional[TicketReport] = None

        if is_unresolvable_answer(answer_text):
            # 검증할 factual claim이 없는 정형 거절 답변이므로 faithfulness 평가는 건너뛴다.
            ticket = build_ticket_report(question, answer_text, chunks)
        else:
            faithfulness_result = evaluate_faithfulness(question, answer_text, chunks)
            if not faithfulness_result.is_faithful:
                ticket = build_ticket_report(question, answer_text, chunks, faithfulness_result)

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            faithfulness=faithfulness_result,
            ticket=ticket,
        )


if __name__ == "__main__":
    chain = RAGChain()
    question = input("질문을 입력하세요 / Ask a question: ")
    result = chain.answer(question)

    print("\n[답변 / Answer]")
    print(result.answer)

    print("\n[출처 / Sources]")
    for src in result.sources:
        parts = [src.source]
        if src.section:
            parts.append(src.section)
        if src.page is not None:
            parts.append(f"p.{src.page}")
        print(f"- {' / '.join(parts)}")

    if result.faithfulness is not None:
        print(f"\n[Faithfulness] score={result.faithfulness.score:.2f}")

    if result.ticket is not None:
        print("\n[담당자 전달용 티켓 리포트 / Escalation Ticket]")
        print(result.ticket.to_text())
