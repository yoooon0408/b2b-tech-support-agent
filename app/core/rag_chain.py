"""사용자 질문 -> ChromaDB 검색 -> 시스템 프롬프트 결합 -> Claude 호출 -> 답변 생성
으로 이어지는 RAG 파이프라인."""
from dataclasses import dataclass
from typing import List, Optional

from app.core.llm import generate_answer
from app.core.prompts import build_system_prompt
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
        return RAGResponse(answer=answer_text, sources=sources)


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
