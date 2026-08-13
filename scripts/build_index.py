"""data/processed/chunks.jsonl(app.ingestion.ingest 출력)을 읽어
ChromaDB에 임베딩·저장하는 스크립트.

실행:
    python -m app.ingestion.ingest   # 1) 문서 -> 청크
    python -m scripts.build_index    # 2) 청크 -> 벡터DB
"""
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.retrieval.vectorstore import VectorStore

CHUNKS_FILE = Path("data/processed/chunks.jsonl")


def load_chunks(path: Path) -> List[Document]:
    documents = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(
                Document(page_content=record["page_content"], metadata=record["metadata"])
            )
    return documents


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_FILE)
    store = VectorStore()
    added = store.add_documents(chunks)
    print(f"{added}개 청크를 벡터DB에 저장했습니다. (컬렉션 총 {store.count}개)")
