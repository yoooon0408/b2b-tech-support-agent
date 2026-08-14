"""문서 청크를 임베딩하여 ChromaDB에 저장하고, 질문에 대해 유사도 검색을 수행하는 벡터스토어 클래스."""
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.retrieval.reranker import mmr_select, normalize

DEFAULT_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
DEFAULT_COLLECTION_NAME = "tech_docs"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
DEFAULT_FETCH_K = 20
DEFAULT_MMR_LAMBDA = 0.7


class VectorStore:
    """청크 임베딩·저장과 유사도 검색을 담당하는 클래스."""

    def __init__(
        self,
        persist_dir: Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir),
        )

    def add_documents(self, chunks: List[Document]) -> int:
        """청크 리스트를 임베딩하여 ChromaDB에 저장한다. chunk_id를 문서 id로 써서
        같은 청크를 다시 넣으면 덮어쓴다(중복 방지)."""
        if not chunks:
            return 0

        ids = [
            chunk.metadata.get("chunk_id", f"chunk-{i}")
            for i, chunk in enumerate(chunks)
        ]
        self.store.add_documents(documents=chunks, ids=ids)
        return len(chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Document]:
        """질문과 가장 유사한 상위 top_k개 청크를 반환한다."""
        return self.store.similarity_search(query, k=top_k)

    def search_with_score(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        """(청크, 거리 점수) 튜플을 반환한다. 점수가 낮을수록 더 유사하다."""
        return self.store.similarity_search_with_score(query, k=top_k)

    def search_mmr(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_: float = DEFAULT_MMR_LAMBDA,
    ) -> List[Document]:
        """관련성과 다양성(중복 회피)을 함께 고려해 상위 top_k개 청크를 선택한다.

        1) 순수 유사도 기준으로 fetch_k개 후보를 먼저 가져오고,
        2) 후보들을 인제스트 시 이미 계산해 ChromaDB에 저장된 임베딩을 재사용해
           (재임베딩 없이) MMR로 top_k개로 압축한다.
        내부 재랭킹 연산은 rerank_cpp(native)가 있으면 그쪽을 쓰고, 없으면 Python으로 폴백한다.
        """
        candidates = self.store.similarity_search(query, k=fetch_k)
        if not candidates:
            return []

        ids = [c.metadata.get("chunk_id") for c in candidates]
        stored = self.store._collection.get(ids=ids, include=["embeddings"])
        id_to_vec = dict(zip(stored["ids"], stored["embeddings"]))
        cand_vecs = normalize(np.array([id_to_vec[i] for i in ids], dtype=np.float32))

        query_vec = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        query_vec = normalize(query_vec.reshape(1, -1))[0]

        idx = mmr_select(cand_vecs, query_vec, top_k=top_k, lambda_=lambda_)
        return [candidates[i] for i in idx]

    @property
    def count(self) -> int:
        """현재 컬렉션에 저장된 청크 수."""
        return self.store._collection.count()
