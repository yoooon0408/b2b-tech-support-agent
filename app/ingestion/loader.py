from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: Path) -> List[Document]:
    """PDF를 페이지 단위 Document 리스트로 읽어온다. metadata에 source, page가 포함된다."""
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = file_path.name
    return documents


def load_markdown(file_path: Path) -> Document:
    """Markdown 파일 전체를 하나의 Document로 읽어온다. 헤더 기반 분할은 splitter에서 처리한다."""
    text = file_path.read_text(encoding="utf-8")
    return Document(page_content=text, metadata={"source": file_path.name})
