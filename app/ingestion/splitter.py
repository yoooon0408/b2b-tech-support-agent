from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_markdown_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)


def split_pdf_documents(documents: List[Document]) -> List[Document]:
    """PDF 페이지 Document 리스트를 chunk_size=1000, chunk_overlap=200으로 분할한다."""
    return _char_splitter.split_documents(documents)


def split_markdown_document(document: Document) -> List[Document]:
    """Markdown은 헤더(#, ##, ###) 단위로 먼저 섹션을 나눠 문맥을 보존한 뒤,
    섹션이 chunk_size를 넘으면 RecursiveCharacterTextSplitter로 재분할한다."""
    header_sections = _markdown_header_splitter.split_text(document.page_content)

    for section in header_sections:
        section.metadata["source"] = document.metadata.get("source")

    return _char_splitter.split_documents(header_sections)
