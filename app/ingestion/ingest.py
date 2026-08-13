"""data/raw 아래의 PDF, Markdown 문서를 읽어 청크 단위 Document로 변환하고
data/processed/chunks.jsonl에 저장하는 CLI 스크립트.

실행:
    python -m app.ingestion.ingest
"""
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.ingestion.loader import load_markdown, load_pdf
from app.ingestion.splitter import split_markdown_document, split_pdf_documents

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SUPPORTED_SUFFIXES = (".pdf", ".md")


def ingest_file(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return split_pdf_documents(load_pdf(file_path))
    if suffix == ".md":
        return split_markdown_document(load_markdown(file_path))

    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


def ingest_all(raw_dir: Path = RAW_DIR) -> List[Document]:
    chunks: List[Document] = []
    for file_path in sorted(raw_dir.glob("**/*")):
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        file_chunks = ingest_file(file_path)
        for i, chunk in enumerate(file_chunks):
            chunk.metadata["chunk_id"] = f"{file_path.stem}-{i}"

        chunks.extend(file_chunks)
        print(f"{file_path.name}: {len(file_chunks)}개 청크")

    return chunks


def save_chunks(chunks: List[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            line = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    all_chunks = ingest_all()
    out_file = PROCESSED_DIR / "chunks.jsonl"
    save_chunks(all_chunks, out_file)
    print(f"총 {len(all_chunks)}개 청크를 {out_file}에 저장했습니다.")
