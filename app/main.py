"""B2B 기술 지원 RAG 에이전트 데모용 Streamlit 웹 애플리케이션.

좌측 사이드바에서 영문 기술 문서(PDF)를 업로드해 벡터DB에 색인하고, 메인 화면에서 질문을 입력하면
Claude가 색인된 문서를 근거로 답변한다. 답변의 Faithfulness가 기준 미달이거나 해결 불가능한 질문이면
담당자(CSM/Support) 전달용 티켓 리포트가 함께 표시된다.

실행:
    streamlit run app/main.py
"""
import sys
import tempfile
from pathlib import Path

# `streamlit run app/main.py`로 직접 실행하면 스크립트 폴더(app/)만 sys.path에 잡혀
# `import app.xxx`가 깨지므로, 프로젝트 루트를 명시적으로 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.config import ANTHROPIC_API_KEY
from app.core.rag_chain import RAGChain, RAGResponse
from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_pdf_documents
from app.retrieval.vectorstore import VectorStore

st.set_page_config(page_title="B2B Tech Support Agent", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_rag_chain() -> RAGChain:
    return RAGChain(vector_store=VectorStore())


def index_uploaded_pdf(vector_store: VectorStore, uploaded_file) -> int:
    """업로드된 PDF를 임시 파일로 저장 -> 로드 -> 청크 분할 -> 벡터DB 색인 후 청크 수를 반환한다."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        documents = load_pdf(tmp_path)
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name  # 임시 파일명 대신 원본 업로드 파일명 사용

        chunks = split_pdf_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{Path(uploaded_file.name).stem}-{i}"

        return vector_store.add_documents(chunks)
    finally:
        tmp_path.unlink(missing_ok=True)


def render_sidebar(vector_store: VectorStore) -> None:
    st.sidebar.header("📄 기술 문서 업로드")
    st.sidebar.caption("영문 기술 문서(PDF)를 업로드하면 벡터DB에 색인되어 답변의 근거로 사용됩니다.")

    uploaded_files = st.sidebar.file_uploader(
        "PDF 파일 선택 (여러 개 가능)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.sidebar.button("문서 색인하기", type="primary", use_container_width=True):
        if "indexed_files" not in st.session_state:
            st.session_state.indexed_files = []

        with st.sidebar.spinner("문서를 색인하는 중입니다..."):
            total_chunks = 0
            for uploaded_file in uploaded_files:
                added = index_uploaded_pdf(vector_store, uploaded_file)
                total_chunks += added
                st.session_state.indexed_files.append((uploaded_file.name, added))

        st.sidebar.success(f"{len(uploaded_files)}개 파일, 총 {total_chunks}개 청크를 색인했습니다.")

    st.sidebar.divider()
    st.sidebar.metric("현재 벡터DB 청크 수", vector_store.count)

    if st.session_state.get("indexed_files"):
        with st.sidebar.expander("이번 세션에 색인한 문서", expanded=False):
            for name, added in st.session_state.indexed_files:
                st.caption(f"- {name} ({added}개 청크)")

    st.sidebar.divider()
    if st.sidebar.button("대화 초기화", use_container_width=True):
        st.session_state.history = []
        st.rerun()


def render_sources(sources) -> None:
    with st.expander(f"📚 참조한 문서 출처 (Source Documents) — {len(sources)}건", expanded=False):
        if not sources:
            st.caption("검색된 출처가 없습니다.")
            return
        for i, src in enumerate(sources, start=1):
            parts = [src.source]
            if src.section:
                parts.append(src.section)
            if src.page is not None:
                parts.append(f"p.{src.page}")
            st.markdown(f"**[{i}]** {' / '.join(parts)}")


def render_faithfulness(faithfulness) -> None:
    if faithfulness is None:
        return
    score_pct = f"{faithfulness.score * 100:.0f}%"
    if faithfulness.is_faithful:
        st.caption(f"✅ Faithfulness {score_pct} — 검색된 문서에 근거한 답변입니다.")
    else:
        st.caption(f"⚠️ Faithfulness {score_pct} — 문서 근거가 부족합니다. 담당자 확인이 필요합니다.")


def render_ticket(ticket) -> None:
    if ticket is None:
        return
    st.warning(
        f"이 답변은 담당자 확인이 필요합니다. (Ticket ID: `{ticket.ticket_id}`, "
        f"Priority: **{ticket.priority}**)"
    )
    with st.expander("🎫 담당자(CSM/Support) 전달용 티켓 리포트 (English)", expanded=True):
        st.code(ticket.to_text(), language="text")


def render_turn(question: str, result: RAGResponse) -> None:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(result.answer)
        render_faithfulness(result.faithfulness)
        render_sources(result.sources)
        render_ticket(result.ticket)


def main() -> None:
    st.title("🛠️ B2B Technical Support Agent")
    st.caption("영문 기술 문서 기반 RAG(Retrieval-Augmented Generation) 고객 지원 에이전트 데모")

    if not ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY가 설정되어 있지 않습니다. `.env` 파일을 확인해주세요.")
        st.stop()

    chain = get_rag_chain()
    render_sidebar(chain.vector_store)

    if chain.vector_store.count == 0:
        st.info("먼저 좌측 사이드바에서 기술 문서(PDF)를 업로드하고 색인해주세요.")

    if "history" not in st.session_state:
        st.session_state.history = []

    for question, result in st.session_state.history:
        render_turn(question, result)

    question = st.chat_input("질문을 입력하세요 / Ask a question about the documentation")
    if question:
        with st.spinner("답변을 생성하는 중입니다..."):
            result = chain.answer(question)
        st.session_state.history.append((question, result))
        render_turn(question, result)


if __name__ == "__main__":
    main()
