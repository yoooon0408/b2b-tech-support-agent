# B2B Technical Support RAG Agent

영문 기술 문서를 기반으로 B2B 고객 지원 질의에 답변하는 RAG(Retrieval-Augmented Generation) AI 에이전트.

## Stack
- **Orchestration**: LangChain
- **LLM**: Claude (Anthropic API)
- **Vector Store**: ChromaDB
- **Embeddings**: sentence-transformers (로컬) — 필요 시 Voyage AI API로 교체 가능
- **UI**: Streamlit

## Project Structure
```
app/
  core/         # RAG 체인, LLM/임베딩 래퍼, 프롬프트 템플릿
  ingestion/    # 문서 로딩 · 청킹 파이프라인
  retrieval/    # ChromaDB 벡터스토어 연동
  agent/        # 에이전트 오케스트레이션 · 툴 정의
  utils/        # 로깅 등 공통 유틸
data/
  raw/          # 원본 기술 문서
  processed/    # 전처리·청킹된 문서
  chroma_db/    # 벡터 DB 영속 저장소
evaluation/     # 검색·답변 품질 평가 (ragas)
tests/          # 단위 테스트
scripts/        # 인덱스 빌드, 평가 실행용 CLI 스크립트
docs/portfolio/ # 포트폴리오 정리 문서 (PAAR)
```

## Setup
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env   # ANTHROPIC_API_KEY 입력
```

## Run
```powershell
streamlit run app/main.py
```
