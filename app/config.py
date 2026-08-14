"""환경 변수 기반 설정."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "1024"))

CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))

# 답변이 검색된 Context에 실제로 근거하는지 판정하는 Faithfulness 점수(0.0~1.0)의 최소 합격 기준.
# 이 값 미만이면 담당자(CSM/Support) 전달용 티켓 리포트가 자동 생성된다.
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7"))

