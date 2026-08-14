"""MMR(Maximal Marginal Relevance) 재랭킹.

Chroma에서 top_k보다 넉넉히(fetch_k) 후보를 가져온 뒤, 질문과의 관련성과 이미 뽑힌
청크와의 중복도를 함께 고려해 최종 top_k를 그리디하게 선택한다. 이 그리디 선택은
매 단계마다 남은 후보를 전부 훑는 이중 루프라 순수 Python으로 돌리면 후보 수가
늘어날수록 인터프리터 오버헤드가 누적된다. native/rerank_cpp(pybind11)가 빌드되어
있으면 그쪽을 쓰고, 없으면 동일한 로직의 Python 구현으로 폴백한다.
"""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_NATIVE_DIR = Path(__file__).resolve().parents[2] / "native"
if str(_NATIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_NATIVE_DIR))

try:
    import rerank_cpp

    HAS_NATIVE = True
except ImportError:
    rerank_cpp = None
    HAS_NATIVE = False


def normalize(vectors: np.ndarray) -> np.ndarray:
    """행 벡터들을 L2-normalize한다 (코사인 유사도를 내적으로 계산하기 위함)."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def mmr_select_py(
    candidates: np.ndarray, query: np.ndarray, top_k: int, lambda_: float
) -> List[int]:
    """rerank_cpp.mmr_select와 동일한 그리디 선택 로직의 순수 Python 버전.
    정확성 검증과 벤치마크 기준선(baseline)으로 쓴다."""
    n = candidates.shape[0]
    rel = candidates @ query
    sim = candidates @ candidates.T

    picked = [False] * n
    result: List[int] = []
    k = min(top_k, n)
    for _ in range(k):
        best_score = -np.inf
        best_idx = -1
        for i in range(n):
            if picked[i]:
                continue
            diversity = max((sim[i, j] for j in result), default=0.0)
            score = lambda_ * rel[i] - (1 - lambda_) * diversity
            if score > best_score:
                best_score = score
                best_idx = i
        picked[best_idx] = True
        result.append(best_idx)
    return result


def mmr_select(
    candidates: np.ndarray,
    query: np.ndarray,
    top_k: int = 3,
    lambda_: float = 0.7,
    use_native: Optional[bool] = None,
) -> List[int]:
    """candidates(n, d)에서 query(d,)와의 MMR 기준 top_k 인덱스를 반환한다.
    use_native=None이면 rerank_cpp이 있을 때 자동으로 사용한다."""
    candidates = np.ascontiguousarray(candidates, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)

    should_use_native = HAS_NATIVE if use_native is None else (use_native and HAS_NATIVE)
    if should_use_native:
        return rerank_cpp.mmr_select(candidates, query, top_k, lambda_)
    return mmr_select_py(candidates, query, top_k, lambda_)
