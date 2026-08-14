// MMR(Maximal Marginal Relevance) 재랭킹의 코사인 유사도 계산 + 그리디 선택 루프를
// C++로 옮긴 확장 모듈.
//
// v1에서는 pybind11의 unchecked<> 프록시로 원소 접근(cand(i,j))을 하고
// std::vector<std::vector<float>>로 pairwise 유사도 행렬을 저장했는데, 벤치마크 결과
// n>=100부터 numpy(@演算, BLAS)보다 오히려 느렸다. 원인 두 가지:
//   1) vector<vector<float>>는 행마다 별도 힙 할당이라 포인터를 두 번 따라가야 하고
//      캐시 지역성이 나쁘다 (row-major flat buffer 대비).
//   2) unchecked<> 접근자는 매 원소마다 stride 계산이 끼어들어 컴파일러가
//      내적 루프를 SIMD로 자동 벡터화하기 어렵게 만든다.
// numpy의 `@`는 BLAS(멀티스레드+SIMD+캐시 블로킹)로 도는데, 이걸 단순 3중 for문으로
// 이기려면 최소한 "raw pointer + 연속 메모리 + 컴파일러가 벡터화할 수 있는 형태"가
// 되어야 한다는 게 확인된 사실. 아래는 그걸 반영한 버전.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

// 내적을 double로 누적한다. numpy(BLAS)는 pairwise summation 등으로 float32 루프보다
// 오차가 훨씬 작은데, 여기서 float로 단순 누적하면 그 오차 때문에 MMR의 근소한 차이
// 후보들에서 tie-break가 뒤집혀 numpy 구현과 다른 선택을 하는 경우가 있었다.
// (n>=30부터 관찰됨: 최종 answer 품질에 큰 영향은 없지만 재현성이 깨지는 문제라 double로 맞춤)
inline double dot(const float* a, const float* b, Py_ssize_t d) {
    double acc = 0.0;
    for (Py_ssize_t k = 0; k < d; ++k) acc += static_cast<double>(a[k]) * static_cast<double>(b[k]);
    return acc;
}

}  // namespace

// candidates: (n, d) row-major, query: (d,) — 둘 다 L2-normalize 되어 있다고 가정.
py::array_t<float> cosine_scores(py::array_t<float, py::array::c_style | py::array::forcecast> candidates,
                                  py::array_t<float, py::array::c_style | py::array::forcecast> query) {
    py::buffer_info cbuf = candidates.request();
    py::buffer_info qbuf = query.request();
    if (cbuf.ndim != 2 || qbuf.ndim != 1 || cbuf.shape[1] != qbuf.shape[0]) {
        throw std::runtime_error("dimension mismatch between candidates and query");
    }

    Py_ssize_t n = cbuf.shape[0];
    Py_ssize_t d = cbuf.shape[1];
    const float* cand = static_cast<const float*>(cbuf.ptr);
    const float* q = static_cast<const float*>(qbuf.ptr);

    py::array_t<float> result(n);
    float* out = static_cast<float*>(result.request().ptr);

#ifdef _OPENMP
#pragma omp parallel for if (n > 64)
#endif
    for (Py_ssize_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(dot(cand + i * d, q, d));
    }
    return result;
}

// MMR 그리디 선택: relevance(query 유사도)와 diversity(이미 뽑힌 후보와의 최대 유사도)를
// lambda로 절충하며 top_k개를 순서대로 뽑는다. pairwise 유사도는 한 번만 계산해 재사용.
std::vector<int> mmr_select(py::array_t<float, py::array::c_style | py::array::forcecast> candidates,
                             py::array_t<float, py::array::c_style | py::array::forcecast> query,
                             int top_k, float lambda_) {
    py::buffer_info cbuf = candidates.request();
    Py_ssize_t n = cbuf.shape[0];
    Py_ssize_t d = cbuf.shape[1];
    const float* cand = static_cast<const float*>(cbuf.ptr);

    py::array_t<float> rel_arr = cosine_scores(candidates, query);
    const float* rel = static_cast<const float*>(rel_arr.request().ptr);

    // pairwise 유사도를 (n*n) 연속 버퍼에 저장 (vector<vector<float>> 대비 캐시 지역성 개선).
    // (i, j) 쌍은 i<=j에서만 계산하고 대칭으로 채우며, i에 대해 병렬화해도 서로 다른
    // 스레드가 쓰는 (row, col) 쌍이 겹치지 않아 락 없이 안전하다.
    std::vector<float> sim(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);
    float* sim_ptr = sim.data();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (n > 64)
#endif
    for (Py_ssize_t i = 0; i < n; ++i) {
        for (Py_ssize_t j = i; j < n; ++j) {
            float s = static_cast<float>(dot(cand + i * d, cand + j * d, d));
            sim_ptr[i * n + j] = s;
            sim_ptr[j * n + i] = s;
        }
    }

    std::vector<bool> picked(n, false);
    std::vector<int> result;
    result.reserve(top_k);

    int k = (top_k < n) ? top_k : static_cast<int>(n);
    for (int step = 0; step < k; ++step) {
        float best_score = -std::numeric_limits<float>::infinity();
        int best_idx = -1;
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (picked[i]) continue;
            float diversity = 0.0f;
            for (int j : result) diversity = std::max(diversity, sim_ptr[i * n + j]);
            float score = lambda_ * rel[i] - (1.0f - lambda_) * diversity;
            if (score > best_score) {
                best_score = score;
                best_idx = static_cast<int>(i);
            }
        }
        picked[best_idx] = true;
        result.push_back(best_idx);
    }
    return result;
}

PYBIND11_MODULE(rerank_cpp, m) {
    m.doc() = "MMR reranking primitives (cosine similarity + greedy MMR selection) in C++";
    m.def("cosine_scores", &cosine_scores, py::arg("candidates"), py::arg("query"));
    m.def("mmr_select", &mmr_select,
          py::arg("candidates"), py::arg("query"),
          py::arg("top_k") = 3, py::arg("lambda_") = 0.7f);
}
