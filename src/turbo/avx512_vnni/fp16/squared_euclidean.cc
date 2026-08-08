// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "avx512_vnni/fp16/squared_euclidean.h"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace zvec::turbo::avx512_vnni {
namespace {

inline float Reduce(__m512 value) {
  // Match the reduction tree used by reimpl/vamana and KGN.
  const __m256 low256 = _mm512_castps512_ps256(value);
  const __m256 high256 = _mm512_extractf32x8_ps(value, 1);
  const __m256 sum256 = _mm256_add_ps(low256, high256);
  const __m128 low128 = _mm256_castps256_ps128(sum256);
  const __m128 high128 = _mm256_extractf128_ps(sum256, 1);
  const __m128 sum128 = _mm_add_ps(low128, high128);
  const __m128 pair = _mm_hadd_ps(sum128, sum128);
  return _mm_cvtss_f32(
      _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55)));
}

inline float HalfToFloat(uint16_t value) {
  return _cvtsh_ss(value);
}

template <size_t Batch>
void DistanceBatch(const void **vectors, const uint16_t *query,
                   size_t dimension, float *distances) {
  __m512 sums0[Batch];
  __m512 sums1[Batch];
  for (size_t lane = 0; lane < Batch; ++lane) {
    sums0[lane] = _mm512_setzero_ps();
    sums1[lane] = _mm512_setzero_ps();
  }

  size_t d = 0;
  for (; d + 32 <= dimension; d += 32) {
    // Expand one cache line of the FP16 query once for all candidate rows.
    // Two accumulators break the dependency chain across the 32 dimensions.
    const __m512i qh = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(query + d));
    const __m512 q0 = _mm512_cvtph_ps(_mm512_castsi512_si256(qh));
    const __m512 q1 =
        _mm512_cvtph_ps(_mm512_extracti64x4_epi64(qh, 1));
    for (size_t lane = 0; lane < Batch; ++lane) {
      const auto *row = static_cast<const uint16_t *>(vectors[lane]);
      const __m512i xh = _mm512_loadu_si512(
          reinterpret_cast<const __m512i *>(row + d));
      const __m512 x0 = _mm512_cvtph_ps(_mm512_castsi512_si256(xh));
      const __m512 x1 =
          _mm512_cvtph_ps(_mm512_extracti64x4_epi64(xh, 1));
      const __m512 delta0 = _mm512_sub_ps(q0, x0);
      const __m512 delta1 = _mm512_sub_ps(q1, x1);
      sums0[lane] = _mm512_fmadd_ps(delta0, delta0, sums0[lane]);
      sums1[lane] = _mm512_fmadd_ps(delta1, delta1, sums1[lane]);
    }
  }

  if (d + 16 <= dimension) {
    const __m512 q = _mm512_cvtph_ps(_mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(query + d)));
    for (size_t lane = 0; lane < Batch; ++lane) {
      const auto *row = static_cast<const uint16_t *>(vectors[lane]);
      const __m512 x = _mm512_cvtph_ps(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row + d)));
      const __m512 delta = _mm512_sub_ps(q, x);
      sums0[lane] = _mm512_fmadd_ps(delta, delta, sums0[lane]);
    }
    d += 16;
  }

  for (size_t lane = 0; lane < Batch; ++lane) {
    float sum = Reduce(_mm512_add_ps(sums0[lane], sums1[lane]));
    const auto *row = static_cast<const uint16_t *>(vectors[lane]);
    for (size_t tail = d; tail < dimension; ++tail) {
      const float delta =
          HalfToFloat(query[tail]) - HalfToFloat(row[tail]);
      sum += delta * delta;
    }
    distances[lane] = sum;
  }
}

inline void PrefetchRow(const void *vector) {
  // Match reimpl/vamana's FP16 refine policy: bring in the first cache line of
  // the next candidate and let sequential loads/hardware prefetch cover the
  // rest. Prefetching every line up front adds instruction pressure for the
  // small (20--40 row) rerank sets used by ann-benchmarks.
  _mm_prefetch(static_cast<const char *>(vector), _MM_HINT_T0);
}

}  // namespace

void squared_euclidean_fp16_batch_distance(
    const void **vectors, const void *query, size_t count, size_t dimension,
    float *distances, const void ** /*extra_values*/) {
  constexpr size_t kBatch = 4;
  const auto *query_fp16 = static_cast<const uint16_t *>(query);
  // Prefetch one four-row group ahead. At SIFT's 128 dimensions this covers
  // all four cache lines per row while the current group is computed.
  for (size_t i = 0; i < std::min(count, kBatch); ++i) {
    PrefetchRow(vectors[i]);
  }

  size_t i = 0;
  for (; i + kBatch <= count; i += kBatch) {
    for (size_t lane = 0; lane < kBatch; ++lane) {
      const size_t next = i + kBatch + lane;
      if (next < count) PrefetchRow(vectors[next]);
    }
    DistanceBatch<kBatch>(vectors + i, query_fp16, dimension, distances + i);
  }
  for (; i < count; ++i) {
    DistanceBatch<1>(vectors + i, query_fp16, dimension, distances + i);
  }
}

}  // namespace zvec::turbo::avx512_vnni
