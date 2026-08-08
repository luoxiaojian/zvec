// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "avx512_vnni/raw_uint8/squared_euclidean.h"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace zvec::turbo::avx512_vnni {
namespace {

inline float ScalarDistance(const uint8_t *lhs, const uint8_t *rhs,
                            size_t dimension) {
  uint64_t sum = 0;
  for (size_t i = 0; i < dimension; ++i) {
    const int delta = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    sum += static_cast<uint64_t>(delta * delta);
  }
  return static_cast<float>(sum);
}

template <size_t Batch>
void DistanceBatch(const void **vectors, const uint8_t *query,
                   size_t dimension, float *distances) {
  // A signed 32-bit lane cannot overflow while dimension <= 32768. The DB's
  // target workloads (SIFT: 128) stay far below this threshold; retain a
  // correct scalar fallback for exceptionally wide vectors.
  if (dimension > 32768) {
    for (size_t lane = 0; lane < Batch; ++lane) {
      distances[lane] = ScalarDistance(
          static_cast<const uint8_t *>(vectors[lane]), query, dimension);
    }
    return;
  }

  __m512i sums[Batch];
  for (size_t lane = 0; lane < Batch; ++lane) {
    sums[lane] = _mm512_setzero_si512();
  }

  size_t offset = 0;
  for (; offset + 32 <= dimension; offset += 32) {
    const __m512i q = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + offset)));
    for (size_t lane = 0; lane < Batch; ++lane) {
      const auto *row = static_cast<const uint8_t *>(vectors[lane]);
      const __m512i x = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(row + offset)));
      const __m512i delta = _mm512_sub_epi16(x, q);
      sums[lane] =
          _mm512_add_epi32(sums[lane], _mm512_madd_epi16(delta, delta));
    }
  }

  for (size_t lane = 0; lane < Batch; ++lane) {
    int64_t sum = _mm512_reduce_add_epi32(sums[lane]);
    const auto *row = static_cast<const uint8_t *>(vectors[lane]);
    for (size_t i = offset; i < dimension; ++i) {
      const int delta = static_cast<int>(row[i]) - static_cast<int>(query[i]);
      sum += delta * delta;
    }
    distances[lane] = static_cast<float>(sum);
  }
}

inline void PrefetchRow(const void *vector) {
  _mm_prefetch(static_cast<const char *>(vector), _MM_HINT_T0);
}

}  // namespace

void fp32_to_raw_uint8(const float *input, size_t dimension, uint8_t *output) {
  size_t i = 0;
  for (; i + 16 <= dimension; i += 16) {
    const __m512 values = _mm512_loadu_ps(input + i);
    const __m512i integers = _mm512_cvttps_epi32(values);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i),
                     _mm512_cvtusepi32_epi8(integers));
  }
  for (; i < dimension; ++i) {
    output[i] = static_cast<uint8_t>(input[i]);
  }
}

void squared_euclidean_uint8_distance(const void *lhs, const void *rhs,
                                      size_t dimension, float *distance) {
  const void *rows[] = {lhs};
  DistanceBatch<1>(rows, static_cast<const uint8_t *>(rhs), dimension,
                   distance);
}

void squared_euclidean_uint8_batch_distance(
    const void **vectors, const void *query, size_t count, size_t dimension,
    float *distances, const void ** /*extra_values*/) {
  constexpr size_t kBatch = 4;
  const auto *query_uint8 = static_cast<const uint8_t *>(query);
  for (size_t i = 0; i < std::min(count, kBatch); ++i) {
    PrefetchRow(vectors[i]);
  }

  size_t i = 0;
  for (; i + kBatch <= count; i += kBatch) {
    for (size_t lane = 0; lane < kBatch; ++lane) {
      const size_t next = i + kBatch + lane;
      if (next < count) PrefetchRow(vectors[next]);
    }
    DistanceBatch<kBatch>(vectors + i, query_uint8, dimension, distances + i);
  }
  for (; i < count; ++i) {
    DistanceBatch<1>(vectors + i, query_uint8, dimension, distances + i);
  }
}

}  // namespace zvec::turbo::avx512_vnni
