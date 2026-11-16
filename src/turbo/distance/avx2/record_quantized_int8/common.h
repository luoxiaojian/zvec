// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Shared AVX2 inner product kernel for record_quantized_int8 distance
// implementations (inner_product, squared_euclidean, cosine).

#pragma once

#if defined(__AVX2__)
#include <immintrin.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <zvec/ailego/internal/platform.h>
#include "sse2/simd.h"

namespace zvec::turbo::avx2::internal {

// Widen before adding lanes so the final sum can exceed the int32 range.
inline int64_t reduce_int32_lanes(__m256i value) {
  const __m256i wide = _mm256_add_epi64(
      _mm256_cvtepi32_epi64(_mm256_castsi256_si128(value)),
      _mm256_cvtepi32_epi64(_mm256_extracti128_si256(value, 1)));
  __m128i sum = _mm_add_epi64(_mm256_castsi256_si128(wide),
                            _mm256_extracti128_si256(wide, 1));
  sum = _mm_add_epi64(sum, _mm_unpackhi_epi64(sum, sum));
  return _mm_cvtsi128_si64(sum);
}

// Raw integer inner product of two int8 code arrays of length `dim`.
inline float raw_inner_product(const int8_t *a, const int8_t *b, size_t dim) {
  // Avoid partial-load, widening, and reduction setup for tiny vectors.
  if (dim < 8) {
    int32_t sum = 0;
    for (size_t i = 0; i < dim; ++i) {
      sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return static_cast<float>(sum);
  }
  __m256i accumulator = _mm256_setzero_si256();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m128i lhs_bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
    const __m128i rhs_bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    const __m256i lhs = _mm256_cvtepi8_epi16(lhs_bytes);
    const __m256i rhs = _mm256_cvtepi8_epi16(rhs_bytes);
    // Widened INT8 inputs allow a direct pairwise multiply-add into int32.
    accumulator =
        _mm256_add_epi32(accumulator, _mm256_madd_epi16(lhs, rhs));
  }

  if (i < dim) {
    // A few scalar products are cheaper than widening another SIMD block.
    if (dim - i < 4) {
      int32_t tail = static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
      if (dim - i >= 2) {
        tail += static_cast<int32_t>(a[i + 1]) * b[i + 1];
      }
      if (dim - i == 3) {
        tail += static_cast<int32_t>(a[i + 2]) * b[i + 2];
      }
      return static_cast<float>(tail + reduce_int32_lanes(accumulator));
    }
    const __m256i lhs =
        _mm256_cvtepi8_epi16(sse2::internal::load_partial_int8(a + i, dim - i));
    const __m256i rhs =
        _mm256_cvtepi8_epi16(sse2::internal::load_partial_int8(b + i, dim - i));
    accumulator =
        _mm256_add_epi32(accumulator, _mm256_madd_epi16(lhs, rhs));
  }

  return static_cast<float>(reduce_int32_lanes(accumulator));
}

// One-to-many raw inner product kernel: the query block is loaded and
// widened to int16 once per iteration and reused across all `dp_batch`
// records.
template <size_t dp_batch>
inline void raw_inner_product_batch_impl(
    const int8_t *query, const void *const *records,
    std::array<const int8_t *, dp_batch> &prefetch_ptrs, size_t dim,
    float *results) {
  __m256i accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_si256();
  }

  size_t d = 0;
  for (; d + 16 <= dim; d += 16) {
    const __m256i rhs = _mm256_cvtepi8_epi16(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(query + d)));

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + d);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      const __m256i lhs = _mm256_cvtepi8_epi16(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(
              static_cast<const int8_t *>(records[i]) + d)));
      accs[i] = _mm256_add_epi32(accs[i], _mm256_madd_epi16(lhs, rhs));
    }
  }

  if (d < dim) {
    const __m256i rhs = _mm256_cvtepi8_epi16(
        sse2::internal::load_partial_int8(query + d, dim - d));
    for (size_t i = 0; i < dp_batch; ++i) {
      if (prefetch_ptrs[i]) ailego_prefetch(prefetch_ptrs[i] + d);
      const __m256i lhs =
          _mm256_cvtepi8_epi16(sse2::internal::load_partial_int8(
              static_cast<const int8_t *>(records[i]) + d, dim - d));
      accs[i] = _mm256_add_epi32(accs[i], _mm256_madd_epi16(lhs, rhs));
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = static_cast<float>(reduce_int32_lanes(accs[i]));
  }
}

// Dispatch batched raw inner products over all `n` records with prefetching.
// The `int8_t` query type keeps this overload distinct from the int4 one.
inline void raw_inner_product_batch(const void *const *vectors,
                                    const int8_t *query, size_t n, size_t dim,
                                    float *results) {
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const int8_t *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = static_cast<const int8_t *>(
            vectors[i + j + batch_size * prefetch_step]);
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    raw_inner_product_batch_impl<batch_size>(query, &vectors[i], prefetch_ptrs,
                                             dim, results + i);
  }
  for (; i < n; i++) {
    std::array<const int8_t *, 1> prefetch_ptrs{nullptr};
    raw_inner_product_batch_impl<1>(query, &vectors[i], prefetch_ptrs, dim,
                                    results + i);
  }
}

}  // namespace zvec::turbo::avx2::internal

#endif  // defined(__AVX2__)
