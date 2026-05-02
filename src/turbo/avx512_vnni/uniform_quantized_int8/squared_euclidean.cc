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

// AVX512 optimized squared Euclidean distance for uniform-quantized INT8.
//
// Since all vectors share a single global scale/bias, the distance is simply:
//   sum((a[i] - b[i])^2)
// computed entirely in the integer domain.  No per-vector reconstruction or
// scalar dequantization is needed — this avoids the ALU frequency throttling
// that comes with mixing AVX512 and scalar instructions.
//
// Algorithm for each 32-element chunk:
//   1. Load 32 int8 values from each vector                (ymm load)
//   2. Sign-extend int8 → int16 in a zmm register          (vpmovsx)
//   3. Subtract int16 vectors: diff = a - b                 (vpsubw)
//   4. Square and pairwise-sum: diff[0]^2+diff[1]^2, ...   (vpmaddwd)
//   5. Accumulate int32 partial sums                        (vpaddd)
//
// This file is compiled with per-file -march=avx512vnni (set in
// CMakeLists.txt).

#include "avx512_vnni/uniform_quantized_int8/squared_euclidean.h"

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#include <array>
#include <cstdint>

#ifdef _MSC_VER
#define TURBO_ALWAYS_INLINE __forceinline
#else
#define TURBO_ALWAYS_INLINE __attribute__((always_inline))
#endif

namespace zvec::turbo::avx512_vnni {

// ---------------------------------------------------------------------------
// Batch kernel template: compute squared L2 for `batch_size` database vectors
// against a single query, with software prefetching of future vectors.
// ---------------------------------------------------------------------------
template <size_t batch_size>
static TURBO_ALWAYS_INLINE void uniform_sq_l2_int8_batch_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs, size_t dim,
    float *distances) {
  const int8_t *q = reinterpret_cast<const int8_t *>(query);

  __m512i accs[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    accs[i] = _mm512_setzero_si512();
  }

  size_t d = 0;
  for (; d + 32 <= dim; d += 32) {
    // Load 32 query bytes and widen int8 → int16
    __m256i q_ymm =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + d));
    __m512i q_zmm = _mm512_cvtepi8_epi16(q_ymm);

    for (size_t i = 0; i < batch_size; ++i) {
      // Prefetch a future vector's cache line at this offset
      if (prefetch_ptrs[i]) {
        _mm_prefetch(
            reinterpret_cast<const char *>(
                reinterpret_cast<const int8_t *>(prefetch_ptrs[i]) + d),
            _MM_HINT_T0);
      }
      // Load 32 database bytes and widen
      __m256i v_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
          reinterpret_cast<const int8_t *>(vectors[i]) + d));
      __m512i v_zmm = _mm512_cvtepi8_epi16(v_ymm);

      // diff = query - vec  (int16)
      __m512i diff = _mm512_sub_epi16(q_zmm, v_zmm);

      // Square and pairwise-accumulate:
      //   madd_epi16(diff, diff) → int32: d[0]²+d[1]², d[2]²+d[3]², …
      accs[i] = _mm512_add_epi32(accs[i], _mm512_madd_epi16(diff, diff));
    }
  }

  // Horizontal reduce each accumulator
  std::array<int, batch_size> results{};
  for (size_t i = 0; i < batch_size; ++i) {
    results[i] = _mm512_reduce_add_epi32(accs[i]);
  }

  // Handle remaining elements (dim not a multiple of 32)
  for (; d < dim; ++d) {
    int qv = static_cast<int>(q[d]);
    for (size_t i = 0; i < batch_size; ++i) {
      int diff = qv - static_cast<int>(
                          reinterpret_cast<const int8_t *>(vectors[i])[d]);
      results[i] += diff * diff;
    }
  }

  for (size_t i = 0; i < batch_size; ++i) {
    distances[i] = static_cast<float>(results[i]);
  }
}

// ---------------------------------------------------------------------------
// Public: single-vector squared Euclidean distance (int8, no tail)
// ---------------------------------------------------------------------------
void uniform_squared_euclidean_int8_distance(const void *a, const void *b,
                                             size_t dim, float *distance) {
  const int8_t *lhs = reinterpret_cast<const int8_t *>(a);
  const int8_t *rhs = reinterpret_cast<const int8_t *>(b);

  __m512i acc = _mm512_setzero_si512();

  size_t d = 0;
  for (; d + 32 <= dim; d += 32) {
    __m256i a_ymm =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs + d));
    __m256i b_ymm =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs + d));

    __m512i a_zmm = _mm512_cvtepi8_epi16(a_ymm);
    __m512i b_zmm = _mm512_cvtepi8_epi16(b_ymm);

    __m512i diff = _mm512_sub_epi16(a_zmm, b_zmm);
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(diff, diff));
  }

  int result = _mm512_reduce_add_epi32(acc);

  for (; d < dim; ++d) {
    int diff = static_cast<int>(lhs[d]) - static_cast<int>(rhs[d]);
    result += diff * diff;
  }

  *distance = static_cast<float>(result);
}

// ---------------------------------------------------------------------------
// Public: batch squared Euclidean distance (int8, no tail, no preprocessing)
// ---------------------------------------------------------------------------
void uniform_squared_euclidean_int8_batch_distance(const void *const *vectors,
                                                   const void *query, size_t n,
                                                   size_t dim,
                                                   float *distances) {
  static constexpr size_t batch_size = 4;
  static constexpr size_t prefetch_step = 2;

  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const void *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      size_t pi = i + j + batch_size * prefetch_step;
      prefetch_ptrs[j] = (pi < n) ? vectors[pi] : nullptr;
    }
    uniform_sq_l2_int8_batch_impl<batch_size>(query, &vectors[i], prefetch_ptrs,
                                              dim, distances + i);
  }
  // Handle remaining vectors one at a time
  for (; i < n; ++i) {
    std::array<const void *, 1> prefetch_ptrs{nullptr};
    uniform_sq_l2_int8_batch_impl<1>(query, &vectors[i], prefetch_ptrs, dim,
                                     distances + i);
  }
}

}  // namespace zvec::turbo::avx512_vnni

#else  // no AVX512 support

namespace zvec::turbo::avx512_vnni {

void uniform_squared_euclidean_int8_distance(const void * /*a*/,
                                             const void * /*b*/, size_t /*dim*/,
                                             float * /*distance*/) {}

void uniform_squared_euclidean_int8_batch_distance(
    const void *const * /*vectors*/, const void * /*query*/, size_t /*n*/,
    size_t /*dim*/, float * /*distances*/) {}

}  // namespace zvec::turbo::avx512_vnni

#endif
