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

#include "avx512_vnni/uniform_int8/squared_euclidean.h"

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

  // x86 cache line = 64B = 64 int8 values; we only need to issue one
  // prefetch per cache line, not per 32-element chunk. The inner loop
  // advances `d` in steps of 32, so the cache-line boundary is hit exactly
  // every other iteration (when (d & 63) == 0).
  size_t d = 0;
  for (; d + 32 <= dim; d += 32) {
    const bool prefetch_now = ((d & 63) == 0);

    // Load 32 query bytes and widen int8 → int16
    __m256i q_ymm =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q + d));
    __m512i q_zmm = _mm512_cvtepi8_epi16(q_ymm);

    for (size_t i = 0; i < batch_size; ++i) {
      // Prefetch the next cache line of a future vector at this offset.
      // Skipped on odd 32-byte chunks since the previous prefetch already
      // covered the whole 64-byte line.
      if (prefetch_now && prefetch_ptrs[i]) {
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

  // Use four independent accumulators to break the data-dependency chain
  // on `acc`. Both vpaddd and vpmaddwd have ~1-cycle latency but multi-port
  // throughput; with a single accumulator the next iteration must wait on
  // the previous add to retire. Four parallel chains let the OoO engine
  // dispatch one madd+add per cycle on Skylake-X / Ice Lake / Sapphire
  // Rapids, yielding ~1.5–2x speedup for dim >= 256.
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  __m512i acc2 = _mm512_setzero_si512();
  __m512i acc3 = _mm512_setzero_si512();

  size_t d = 0;

  // Main loop: process 128 bytes (4 × 32) per iteration, one chunk per acc.
  for (; d + 128 <= dim; d += 128) {
    __m512i diff0 = _mm512_sub_epi16(
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(lhs + d + 0))),
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(rhs + d + 0))));
    __m512i diff1 = _mm512_sub_epi16(
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(lhs + d + 32))),
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(rhs + d + 32))));
    __m512i diff2 = _mm512_sub_epi16(
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(lhs + d + 64))),
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(rhs + d + 64))));
    __m512i diff3 = _mm512_sub_epi16(
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(lhs + d + 96))),
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(rhs + d + 96))));

    acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(diff0, diff0));
    acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(diff1, diff1));
    acc2 = _mm512_add_epi32(acc2, _mm512_madd_epi16(diff2, diff2));
    acc3 = _mm512_add_epi32(acc3, _mm512_madd_epi16(diff3, diff3));
  }

  // Bridge loop: 32-byte chunks for the remaining (dim % 128) bytes.
  for (; d + 32 <= dim; d += 32) {
    __m512i diff = _mm512_sub_epi16(
        _mm512_cvtepi8_epi16(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs + d))),
        _mm512_cvtepi8_epi16(
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs + d))));
    acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(diff, diff));
  }

  // Reduce four accumulators -> one, then horizontally to a scalar.
  __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1),
                                 _mm512_add_epi32(acc2, acc3));
  int result = _mm512_reduce_add_epi32(acc);

  // Scalar tail (dim not a multiple of 32).
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
  // Tail (n % batch_size vectors): delegate to the single-vector kernel.
  // It already uses 4-way independent accumulators (see P1-2) and avoids
  // both an extra `batch_size=1` template instantiation and the per-call
  // std::array setup that the batch_impl path requires.
  for (; i < n; ++i) {
    uniform_squared_euclidean_int8_distance(vectors[i], query, dim,
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
