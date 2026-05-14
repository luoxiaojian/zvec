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

// AVX512-VNNI optimized cosine distance for uniform-quantized INT8.
//
// Since all vectors share a single global scale/bias and values are in
// [0, 127], cosine distance reduces to the negative inner product:
//   cosine_dist = -sum(a[i] * b[i])
// computed entirely in the integer domain via VNNI dpbusd.
//
// Because values are in [0, 127], they fit in both uint8 and int8, so
// dpbusd(uint8, int8) can be used directly without any +128 shift.
//
// This file is compiled with per-file -march=avx512vnni (set in
// CMakeLists.txt).

#include "avx512_vnni/uniform_int8/cosine.h"

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
// Batch kernel template: compute negative inner product for `batch_size`
// database vectors against a single query, with software prefetching.
//
// Uses VNNI dpbusd: acc += sum_4(uint8(a[j]) * int8(b[j])) per 32-bit lane.
// Since values are in [0, 127], int8 values are valid as both uint8 and int8
// operands for dpbusd.
//
// Two-phase load/compute: load ALL vectors first, then compute (allows CPU
// to issue multiple loads in parallel, hiding memory latency).
// ---------------------------------------------------------------------------
template <size_t batch_size>
static TURBO_ALWAYS_INLINE void uniform_cosine_int8_batch_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs, size_t dim,
    float *distances) {
  const int8_t *q = reinterpret_cast<const int8_t *>(query);

  __m512i accs[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    accs[i] = _mm512_setzero_si512();
  }

  // Process 64 bytes (one cache line) per iteration.
  size_t d = 0;
  for (; d + 64 <= dim; d += 64) {
    // Load 64 query bytes (treated as uint8 for dpbusd first operand)
    __m512i q_zmm =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(q + d));

    // Phase 1: load all data vectors into registers first
    __m512i data_regs[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
      data_regs[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
          reinterpret_cast<const int8_t *>(vectors[i]) + d));
    }

    // Phase 2: prefetch + compute (data already in registers)
    for (size_t i = 0; i < batch_size; ++i) {
      if (prefetch_ptrs[i]) {
        _mm_prefetch(
            reinterpret_cast<const char *>(
                reinterpret_cast<const int8_t *>(prefetch_ptrs[i]) + d),
            _MM_HINT_T0);
      }
      // dpbusd: acc += sum_4(uint8(q) * int8(data))
      // Values in [0, 127] are valid for both uint8 and int8 interpretation.
      accs[i] = _mm512_dpbusd_epi32(accs[i], q_zmm, data_regs[i]);
    }
  }

  // Horizontal reduce each accumulator
  std::array<int, batch_size> results{};
  for (size_t i = 0; i < batch_size; ++i) {
    results[i] = _mm512_reduce_add_epi32(accs[i]);
  }

  // Handle remaining elements (dim not a multiple of 64)
  for (; d < dim; ++d) {
    int qv = static_cast<int>(q[d]);
    for (size_t i = 0; i < batch_size; ++i) {
      results[i] +=
          qv * static_cast<int>(
                   reinterpret_cast<const int8_t *>(vectors[i])[d]);
    }
  }

  // Negate: cosine distance = -inner_product
  for (size_t i = 0; i < batch_size; ++i) {
    distances[i] = -static_cast<float>(results[i]);
  }
}

// ---------------------------------------------------------------------------
// Public: single-vector cosine distance (int8, VNNI dpbusd)
// ---------------------------------------------------------------------------
void uniform_cosine_int8_distance(const void *a, const void *b, size_t dim,
                                  float *distance) {
  const int8_t *lhs = reinterpret_cast<const int8_t *>(a);
  const int8_t *rhs = reinterpret_cast<const int8_t *>(b);

  // Four independent accumulators to break the data-dependency chain.
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  __m512i acc2 = _mm512_setzero_si512();
  __m512i acc3 = _mm512_setzero_si512();

  size_t d = 0;

  // Main loop: process 256 bytes (4 × 64) per iteration.
  for (; d + 256 <= dim; d += 256) {
    __m512i a0 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d + 0));
    __m512i b0 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d + 0));
    __m512i a1 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d + 64));
    __m512i b1 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d + 64));
    __m512i a2 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d + 128));
    __m512i b2 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d + 128));
    __m512i a3 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d + 192));
    __m512i b3 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d + 192));

    acc0 = _mm512_dpbusd_epi32(acc0, a0, b0);
    acc1 = _mm512_dpbusd_epi32(acc1, a1, b1);
    acc2 = _mm512_dpbusd_epi32(acc2, a2, b2);
    acc3 = _mm512_dpbusd_epi32(acc3, a3, b3);
  }

  // Bridge loop: 64-byte chunks for the remaining (dim % 256) bytes.
  for (; d + 64 <= dim; d += 64) {
    __m512i va =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d));
    __m512i vb =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d));
    acc0 = _mm512_dpbusd_epi32(acc0, va, vb);
  }

  // Reduce four accumulators -> one, then horizontally to a scalar.
  __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1),
                                 _mm512_add_epi32(acc2, acc3));
  int result = _mm512_reduce_add_epi32(acc);

  // Scalar tail (dim not a multiple of 64).
  for (; d < dim; ++d) {
    result += static_cast<int>(lhs[d]) * static_cast<int>(rhs[d]);
  }

  // Negate: cosine distance = -inner_product
  *distance = -static_cast<float>(result);
}

// ---------------------------------------------------------------------------
// Public: batch cosine distance (int8, no tail, no preprocessing)
// ---------------------------------------------------------------------------
void uniform_cosine_int8_batch_distance(const void *const *vectors,
                                        const void *query, size_t n,
                                        size_t dim, float *distances) {
  static constexpr size_t batch_size = 4;
  static constexpr size_t prefetch_step = 2;

  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const void *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      size_t pi = i + j + batch_size * prefetch_step;
      prefetch_ptrs[j] = (pi < n) ? vectors[pi] : nullptr;
    }
    uniform_cosine_int8_batch_impl<batch_size>(query, &vectors[i],
                                               prefetch_ptrs, dim,
                                               distances + i);
  }
  // Tail (n % batch_size vectors): delegate to the single-vector kernel.
  for (; i < n; ++i) {
    uniform_cosine_int8_distance(vectors[i], query, dim, distances + i);
  }
}

}  // namespace zvec::turbo::avx512_vnni

#else  // no AVX512 support

namespace zvec::turbo::avx512_vnni {

void uniform_cosine_int8_distance(const void * /*a*/, const void * /*b*/,
                                  size_t /*dim*/, float * /*distance*/) {}

void uniform_cosine_int8_batch_distance(const void *const * /*vectors*/,
                                        const void * /*query*/, size_t /*n*/,
                                        size_t /*dim*/,
                                        float * /*distances*/) {}

}  // namespace zvec::turbo::avx512_vnni

#endif
