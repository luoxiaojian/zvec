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

// AVX512-VNNI optimized squared Euclidean distance for SIFT-quantized INT8.
//
// SIFT data has scale=1, so quantization is lossless for integer-valued
// inputs. Each database vector is stored as:
//   [ original_dim bytes: int8_t  (value = round(float) - 128) ]
//   [ 4 bytes float: sq_sum_half  (sum of original float squares / 2) ]
//
// The query is stored as pure uint8 (value = round(float)), no tail.
//
// Distance computation uses dpbusd(uint8_query, int8_data) to compute the
// inner product, then: dist = sq_sum_half - ip.
// This is a monotonic proxy for L2 ranking when the query norm is constant
// across all comparisons.
//
// This file is compiled with per-file -march=avx512vnni (set in
// src/turbo/CMakeLists.txt).

#include "avx512_vnni/sift_int8/squared_euclidean.h"

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
// Batch kernel: compute dpbusd inner product for `batch_size` database
// vectors against a single uint8 query, with software prefetching.
// Results are written as raw integer inner products (not yet combined with
// sq_sum_half).
// ---------------------------------------------------------------------------
template <size_t batch_size>
static TURBO_ALWAYS_INLINE void sift_ip_batch_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs,
    size_t original_dim, float *ip_results) {
  const int8_t *q = reinterpret_cast<const int8_t *>(query);

  __m512i accs[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    accs[i] = _mm512_setzero_si512();
  }

  size_t d = 0;
  for (; d + 64 <= original_dim; d += 64) {
    __m512i q_reg = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(q + d));

    for (size_t i = 0; i < batch_size; ++i) {
      if (prefetch_ptrs[i]) {
        _mm_prefetch(
            reinterpret_cast<const char *>(
                reinterpret_cast<const int8_t *>(prefetch_ptrs[i]) + d),
            _MM_HINT_T0);
      }
      __m512i v_reg = _mm512_loadu_si512(
          reinterpret_cast<const __m512i *>(
              reinterpret_cast<const int8_t *>(vectors[i]) + d));
      // dpbusd: treats first operand as uint8, second as int8
      accs[i] = _mm512_dpbusd_epi32(accs[i], q_reg, v_reg);
    }
  }

  std::array<int, batch_size> temp_results{};
  for (size_t i = 0; i < batch_size; ++i) {
    temp_results[i] = _mm512_reduce_add_epi32(accs[i]);
  }

  // Scalar tail for remaining elements
  for (; d < original_dim; ++d) {
    int qv = static_cast<int>(reinterpret_cast<const uint8_t *>(q)[d]);
    for (size_t i = 0; i < batch_size; ++i) {
      temp_results[i] +=
          qv *
          static_cast<int>(
              reinterpret_cast<const int8_t *>(vectors[i])[d]);
    }
  }

  for (size_t i = 0; i < batch_size; ++i) {
    ip_results[i] = static_cast<float>(temp_results[i]);
  }
}

// ---------------------------------------------------------------------------
// Public: single-vector distance
// ---------------------------------------------------------------------------
void sift_squared_euclidean_int8_distance(const void *database_vec,
                                          const void *query, size_t dim,
                                          float *distance) {
  const int original_dim = static_cast<int>(dim) - 4;
  if (original_dim <= 0) {
    return;
  }

  // Compute dpbusd inner product
  std::array<const void *, 1> prefetch_ptrs{nullptr};
  const void *vecs[1] = {database_vec};
  float ip_result;
  sift_ip_batch_impl<1>(query, vecs, prefetch_ptrs,
                         static_cast<size_t>(original_dim), &ip_result);

  // Read sq_sum_half from the database vector tail
  const float *tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(database_vec) + original_dim);
  float sq_sum_half = tail[0];

  *distance = sq_sum_half - ip_result;
}

// ---------------------------------------------------------------------------
// Public: batch distance
// ---------------------------------------------------------------------------
void sift_squared_euclidean_int8_batch_distance(const void *const *vectors,
                                                const void *query, size_t n,
                                                size_t dim,
                                                float *distances) {
  const int original_dim = static_cast<int>(dim) - 4;
  if (original_dim <= 0) {
    return;
  }

  static constexpr size_t batch_size = 4;
  static constexpr size_t prefetch_step = 2;

  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const void *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      size_t pi = i + j + batch_size * prefetch_step;
      prefetch_ptrs[j] = (pi < n) ? vectors[pi] : nullptr;
    }

    std::array<float, batch_size> ip_results;
    sift_ip_batch_impl<batch_size>(
        query, &vectors[i], prefetch_ptrs,
        static_cast<size_t>(original_dim), ip_results.data());

    // Combine with per-vector sq_sum_half
    for (size_t j = 0; j < batch_size; ++j) {
      const float *tail = reinterpret_cast<const float *>(
          reinterpret_cast<const int8_t *>(vectors[i + j]) + original_dim);
      float sq_sum_half = tail[0];
      distances[i + j] = sq_sum_half - ip_results[j];
    }
  }

  // Tail: remaining vectors
  for (; i < n; ++i) {
    sift_squared_euclidean_int8_distance(vectors[i], query, dim,
                                         distances + i);
  }
}

}  // namespace zvec::turbo::avx512_vnni

#else  // no AVX512 support

namespace zvec::turbo::avx512_vnni {

void sift_squared_euclidean_int8_distance(const void * /*database_vec*/,
                                          const void * /*query*/,
                                          size_t /*dim*/,
                                          float * /*distance*/) {}

void sift_squared_euclidean_int8_batch_distance(
    const void *const * /*vectors*/, const void * /*query*/, size_t /*n*/,
    size_t /*dim*/, float * /*distances*/) {}

}  // namespace zvec::turbo::avx512_vnni

#endif
