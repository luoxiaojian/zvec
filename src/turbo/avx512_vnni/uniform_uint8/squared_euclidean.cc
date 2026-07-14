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

// AVX512-VNNI optimized score for UNIFORM_UINT8.
//
// Stored record layout: [ dim int8 values = uint8(value) - 128 | int32 sum_sq ].
// Query layout:         [ dim raw uint8 values                         | int32 sum_sq ].
//
// Build-time pairwise distance uses true L2 between two stored shifted vectors.
// Search-time batch score drops the query-only constant from true uint8 L2:
//   score = sum_sq(v_raw) - 2 * dot(v_shifted, q_raw)
//         = ||v_raw - q_raw||^2 - query_constant
// This preserves ranking exactly while avoiding per-vector sum correction.
// VNNI uses vpdpbusd's unsigned x signed contract as:
//   dot(v_shifted, q_raw) = dpbusd(q_raw, v_shifted)
//
// Batch kernel design (hot path for graph search):
//   - 4 vectors per block, two-phase load/compute to maximize MLP
//   - software prefetch of future vectors INCLUDING the tail cache line
//     (the tail lives past the last 64B chunk, e.g. bytes 128..135 of a
//     192B-strided record for dim=128, i.e. a third cache line)
//   - no per-vector scalar epilogue: the 4 accumulators are reduced with a
//     single SIMD 4-to-1 horizontal reduction, the 4 tails are gathered with
//     SIMD, and the final distance formula is evaluated on int32x4 lanes:
//       score = sum_sq - 2*ip
//     (exact in int32 for any realistic dim; converted to float at the end)
//
// This file is compiled with per-file -march=avx512vnni (set in
// CMakeLists.txt).

#include "avx512_vnni/uniform_uint8/squared_euclidean.h"
#include "zvec/ailego/internal/platform.h"
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#include <array>
#endif
#include <cstdint>

namespace zvec::turbo::avx512_vnni {

namespace {

constexpr size_t kTailBytes = sizeof(int32_t);

static inline size_t original_dim(size_t dim) {
  return dim > kTailBytes ? dim - kTailBytes : 0;
}

static inline const int32_t *tail(const void *ptr, size_t orig_dim) {
  return reinterpret_cast<const int32_t *>(
      reinterpret_cast<const uint8_t *>(ptr) + orig_dim);
}

}  // namespace

void uniform_squared_euclidean_uint8_distance(const void *a, const void *b,
                                              size_t dim, float *distance) {
  const size_t orig_dim = original_dim(dim);
  if (orig_dim == 0) {
    *distance = 0.0f;
    return;
  }

  const auto *lhs = reinterpret_cast<const int8_t *>(a);
  const auto *rhs = reinterpret_cast<const int8_t *>(b);
  int64_t dist = 0;
  for (size_t i = 0; i < orig_dim; ++i) {
    int diff = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    dist += diff * diff;
  }
  *distance = static_cast<float>(dist);
}

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))

namespace {

// Reduce 4 zmm int32 accumulators to one xmm holding [s0, s1, s2, s3].
static ailego_force_inline __m128i reduce_add_4x16_epi32(__m512i a0,
                                                         __m512i a1,
                                                         __m512i a2,
                                                         __m512i a3) {
  __m256i b0 = _mm256_add_epi32(_mm512_castsi512_si256(a0),
                                _mm512_extracti64x4_epi64(a0, 1));
  __m256i b1 = _mm256_add_epi32(_mm512_castsi512_si256(a1),
                                _mm512_extracti64x4_epi64(a1, 1));
  __m256i b2 = _mm256_add_epi32(_mm512_castsi512_si256(a2),
                                _mm512_extracti64x4_epi64(a2, 1));
  __m256i b3 = _mm256_add_epi32(_mm512_castsi512_si256(a3),
                                _mm512_extracti64x4_epi64(a3, 1));
  __m256i c01 = _mm256_hadd_epi32(b0, b1);  // [s0 s0 s1 s1 | s0 s0 s1 s1]
  __m256i c23 = _mm256_hadd_epi32(b2, b3);
  __m256i d = _mm256_hadd_epi32(c01, c23);  // [s0 s1 s2 s3 | s0 s1 s2 s3]
  return _mm_add_epi32(_mm256_castsi256_si128(d),
                       _mm256_extracti128_si256(d, 1));
}

// Compute distances for exactly 4 vectors against the (already signed)
// query. `prefetch_ptrs[j]`, when non-null, points to a future vector whose
// full record (including the tail line) is prefetched.
static ailego_force_inline void uniform_sq_l2_uint8_batch4(
    const void *const *vectors, const uint8_t *raw_query,
    size_t orig_dim, const void *const *prefetch_ptrs,
    float *distances) {
  __m512i a0 = _mm512_setzero_si512();
  __m512i a1 = _mm512_setzero_si512();
  __m512i a2 = _mm512_setzero_si512();
  __m512i a3 = _mm512_setzero_si512();

  const auto *v0 = reinterpret_cast<const int8_t *>(vectors[0]);
  const auto *v1 = reinterpret_cast<const int8_t *>(vectors[1]);
  const auto *v2 = reinterpret_cast<const int8_t *>(vectors[2]);
  const auto *v3 = reinterpret_cast<const int8_t *>(vectors[3]);

  size_t d = 0;
  for (; d + 64 <= orig_dim; d += 64) {
    __m512i q = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(raw_query + d));
    __m512i r0 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v0 + d));
    __m512i r1 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v1 + d));
    __m512i r2 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v2 + d));
    __m512i r3 = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v3 + d));
    for (int j = 0; j < 4; ++j) {
      if (prefetch_ptrs[j]) {
        _mm_prefetch(reinterpret_cast<const char *>(prefetch_ptrs[j]) + d,
                     _MM_HINT_T0);
      }
    }
    a0 = _mm512_dpbusd_epi32(a0, q, r0);
    a1 = _mm512_dpbusd_epi32(a1, q, r1);
    a2 = _mm512_dpbusd_epi32(a2, q, r2);
    a3 = _mm512_dpbusd_epi32(a3, q, r3);
  }

  // Prefetch the tail cache line of future vectors; the main loop above only
  // covers full 64B chunks, and the 8-byte tail lands past them.
  for (int j = 0; j < 4; ++j) {
    if (prefetch_ptrs[j]) {
      _mm_prefetch(
          reinterpret_cast<const char *>(prefetch_ptrs[j]) + orig_dim,
          _MM_HINT_T0);
    }
  }

  __m128i ip4 = reduce_add_4x16_epi32(a0, a1, a2, a3);

  // Scalar remainder for orig_dim % 64 (dead code for dim=128).
  if (d < orig_dim) {
    alignas(16) int32_t tmp[4];
    _mm_store_si128(reinterpret_cast<__m128i *>(tmp), ip4);
    const int8_t *vecs[4] = {v0, v1, v2, v3};
    for (int j = 0; j < 4; ++j) {
      int32_t extra = 0;
      for (size_t k = d; k < orig_dim; ++k) {
        extra += static_cast<int>(vecs[j][k]) *
                 static_cast<int>(raw_query[k]);
      }
      tmp[j] += extra;
    }
    ip4 = _mm_load_si128(reinterpret_cast<const __m128i *>(tmp));
  }

  __m128i sqs =
      _mm_set_epi32(tail(v3, orig_dim)[0], tail(v2, orig_dim)[0],
                    tail(v1, orig_dim)[0], tail(v0, orig_dim)[0]);

  __m128i dist = _mm_sub_epi32(sqs, _mm_slli_epi32(ip4, 1));
  _mm_storeu_ps(distances, _mm_cvtepi32_ps(dist));
}

// Single-vector path against the (already signed) query, used for the
// n % 4 remainder of a batch.
static ailego_force_inline void uniform_sq_l2_uint8_single(
    const void *vector, const uint8_t *raw_query, size_t orig_dim,
    float *distance) {
  const auto *vec = reinterpret_cast<const int8_t *>(vector);
  __m512i acc = _mm512_setzero_si512();
  size_t d = 0;
  for (; d + 64 <= orig_dim; d += 64) {
    __m512i q = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(raw_query + d));
    __m512i v =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vec + d));
    acc = _mm512_dpbusd_epi32(acc, q, v);
  }
  int64_t dot = _mm512_reduce_add_epi32(acc);
  for (; d < orig_dim; ++d) {
    dot += static_cast<int>(vec[d]) * static_cast<int>(raw_query[d]);
  }
  const int32_t *vec_tail = tail(vector, orig_dim);
  *distance = static_cast<float>(static_cast<int64_t>(vec_tail[0]) - 2 * dot);
}

}  // namespace

#endif  // AVX512

template <bool QueryPreprocessed>
void uniform_squared_euclidean_uint8_batch_distance_impl(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const size_t orig_dim = original_dim(dim);
  if (orig_dim == 0) {
    return;
  }

  (void)QueryPreprocessed;
  const auto *raw_query = reinterpret_cast<const uint8_t *>(query);

  static constexpr size_t batch_size = 4;
  // Prefetch ~8 vectors ahead for small dims, ~4 ahead for large ones.
  const size_t prefetch_step = orig_dim > 256 ? 1 : 2;

  size_t i = 0;
  const void *pf[batch_size];
  for (; i + batch_size <= n; i += batch_size) {
    for (size_t j = 0; j < batch_size; ++j) {
      size_t pi = i + j + batch_size * prefetch_step;
      pf[j] = (pi < n) ? vectors[pi] : nullptr;
    }
    uniform_sq_l2_uint8_batch4(&vectors[i], raw_query, orig_dim, pf,
                               distances + i);
  }
  for (; i < n; ++i) {
    uniform_sq_l2_uint8_single(vectors[i], raw_query, orig_dim, distances + i);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void uniform_squared_euclidean_uint8_batch_distance(const void *const *vectors,
                                                    const void *query, size_t n,
                                                    size_t dim,
                                                    float *distances) {
  uniform_squared_euclidean_uint8_batch_distance_impl<false>(
      vectors, query, n, dim, distances);
}

#if ZVEC_UNIFORM_UINT8_QUERY_PREPROCESS
void uniform_squared_euclidean_uint8_preprocessed_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {
  uniform_squared_euclidean_uint8_batch_distance_impl<true>(
      vectors, query, n, dim, distances);
}

void uniform_squared_euclidean_uint8_query_preprocess(void *query, size_t dim) {
  (void)query;
  (void)dim;
}
#endif

}  // namespace zvec::turbo::avx512_vnni
