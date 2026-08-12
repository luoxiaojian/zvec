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

// AVX512-VNNI optimized squared L2 for UNIFORM_UINT8.
//
// Stored inline record layout:
//   [ dim int8 values = uint8(value) - 128 | int32 sum_sq ]
// Preprocessed batch-query layout:
//   [ dim raw uint8 values | int32 query_distance_correction ]
//
// The batch kernel returns true squared L2. Its hot path adds the query-only
// correction to the unsigned/signed VNNI dot-product identity. Pairwise stored
// comparisons and unusually large vectors use widened squared differences.
// Four candidates are evaluated together to share query work and overlap
// independent memory streams.
//
// This file is compiled with per-file -march=avx512vnni (set in
// CMakeLists.txt).

#include "avx512_vnni/uniform_uint8/squared_euclidean.h"
#include "zvec/ailego/internal/platform.h"
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#endif
#include <cstdint>
#include <cstring>

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

static ailego_force_inline int32_t load_extra_sum_sq(const void *extra_values) {
  return *reinterpret_cast<const int32_t *>(extra_values);
}

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))

// Sign-extend 32 stored int8 values, subtract without int8 overflow, and use
// VNNI's signed-word dot product to accumulate pairs of squared differences
// into 16 int32 lanes.  The full stored range [-128, 127] produces differences
// in [-255, 255], so every int16 product and two-product int32 lane is exact.
static ailego_force_inline __m512i squared_diff_32(__m512i acc,
                                                   const int8_t *lhs,
                                                   const int8_t *rhs) {
  const __m512i lhs16 = _mm512_cvtepi8_epi16(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs)));
  const __m512i rhs16 = _mm512_cvtepi8_epi16(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs)));
  const __m512i diff16 = _mm512_sub_epi16(lhs16, rhs16);
  return _mm512_dpwssd_epi32(acc, diff16, diff16);
}

static ailego_force_inline __m512i squared_diff_masked_32(__m512i acc,
                                                          const int8_t *lhs,
                                                          const int8_t *rhs,
                                                          __mmask32 mask) {
  const __m512i lhs16 = _mm512_cvtepi8_epi16(
      _mm256_maskz_loadu_epi8(mask, static_cast<const void *>(lhs)));
  const __m512i rhs16 = _mm512_cvtepi8_epi16(
      _mm256_maskz_loadu_epi8(mask, static_cast<const void *>(rhs)));
  const __m512i diff16 = _mm512_sub_epi16(lhs16, rhs16);
  return _mm512_dpwssd_epi32(acc, diff16, diff16);
}

static ailego_force_inline __m512i squared_diff_32_with_rhs16(__m512i acc,
                                                              const int8_t *lhs,
                                                              __m512i rhs16) {
  const __m512i lhs16 = _mm512_cvtepi8_epi16(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs)));
  const __m512i diff16 = _mm512_sub_epi16(lhs16, rhs16);
  return _mm512_dpwssd_epi32(acc, diff16, diff16);
}

static ailego_force_inline __m512i squared_diff_masked_32_with_rhs16(
    __m512i acc, const int8_t *lhs, __m512i rhs16, __mmask32 mask) {
  const __m512i lhs16 = _mm512_cvtepi8_epi16(
      _mm256_maskz_loadu_epi8(mask, static_cast<const void *>(lhs)));
  const __m512i diff16 = _mm512_sub_epi16(lhs16, rhs16);
  return _mm512_dpwssd_epi32(acc, diff16, diff16);
}

// Widen before the horizontal sum.  A valid uint8 squared distance can exceed
// INT32_MAX even at moderate dimensions (e.g. 65,536 * 255^2), so reducing as
// int32 would silently wrap despite every individual accumulator lane being
// in range.
static ailego_force_inline int64_t reduce_add_epi32_to_int64(__m512i acc) {
  const __m256i lo32 = _mm512_castsi512_si256(acc);
  const __m256i hi32 = _mm512_extracti64x4_epi64(acc, 1);
  const __m512i lo64 = _mm512_cvtepi32_epi64(lo32);
  const __m512i hi64 = _mm512_cvtepi32_epi64(hi32);
  return _mm512_reduce_add_epi64(_mm512_add_epi64(lo64, hi64));
}

#endif

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

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  // Four independent dependency chains cover 128 bytes per iteration.  Each
  // VPDPWSSD lane receives two squares, whose maximum contribution is
  // 2 * 255^2 = 130,050.  Flush every 8,192 unrolled iterations so each int32
  // lane stays below 1.1 billion even for extremely long vectors.
  constexpr size_t kBlockBytes = 32;
  constexpr size_t kUnrolledBytes = 4 * kBlockBytes;
  constexpr size_t kFlushIterations = 8192;

  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  __m512i acc2 = _mm512_setzero_si512();
  __m512i acc3 = _mm512_setzero_si512();
  int64_t dist = 0;

  size_t d = 0;
  size_t iterations_since_flush = 0;
  for (; d + kUnrolledBytes <= orig_dim; d += kUnrolledBytes) {
    acc0 = squared_diff_32(acc0, lhs + d, rhs + d);
    acc1 = squared_diff_32(acc1, lhs + d + 32, rhs + d + 32);
    acc2 = squared_diff_32(acc2, lhs + d + 64, rhs + d + 64);
    acc3 = squared_diff_32(acc3, lhs + d + 96, rhs + d + 96);

    if (++iterations_since_flush == kFlushIterations) {
      dist += reduce_add_epi32_to_int64(acc0);
      dist += reduce_add_epi32_to_int64(acc1);
      dist += reduce_add_epi32_to_int64(acc2);
      dist += reduce_add_epi32_to_int64(acc3);
      acc0 = _mm512_setzero_si512();
      acc1 = _mm512_setzero_si512();
      acc2 = _mm512_setzero_si512();
      acc3 = _mm512_setzero_si512();
      iterations_since_flush = 0;
    }
  }

  for (; d + kBlockBytes <= orig_dim; d += kBlockBytes) {
    acc0 = squared_diff_32(acc0, lhs + d, rhs + d);
  }

  if (d < orig_dim) {
    const size_t remaining = orig_dim - d;
    const __mmask32 mask =
        static_cast<__mmask32>((uint32_t{1} << remaining) - 1);
    acc0 = squared_diff_masked_32(acc0, lhs + d, rhs + d, mask);
  }

  dist += reduce_add_epi32_to_int64(acc0);
  dist += reduce_add_epi32_to_int64(acc1);
  dist += reduce_add_epi32_to_int64(acc2);
  dist += reduce_add_epi32_to_int64(acc3);
  *distance = static_cast<float>(dist);
#else
  int64_t dist = 0;
  for (size_t i = 0; i < orig_dim; ++i) {
    const int diff = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    dist += diff * diff;
  }
  *distance = static_cast<float>(dist);
#endif
}

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))

namespace {

static ailego_force_inline __m512i load_query_32(const uint8_t *query) {
  const __m256i bytes =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query));
  return _mm512_sub_epi16(_mm512_cvtepu8_epi16(bytes), _mm512_set1_epi16(128));
}

static ailego_force_inline __m512i load_query_masked_32(const uint8_t *query,
                                                        __mmask32 mask) {
  const __m256i bytes =
      _mm256_maskz_loadu_epi8(mask, static_cast<const void *>(query));
  // Mask again after subtracting 128 so inactive lanes remain zero.
  return _mm512_maskz_mov_epi16(
      mask,
      _mm512_sub_epi16(_mm512_cvtepu8_epi16(bytes), _mm512_set1_epi16(128)));
}

// Compute exact squared L2 for four stored vectors and one preprocessed raw
// uint8 query. Periodic widening keeps unusually large dimensions exact.
static ailego_force_inline void uniform_sq_l2_uint8_batch4(
    const void *const *vectors, const void *query, size_t orig_dim,
    const void *const *prefetch_ptrs, float *distances) {
  const auto *v0 = reinterpret_cast<const int8_t *>(vectors[0]);
  const auto *v1 = reinterpret_cast<const int8_t *>(vectors[1]);
  const auto *v2 = reinterpret_cast<const int8_t *>(vectors[2]);
  const auto *v3 = reinterpret_cast<const int8_t *>(vectors[3]);
  const auto *query_bytes = reinterpret_cast<const uint8_t *>(query);

  __m512i a0 = _mm512_setzero_si512();
  __m512i a1 = _mm512_setzero_si512();
  __m512i a2 = _mm512_setzero_si512();
  __m512i a3 = _mm512_setzero_si512();
  int64_t totals[4] = {};

  constexpr size_t kBlockBytes = 32;
  constexpr size_t kFlushIterations = 8192;
  size_t iterations_since_flush = 0;
  size_t d = 0;
  for (; d + kBlockBytes <= orig_dim; d += kBlockBytes) {
    const __m512i query16 = load_query_32(query_bytes + d);
    for (size_t j = 0; j < 4; ++j) {
      if (prefetch_ptrs[j] != nullptr) {
        _mm_prefetch(reinterpret_cast<const char *>(prefetch_ptrs[j]) + d,
                     _MM_HINT_T0);
      }
    }
    a0 = squared_diff_32_with_rhs16(a0, v0 + d, query16);
    a1 = squared_diff_32_with_rhs16(a1, v1 + d, query16);
    a2 = squared_diff_32_with_rhs16(a2, v2 + d, query16);
    a3 = squared_diff_32_with_rhs16(a3, v3 + d, query16);

    if (++iterations_since_flush == kFlushIterations) {
      totals[0] += reduce_add_epi32_to_int64(a0);
      totals[1] += reduce_add_epi32_to_int64(a1);
      totals[2] += reduce_add_epi32_to_int64(a2);
      totals[3] += reduce_add_epi32_to_int64(a3);
      a0 = _mm512_setzero_si512();
      a1 = _mm512_setzero_si512();
      a2 = _mm512_setzero_si512();
      a3 = _mm512_setzero_si512();
      iterations_since_flush = 0;
    }
  }

  if (d < orig_dim) {
    const size_t remaining = orig_dim - d;
    const __mmask32 mask =
        static_cast<__mmask32>((uint32_t{1} << remaining) - 1);
    const __m512i query16 = load_query_masked_32(query_bytes + d, mask);
    a0 = squared_diff_masked_32_with_rhs16(a0, v0 + d, query16, mask);
    a1 = squared_diff_masked_32_with_rhs16(a1, v1 + d, query16, mask);
    a2 = squared_diff_masked_32_with_rhs16(a2, v2 + d, query16, mask);
    a3 = squared_diff_masked_32_with_rhs16(a3, v3 + d, query16, mask);
  }

  totals[0] += reduce_add_epi32_to_int64(a0);
  totals[1] += reduce_add_epi32_to_int64(a1);
  totals[2] += reduce_add_epi32_to_int64(a2);
  totals[3] += reduce_add_epi32_to_int64(a3);
  distances[0] = static_cast<float>(totals[0]);
  distances[1] = static_cast<float>(totals[1]);
  distances[2] = static_cast<float>(totals[2]);
  distances[3] = static_cast<float>(totals[3]);
}

// Reduce four zmm int32 accumulators to one xmm holding [s0, s1, s2, s3].
static ailego_force_inline __m128i reduce_add_4x16_epi32(__m512i a0, __m512i a1,
                                                         __m512i a2,
                                                         __m512i a3) {
  const __m256i b0 = _mm256_add_epi32(_mm512_castsi512_si256(a0),
                                      _mm512_extracti64x4_epi64(a0, 1));
  const __m256i b1 = _mm256_add_epi32(_mm512_castsi512_si256(a1),
                                      _mm512_extracti64x4_epi64(a1, 1));
  const __m256i b2 = _mm256_add_epi32(_mm512_castsi512_si256(a2),
                                      _mm512_extracti64x4_epi64(a2, 1));
  const __m256i b3 = _mm256_add_epi32(_mm512_castsi512_si256(a3),
                                      _mm512_extracti64x4_epi64(a3, 1));
  const __m256i c01 = _mm256_hadd_epi32(b0, b1);
  const __m256i c23 = _mm256_hadd_epi32(b2, b3);
  const __m256i d = _mm256_hadd_epi32(c01, c23);
  return _mm_add_epi32(_mm256_castsi256_si128(d),
                       _mm256_extracti128_si256(d, 1));
}

static ailego_force_inline int32_t query_distance_offset(
    int32_t query_correction) {
#if ZVEC_UNIFORM_UINT8_EXACT_QUERY_DISTANCE
  return query_correction;
#else
  (void)query_correction;
  return 0;
#endif
}

// SIFT contiguous-entity hot path: load the query's two cache lines once per
// batch call and reuse them for every group of four candidates.
static ailego_force_inline void uniform_raw_sq_l2_uint8_extra_batch4_128(
    const void *const *vectors, __m512i query0, __m512i query1,
    int32_t query_correction, const void *const *extra_values,
    float *distances) {
  const auto *v0 = reinterpret_cast<const __m512i *>(vectors[0]);
  const auto *v1 = reinterpret_cast<const __m512i *>(vectors[1]);
  const auto *v2 = reinterpret_cast<const __m512i *>(vectors[2]);
  const auto *v3 = reinterpret_cast<const __m512i *>(vectors[3]);

  __m512i a0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), query0,
                                   _mm512_loadu_si512(v0));
  __m512i a1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), query0,
                                   _mm512_loadu_si512(v1));
  __m512i a2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), query0,
                                   _mm512_loadu_si512(v2));
  __m512i a3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), query0,
                                   _mm512_loadu_si512(v3));
  a0 = _mm512_dpbusd_epi32(a0, query1, _mm512_loadu_si512(v0 + 1));
  a1 = _mm512_dpbusd_epi32(a1, query1, _mm512_loadu_si512(v1 + 1));
  a2 = _mm512_dpbusd_epi32(a2, query1, _mm512_loadu_si512(v2 + 1));
  a3 = _mm512_dpbusd_epi32(a3, query1, _mm512_loadu_si512(v3 + 1));

  const __m128i inner_products = reduce_add_4x16_epi32(a0, a1, a2, a3);
  const __m128i norms = _mm_set_epi32(
      load_extra_sum_sq(extra_values[3]), load_extra_sum_sq(extra_values[2]),
      load_extra_sum_sq(extra_values[1]), load_extra_sum_sq(extra_values[0]));
  const __m128i distance =
      _mm_add_epi32(_mm_sub_epi32(norms, _mm_slli_epi32(inner_products, 1)),
                    _mm_set1_epi32(query_distance_offset(query_correction)));
  _mm_storeu_ps(distances, _mm_cvtepi32_ps(distance));
}

template <bool HasExtraValues>
static ailego_force_inline void uniform_raw_sq_l2_uint8_batch4(
    const void *const *vectors, const uint8_t *query, size_t orig_dim,
    int32_t query_correction, const void *const *prefetch_ptrs,
    const void *const *extra_values, float *distances) {
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
    const __m512i q =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + d));
    const __m512i r0 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v0 + d));
    const __m512i r1 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v1 + d));
    const __m512i r2 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v2 + d));
    const __m512i r3 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(v3 + d));
    for (size_t j = 0; j < 4; ++j) {
      if (prefetch_ptrs[j] != nullptr) {
        _mm_prefetch(reinterpret_cast<const char *>(prefetch_ptrs[j]) + d,
                     _MM_HINT_T0);
      }
    }
    a0 = _mm512_dpbusd_epi32(a0, q, r0);
    a1 = _mm512_dpbusd_epi32(a1, q, r1);
    a2 = _mm512_dpbusd_epi32(a2, q, r2);
    a3 = _mm512_dpbusd_epi32(a3, q, r3);
  }

  __m128i inner_products = reduce_add_4x16_epi32(a0, a1, a2, a3);
  if (d < orig_dim) {
    alignas(16) int32_t values[4];
    _mm_store_si128(reinterpret_cast<__m128i *>(values), inner_products);
    const int8_t *stored_vectors[4] = {v0, v1, v2, v3};
    for (size_t i = 0; i < 4; ++i) {
      for (size_t k = d; k < orig_dim; ++k) {
        values[i] +=
            static_cast<int>(stored_vectors[i][k]) * static_cast<int>(query[k]);
      }
    }
    inner_products = _mm_load_si128(reinterpret_cast<const __m128i *>(values));
  }

  __m128i norms;
  if constexpr (HasExtraValues) {
    norms = _mm_set_epi32(
        load_extra_sum_sq(extra_values[3]), load_extra_sum_sq(extra_values[2]),
        load_extra_sum_sq(extra_values[1]), load_extra_sum_sq(extra_values[0]));
  } else {
    norms = _mm_set_epi32(tail(v3, orig_dim)[0], tail(v2, orig_dim)[0],
                          tail(v1, orig_dim)[0], tail(v0, orig_dim)[0]);
  }
  const __m128i distance =
      _mm_add_epi32(_mm_sub_epi32(norms, _mm_slli_epi32(inner_products, 1)),
                    _mm_set1_epi32(query_distance_offset(query_correction)));
  _mm_storeu_ps(distances, _mm_cvtepi32_ps(distance));
}

template <bool HasExtraValues>
static ailego_force_inline void uniform_raw_sq_l2_uint8_single(
    const void *vector, const uint8_t *query, size_t orig_dim,
    int32_t query_correction, const void *extra_value, float *distance) {
  const auto *stored = reinterpret_cast<const int8_t *>(vector);
  __m512i acc = _mm512_setzero_si512();
  size_t d = 0;
  for (; d + 64 <= orig_dim; d += 64) {
    const __m512i q =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + d));
    const __m512i v =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(stored + d));
    acc = _mm512_dpbusd_epi32(acc, q, v);
  }
  int64_t dot = _mm512_reduce_add_epi32(acc);
  for (; d < orig_dim; ++d) {
    dot += static_cast<int>(stored[d]) * static_cast<int>(query[d]);
  }
  const int32_t norm = HasExtraValues ? load_extra_sum_sq(extra_value)
                                      : tail(vector, orig_dim)[0];
  *distance = static_cast<float>(static_cast<int64_t>(norm) - 2 * dot +
                                 query_distance_offset(query_correction));
}

template <bool HasExtraValues>
static void uniform_raw_sq_l2_uint8_fast_batch(
    const void *const *vectors, const uint8_t *query, size_t n, size_t orig_dim,
    int32_t query_correction, float *distances,
    const void *const *extra_values) {
  constexpr size_t kBatchSize = 4;
  const size_t prefetch_step = orig_dim > 256 ? 1 : 2;
  const void *prefetch_ptrs[kBatchSize];

  size_t i = 0;
  if constexpr (HasExtraValues) {
    if (orig_dim == 128) {
      const __m512i query0 =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query));
      const __m512i query1 =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + 64));
      for (; i + kBatchSize <= n; i += kBatchSize) {
        uniform_raw_sq_l2_uint8_extra_batch4_128(
            &vectors[i], query0, query1, query_correction, &extra_values[i],
            distances + i);
      }
      for (; i < n; ++i) {
        uniform_raw_sq_l2_uint8_single<true>(vectors[i], query, orig_dim,
                                             query_correction, extra_values[i],
                                             distances + i);
      }
      return;
    }
  }

  for (; i + kBatchSize <= n; i += kBatchSize) {
    for (size_t j = 0; j < kBatchSize; ++j) {
      const size_t prefetch_index = i + j + kBatchSize * prefetch_step;
      prefetch_ptrs[j] = prefetch_index < n ? vectors[prefetch_index] : nullptr;
    }
    uniform_raw_sq_l2_uint8_batch4<HasExtraValues>(
        &vectors[i], query, orig_dim, query_correction, prefetch_ptrs,
        HasExtraValues ? &extra_values[i] : nullptr, distances + i);
  }
  for (; i < n; ++i) {
    uniform_raw_sq_l2_uint8_single<HasExtraValues>(
        vectors[i], query, orig_dim, query_correction,
        HasExtraValues ? extra_values[i] : nullptr, distances + i);
  }
}

}  // namespace

#endif  // AVX512

static void uniform_squared_euclidean_uint8_batch_distance_impl(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
  const size_t orig_dim = original_dim(dim);
  if (orig_dim == 0) {
    for (size_t i = 0; i < n; ++i) {
      distances[i] = 0.0f;
    }
    return;
  }

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  // The int32 dot-product identity is exact in this range. Above it, use the
  // widened squared-difference path, which does not depend on the int32 norm
  // stored in the record tail.
  constexpr size_t kMaxFastIdentityDimension = 8192;
  if (orig_dim <= kMaxFastIdentityDimension) {
    const auto *raw_query = reinterpret_cast<const uint8_t *>(query);
    const int32_t query_correction = tail(query, orig_dim)[0];
    if (extra_values != nullptr) {
      uniform_raw_sq_l2_uint8_fast_batch<true>(vectors, raw_query, n, orig_dim,
                                               query_correction, distances,
                                               extra_values);
    } else {
      uniform_raw_sq_l2_uint8_fast_batch<false>(vectors, raw_query, n, orig_dim,
                                                query_correction, distances,
                                                nullptr);
    }
    return;
  }

  constexpr size_t kBatchSize = 4;
  // Prefetch ~8 vectors ahead for small dims, ~4 ahead for large ones.
  const size_t prefetch_step = orig_dim > 256 ? 1 : 2;
  const void *prefetch_ptrs[kBatchSize];

  size_t i = 0;
  for (; i + kBatchSize <= n; i += kBatchSize) {
    for (size_t j = 0; j < kBatchSize; ++j) {
      const size_t prefetch_index = i + j + kBatchSize * prefetch_step;
      prefetch_ptrs[j] = prefetch_index < n ? vectors[prefetch_index] : nullptr;
    }
    uniform_sq_l2_uint8_batch4(&vectors[i], query, orig_dim, prefetch_ptrs,
                               distances + i);
  }
  for (; i < n; ++i) {
    const auto *vector = reinterpret_cast<const int8_t *>(vectors[i]);
    const auto *raw_query = reinterpret_cast<const uint8_t *>(query);
    int64_t dist = 0;
    for (size_t d = 0; d < orig_dim; ++d) {
      const int diff =
          static_cast<int>(vector[d]) - (static_cast<int>(raw_query[d]) - 128);
      dist += diff * diff;
    }
    distances[i] = static_cast<float>(dist);
  }
#else
  (void)extra_values;
  for (size_t i = 0; i < n; ++i) {
    const auto *vector = reinterpret_cast<const int8_t *>(vectors[i]);
    const auto *raw_query = reinterpret_cast<const uint8_t *>(query);
    int64_t dist = 0;
    for (size_t d = 0; d < orig_dim; ++d) {
      const int diff =
          static_cast<int>(vector[d]) - (static_cast<int>(raw_query[d]) - 128);
      dist += diff * diff;
    }
    distances[i] = static_cast<float>(dist);
  }
#endif
}

void uniform_squared_euclidean_uint8_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
  uniform_squared_euclidean_uint8_batch_distance_impl(vectors, query, n, dim,
                                                      distances, extra_values);
}

void uniform_squared_euclidean_uint8_query_preprocess(void *query, size_t dim) {
  const size_t orig_dim = original_dim(dim);
  auto *bytes = reinterpret_cast<uint8_t *>(query);
  uint64_t raw_sum = 0;
  size_t d = 0;

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const __m512i sign_bit = _mm512_set1_epi8(static_cast<char>(0x80));
  const __m512i zero = _mm512_setzero_si512();
  __m512i sums = _mm512_setzero_si512();
  for (; d + 64 <= orig_dim; d += 64) {
    const __m512i stored = _mm512_loadu_si512(bytes + d);
    const __m512i raw = _mm512_xor_si512(stored, sign_bit);
    _mm512_storeu_si512(bytes + d, raw);
    sums = _mm512_add_epi64(sums, _mm512_sad_epu8(raw, zero));
  }
  alignas(64) uint64_t lanes[8];
  _mm512_store_si512(lanes, sums);
  for (uint64_t lane : lanes) {
    raw_sum += lane;
  }
#endif

  for (; d < orig_dim; ++d) {
    bytes[d] ^= uint8_t{0x80};
    raw_sum += bytes[d];
  }

  int32_t sum_sq = 0;
  std::memcpy(&sum_sq, bytes + orig_dim, sizeof(sum_sq));
  const int32_t correction = static_cast<int32_t>(
      static_cast<int64_t>(sum_sq) - 256 * static_cast<int64_t>(raw_sum));
  std::memcpy(bytes + orig_dim, &correction, sizeof(correction));
}

}  // namespace zvec::turbo::avx512_vnni
