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

#include "avx512_vnni/uniform_uint8/squared_euclidean.h"
#include "zvec/ailego/internal/platform.h"
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#include <array>
#endif
#include <cstdint>

namespace zvec::turbo::avx512_vnni {

namespace {

constexpr size_t kTailBytes = sizeof(int32_t) * 2;

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

  const auto *lhs = reinterpret_cast<const uint8_t *>(a);
  const auto *rhs = reinterpret_cast<const uint8_t *>(b);
  int64_t dot = 0;
  for (size_t i = 0; i < orig_dim; ++i) {
    dot += static_cast<int>(lhs[i]) * static_cast<int>(rhs[i]);
  }

  const int32_t *lhs_tail = tail(a, orig_dim);
  const int32_t *rhs_tail = tail(b, orig_dim);
  const int64_t lhs_sum_sq = lhs_tail[1];
  const int64_t rhs_sum_sq = rhs_tail[1];
  *distance = static_cast<float>(lhs_sum_sq + rhs_sum_sq - 2 * dot);
}

template <bool QueryPreprocessed>
void uniform_squared_euclidean_uint8_batch_distance_impl(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const size_t orig_dim = original_dim(dim);
  if (orig_dim == 0) {
    return;
  }

  const int32_t *query_tail = tail(query, orig_dim);
  const int32_t query_sum_sq = query_tail[1];
  static constexpr size_t batch_size = 2;
  const size_t prefetch_step = orig_dim > 256 ? 2 : 4;
  const auto *raw_query = reinterpret_cast<const uint8_t *>(query);
  const auto *signed_query = reinterpret_cast<const int8_t *>(query);
  const __m512i offset = _mm512_set1_epi8(static_cast<int8_t>(128));

  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    __m512i accs[batch_size];
    for (size_t j = 0; j < batch_size; ++j) {
      accs[j] = _mm512_setzero_si512();
    }

    size_t d = 0;
    for (; d + 64 <= orig_dim; d += 64) {
      __m512i q = _mm512_loadu_si512(
          reinterpret_cast<const __m512i *>(
              (QueryPreprocessed ? reinterpret_cast<const uint8_t *>(
                                       signed_query)
                                 : raw_query) +
              d));
      if constexpr (!QueryPreprocessed) {
        q = _mm512_sub_epi8(q, offset);
      }
      __m512i data_regs[batch_size];
      for (size_t j = 0; j < batch_size; ++j) {
        data_regs[j] = _mm512_loadu_si512(
            reinterpret_cast<const __m512i *>(
                reinterpret_cast<const uint8_t *>(vectors[i + j]) + d));
      }
      for (size_t j = 0; j < batch_size; ++j) {
        if (i + j + batch_size * prefetch_step < n) {
          _mm_prefetch(
              reinterpret_cast<const char *>(vectors[i + j + batch_size * prefetch_step]) +
                  d,
              _MM_HINT_T0);
        }
        accs[j] = _mm512_dpbusd_epi32(accs[j], data_regs[j], q);
      }
    }

    for (size_t j = 0; j < batch_size; ++j) {
      int64_t ip_shifted = _mm512_reduce_add_epi32(accs[j]);
      const auto *vec = reinterpret_cast<const uint8_t *>(vectors[i + j]);
      for (size_t d_tail = d; d_tail < orig_dim; ++d_tail) {
        const int qv =
            QueryPreprocessed ? static_cast<int>(signed_query[d_tail])
                              : static_cast<int>(raw_query[d_tail]) - 128;
        ip_shifted += static_cast<int>(vec[d_tail]) * qv;
      }
      const int32_t *vec_tail = tail(vectors[i + j], orig_dim);
      const int64_t vec_sum = vec_tail[0];
      const int64_t vec_sum_sq = vec_tail[1];
      const int64_t dot = ip_shifted + 128 * vec_sum;
      distances[i + j] =
          static_cast<float>(vec_sum_sq + query_sum_sq - 2 * dot);
    }
  }

  for (; i < n; ++i) {
    __m512i acc = _mm512_setzero_si512();
    size_t d = 0;
    for (; d + 64 <= orig_dim; d += 64) {
      __m512i q = _mm512_loadu_si512(
          reinterpret_cast<const __m512i *>(
              (QueryPreprocessed ? reinterpret_cast<const uint8_t *>(
                                       signed_query)
                                 : raw_query) +
              d));
      if constexpr (!QueryPreprocessed) {
        q = _mm512_sub_epi8(q, offset);
      }
      __m512i v = _mm512_loadu_si512(
          reinterpret_cast<const __m512i *>(
              reinterpret_cast<const uint8_t *>(vectors[i]) + d));
      acc = _mm512_dpbusd_epi32(acc, v, q);
    }
    int64_t ip_shifted = _mm512_reduce_add_epi32(acc);
    const auto *vec = reinterpret_cast<const uint8_t *>(vectors[i]);
    for (; d < orig_dim; ++d) {
      const int qv = QueryPreprocessed ? static_cast<int>(signed_query[d])
                                       : static_cast<int>(raw_query[d]) - 128;
      ip_shifted += static_cast<int>(vec[d]) * qv;
    }
    const int32_t *vec_tail = tail(vectors[i], orig_dim);
    const int64_t dot = ip_shifted + 128 * static_cast<int64_t>(vec_tail[0]);
    distances[i] =
        static_cast<float>(static_cast<int64_t>(vec_tail[1]) + query_sum_sq -
                           2 * dot);
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
#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const size_t orig_dim = original_dim(dim);
  auto *bytes = reinterpret_cast<uint8_t *>(query);
  const __m512i offset = _mm512_set1_epi8(static_cast<int8_t>(128));
  size_t i = 0;
  for (; i + 64 <= orig_dim; i += 64) {
    __m512i data =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(bytes + i));
    __m512i shifted = _mm512_sub_epi8(data, offset);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(bytes + i), shifted);
  }
  for (; i < orig_dim; ++i) {
    bytes[i] = static_cast<uint8_t>(static_cast<int>(bytes[i]) - 128);
  }
#else
  (void)query;
  (void)dim;
#endif
}
#endif

}  // namespace zvec::turbo::avx512_vnni
