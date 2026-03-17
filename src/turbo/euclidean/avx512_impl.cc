#include "turbo/euclidean/avx512_impl.h"
#if defined(__AVX512VNNI__)
#include <immintrin.h>
#endif
#include <array>

namespace zvec::turbo {

#if defined(__AVX512VNNI__)
static inline int32_t HorizontalAdd_INT32_V256(__m256i v) {
  __m256i x1 = _mm256_hadd_epi32(v, v);
  __m256i x2 = _mm256_hadd_epi32(x1, x1);
  __m128i x3 = _mm256_extractf128_si256(x2, 1);
  __m128i x4 = _mm_add_epi32(_mm256_castsi256_si128(x2), x3);
  return _mm_cvtsi128_si32(x4);
}

#define FMA_INT8_GENERAL(m, q, sum) sum += static_cast<float>(m * q);

// This is done to align with the previous behavior
// (DistanceMatrixCompute<SquaredEuclidean, int8_t>), where SquaredEuclidean
// assumes no preprocessing on the query, and both the query and data are of
// type int8_t.
static __attribute__((always_inline)) void ip_int8_distance_avx512_vnni(
    const void *a, const void *b, int size, float *distance) {
  const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);
  const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);

  const int8_t *lhs = reinterpret_cast<const int8_t *>(a);
  const int8_t *rhs = reinterpret_cast<const int8_t *>(b);

  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 6) << 6);

  float result = 0.0f;

  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_load_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_load_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }
  result = static_cast<float>(
      HorizontalAdd_INT32_V256(_mm256_add_epi32(ymm_sum_0, ymm_sum_1)));

  switch (last - lhs) {
    case 15:
      FMA_INT8_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      FMA_INT8_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      FMA_INT8_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      FMA_INT8_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      FMA_INT8_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      FMA_INT8_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      FMA_INT8_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      FMA_INT8_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      FMA_INT8_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_INT8_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_INT8_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_INT8_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_INT8_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_INT8_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_INT8_GENERAL(lhs[0], rhs[0], result)
  }
  *distance = result;
}
#endif

void l2_int8_distance_avx512_vnni(const void *a, const void *b, int dim,
                                  float *distance) {
#if defined(__AVX512VNNI__)
  const int d = dim - 20;
  ip_int8_distance_avx512_vnni(a, b, d, distance);

  const float *a_tail =
      reinterpret_cast<const float *>(reinterpret_cast<const int8_t *>(a) + d);
  const float *b_tail =
      reinterpret_cast<const float *>(reinterpret_cast<const int8_t *>(b) + d);

  float qa = b_tail[0];
  float qb = b_tail[1];
  float qs = b_tail[2];
  float qs2 = b_tail[3];

  const float sum = qa * qs;
  const float sum2 = qa * qa * qs2;

  float ma = a_tail[0];
  float mb = a_tail[1];
  float ms = a_tail[2];
  float ms2 = a_tail[3];

  *distance = ma * ma * ms2 + sum2 - 2 * ma * qa * *distance +
              (mb - qb) * (mb - qb) * d + 2 * (mb - qb) * (ms * ma - sum);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

#if defined(__AVX512VNNI__)
template <int batch_size>
__attribute__((always_inline)) void ip_int8_batch_distance_avx512_vnni_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs,
    int dimensionality, float *distances) {
  __m512i accs[batch_size];
  for (int i = 0; i < batch_size; ++i) {
    accs[i] = _mm512_setzero_si512();
  }
  int dim = 0;
  for (; dim + 64 <= dimensionality; dim += 64) {
    __m512i q = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
        reinterpret_cast<const int8_t *>(query) + dim));
    __m512i data_regs[batch_size];
    for (int i = 0; i < batch_size; ++i) {
      data_regs[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
          reinterpret_cast<const int8_t *>(vectors[i]) + dim));
    }
    if (prefetch_ptrs[0]) {
      for (int i = 0; i < batch_size; ++i) {
        _mm_prefetch(
            reinterpret_cast<const char *>(
                reinterpret_cast<const int8_t *>(prefetch_ptrs[i]) + dim),
            _MM_HINT_T0);
      }
    }
    for (int i = 0; i < batch_size; ++i) {
      accs[i] = _mm512_dpbusd_epi32(accs[i], q, data_regs[i]);
    }
  }
  std::array<int, batch_size> temp_results{};
  for (int i = 0; i < batch_size; ++i) {
    temp_results[i] = _mm512_reduce_add_epi32(accs[i]);
  }
  for (; dim < dimensionality; ++dim) {
    int q = static_cast<int>(reinterpret_cast<const uint8_t *>(query)[dim]);
    for (int i = 0; i < batch_size; ++i) {
      temp_results[i] +=
          q *
          static_cast<int>(reinterpret_cast<const int8_t *>(vectors[i])[dim]);
    }
  }
  for (int i = 0; i < batch_size; ++i) {
    distances[i] = static_cast<float>(temp_results[i]);
  }
}

static __attribute__((always_inline)) void ip_int8_batch_distance_avx512_vnni(
    const void *const *vectors, const void *query, int n, int dim,
    float *distances) {
  static constexpr int batch_size = 2;
  static constexpr int prefetch_step = 2;
  int i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const void *, batch_size> prefetch_ptrs;
    for (int j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = vectors[i + j + batch_size * prefetch_step];
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    ip_int8_batch_distance_avx512_vnni_impl<batch_size>(
        query, &vectors[i], prefetch_ptrs, dim, distances + i);
  }
  for (; i < n; i++) {
    std::array<const void *, 1> prefetch_ptrs{nullptr};
    ip_int8_batch_distance_avx512_vnni_impl<1>(
        query, &vectors[i], prefetch_ptrs, dim, distances + i);
  }
}
#endif

void l2_int8_batch_distance_avx512_vnni(const void *const *vectors,
                                        const void *query, int n, int dim,
                                        float *distances) {
#if defined(__AVX512VNNI__)
  int original_dim = dim - 20;

  ip_int8_batch_distance_avx512_vnni(vectors, query, n, original_dim,
                                     distances);
  const float *q_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(query) + original_dim);
  float qa = q_tail[0];
  float qb = q_tail[1];
  float qs = q_tail[2];
  float qs2 = q_tail[3];

  const float sum = qa * qs;
  const float sum2 = qa * qa * qs2;
  for (int i = 0; i < n; ++i) {
    const float *m_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const int8_t *>(vectors[i]) + original_dim);
    float ma = m_tail[0];
    float mb = m_tail[1];
    float ms = m_tail[2];
    float ms2 = m_tail[3];
    int int8_sum = reinterpret_cast<const int *>(m_tail)[4];
    float &result = distances[i];
    result -= 128 * int8_sum;
    result = ma * ma * ms2 + sum2 - 2 * ma * qa * result +
             (mb - qb) * (mb - qb) * original_dim +
             2 * (mb - qb) * (ms * ma - sum);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void l2_int8_query_preprocess_avx512_vnni(void *query, size_t dim) {
#if defined(__AVX512VNNI__)
  int d = dim - 20;

  const int8_t *input = reinterpret_cast<const int8_t *>(query);
  uint8_t *output = reinterpret_cast<uint8_t *>(query);

  // AVX512 constant: 128 in each byte (cast to int8_t, which becomes -128
  // in signed representation, but addition works correctly due to two's
  // complement arithmetic)
  const __m512i offset = _mm512_set1_epi8(static_cast<int8_t>(128));

  int i = 0;
  // Process 64 bytes at a time using AVX512
  for (; i + 64 <= d; i += 64) {
    __m512i data =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    __m512i result = _mm512_add_epi8(data, offset);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), result);
  }

  // Handle remaining elements with scalar loop
  for (; i < d; ++i) {
    output[i] = static_cast<uint8_t>(static_cast<int>(input[i]) + 128);
  }
#else
  (void)query;
  (void)dim;
#endif
}

}  // namespace zvec::turbo
