#include "turbo/euclidean/avx512_impl.h"
#include "turbo/inner_product/avx512_inner_product.h"
#if defined(__AVX512VNNI__)
#include <immintrin.h>
#endif

namespace zvec::turbo {

void l2_int8_distance_avx512_vnni(const void *a, const void *b, int dim,
                                  float *distance) {
#if defined(__AVX512VNNI__)
  const int d = dim - 20;
  internal::ip_int8_avx512_vnni(a, b, d, distance);

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

void l2_int8_batch_distance_avx512_vnni(const void *const *vectors,
                                        const void *query, int n, int dim,
                                        float *distances) {
#if defined(__AVX512VNNI__)
  int original_dim = dim - 20;

  internal::ip_int8_batch_avx512_vnni(vectors, query, n, original_dim,
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
  internal::shift_int8_to_uint8_avx512(query, static_cast<int>(dim) - 20);
#else
  (void)query;
  (void)dim;
#endif
}

}  // namespace zvec::turbo
