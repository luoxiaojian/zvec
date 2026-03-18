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

// This file is compiled with per-file -march=avx512vnni (set in CMakeLists.txt)
// so that all AVX512-VNNI intrinsics and the inlined inner product kernels from
// avx512_inner_product.h are compiled with the correct target ISA.

#include "turbo/cosine/avx512_impl.h"
#include "turbo/inner_product/avx512_inner_product.h"
#if defined(__AVX512VNNI__)
#include <immintrin.h>
#endif

// Tail layout for quantized INT8 cosine vectors (matches the encoding in
// CosineMinusInnerProductDistanceBatchWithScoreUnquantized<int8_t>):
//
//   [ original_dim bytes: int8_t elements ]
//   [ float scale_a  ]  (ma)
//   [ float bias_a   ]  (mb)
//   [ float sum_a    ]  (ms)
//   [ int  int8_sum  ]  (sum of raw int8 elements, used when query is
//                        preprocessed to uint8 via +128 shift)
//
// The query tail has the same layout (qa, qb, qs) without int8_sum.
// Total tail size: 3 floats = 12 bytes for query; 3 floats + 1 int = 16 bytes
// for data vectors (but the dim passed in already accounts for the full
// encoded size, i.e. original_dim + 24 bytes, matching the +24 offset used in
// CosineMinusInnerProductDistanceBatchWithScoreUnquantized<int8_t>).

namespace zvec::turbo {

void cosine_int8_distance_avx512_vnni(const void *a, const void *b, int dim,
                                      float *distance) {
#if defined(__AVX512VNNI__)
  // `dim` is the full encoded size; the original vector occupies dim-24 bytes.
  const int original_dim = dim - 24;

  // Compute raw integer inner product over the original_dim bytes.
  // Note: for the single-vector path there is no query preprocessing, so both
  // sides are treated as int8_t (same as the non-preprocessed path in
  // MinusInnerProductDistanceBatchWithScoreUnquantized<int8_t>).
  internal::ip_int8_avx512_vnni(a, b, original_dim, distance);

  const float *a_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(a) + original_dim);
  const float *b_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(b) + original_dim);

  float ma = a_tail[0];
  float mb = a_tail[1];
  float ms = a_tail[2];

  float qa = b_tail[0];
  float qb = b_tail[1];
  float qs = b_tail[2];

  // Dequantize and compute cosine distance:
  //   cosine_dist = -(ma * qa * ip + mb * qa * qs + qb * ma * ms
  //                   + original_dim * qb * mb)
  *distance = -(ma * qa * *distance + mb * qa * qs + qb * ma * ms +
                static_cast<float>(original_dim) * qb * mb);
#else
  (void)a;
  (void)b;
  (void)dim;
  (void)distance;
#endif
}

void cosine_int8_batch_distance_avx512_vnni(const void *const *vectors,
                                            const void *query, int n, int dim,
                                            float *distances) {
#if defined(__AVX512VNNI__)
  // `dim` is the full encoded size; the original vector occupies dim-24 bytes.
  const int original_dim = dim - 24;

  // Compute raw inner products for all vectors. The query has been preprocessed
  // (int8 + 128 -> uint8) so dpbusd can be used via ip_int8_batch_avx512_vnni.
  internal::ip_int8_batch_avx512_vnni(vectors, query, n, original_dim,
                                      distances);

  const float *q_tail = reinterpret_cast<const float *>(
      reinterpret_cast<const int8_t *>(query) + original_dim);
  float qa = q_tail[0];
  float qb = q_tail[1];
  float qs = q_tail[2];

  for (int i = 0; i < n; ++i) {
    const float *m_tail = reinterpret_cast<const float *>(
        reinterpret_cast<const int8_t *>(vectors[i]) + original_dim);
    float ma = m_tail[0];
    float mb = m_tail[1];
    float ms = m_tail[2];
    // Correct for the +128 shift applied to the query during preprocessing:
    //   dpbusd computes sum(uint8_query[i] * int8_data[i])
    //         = sum((int8_query[i] + 128) * int8_data[i])
    //         = true_ip + 128 * sum(int8_data[i])
    // int8_sum is stored as the 5th int-sized field after the 3 floats.
    int int8_sum = reinterpret_cast<const int *>(m_tail)[4];
    float &result = distances[i];
    result -= 128.0f * static_cast<float>(int8_sum);

    // Dequantize and compute cosine distance:
    //   cosine_dist = -(ma * qa * ip + mb * qa * qs + qb * ma * ms
    //                   + original_dim * qb * mb)
    result = -(ma * qa * result + mb * qa * qs + qb * ma * ms +
               static_cast<float>(original_dim) * qb * mb);
  }
#else
  (void)vectors;
  (void)query;
  (void)n;
  (void)dim;
  (void)distances;
#endif
}

void cosine_int8_query_preprocess_avx512_vnni(void *query, size_t dim) {
#if defined(__AVX512VNNI__)
  // The original vector occupies dim-24 bytes; only those bytes are shifted.
  internal::shift_int8_to_uint8_avx512(query, static_cast<int>(dim) - 24);
#else
  (void)query;
  (void)dim;
#endif
}

}  // namespace zvec::turbo
