// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "avx512_vnni/fp16/squared_euclidean.h"
#include <immintrin.h>
#include <algorithm>
#include <cstdint>

namespace zvec::turbo::avx512_vnni {
namespace {

inline float Reduce(__m512 value) {
  // Match the reduction tree used by reimpl/vamana and KGN.
  const __m256 low256 = _mm512_castps512_ps256(value);
  const __m256 high256 = _mm512_extractf32x8_ps(value, 1);
  const __m256 sum256 = _mm256_add_ps(low256, high256);
  const __m128 low128 = _mm256_castps256_ps128(sum256);
  const __m128 high128 = _mm256_extractf128_ps(sum256, 1);
  const __m128 sum128 = _mm_add_ps(low128, high128);
  const __m128 pair = _mm_hadd_ps(sum128, sum128);
  return _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55)));
}

inline float HalfToFloat(uint16_t value) {
  return _cvtsh_ss(value);
}

template <size_t Batch, bool PrefetchNext = false>
void DistanceBatch(const void **vectors, const uint16_t *query,
                   size_t dimension, float *distances,
                   const void **prefetch_vectors = nullptr,
                   size_t prefetch_count = 0) {
  __m512 sums0[Batch];
  __m512 sums1[Batch];
  for (size_t lane = 0; lane < Batch; ++lane) {
    sums0[lane] = _mm512_setzero_ps();
    sums1[lane] = _mm512_setzero_ps();
  }

  size_t d = 0;
  for (; d + 32 <= dimension; d += 32) {
    // Expand one cache line of the FP16 query once for all candidate rows.
    // Two accumulators break the dependency chain across the 32 dimensions.
    const __m512i qh =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + d));
    const __m512 q0 = _mm512_cvtph_ps(_mm512_castsi512_si256(qh));
    const __m512 q1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(qh, 1));
    for (size_t lane = 0; lane < Batch; ++lane) {
      if constexpr (PrefetchNext) {
        if (lane < prefetch_count) {
          const auto *next_row =
              static_cast<const char *>(prefetch_vectors[lane]);
          _mm_prefetch(next_row + d * sizeof(uint16_t), _MM_HINT_T0);
        }
      }
      const auto *row = static_cast<const uint16_t *>(vectors[lane]);
      const __m512i xh =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(row + d));
      const __m512 x0 = _mm512_cvtph_ps(_mm512_castsi512_si256(xh));
      const __m512 x1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(xh, 1));
      const __m512 delta0 = _mm512_sub_ps(q0, x0);
      const __m512 delta1 = _mm512_sub_ps(q1, x1);
      sums0[lane] = _mm512_fmadd_ps(delta0, delta0, sums0[lane]);
      sums1[lane] = _mm512_fmadd_ps(delta1, delta1, sums1[lane]);
    }
  }

  if (d + 16 <= dimension) {
    const __m512 q = _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + d)));
    for (size_t lane = 0; lane < Batch; ++lane) {
      const auto *row = static_cast<const uint16_t *>(vectors[lane]);
      const __m512 x = _mm512_cvtph_ps(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + d)));
      const __m512 delta = _mm512_sub_ps(q, x);
      sums0[lane] = _mm512_fmadd_ps(delta, delta, sums0[lane]);
    }
    d += 16;
  }

  for (size_t lane = 0; lane < Batch; ++lane) {
    float sum = Reduce(_mm512_add_ps(sums0[lane], sums1[lane]));
    const auto *row = static_cast<const uint16_t *>(vectors[lane]);
    for (size_t tail = d; tail < dimension; ++tail) {
      const float delta = HalfToFloat(query[tail]) - HalfToFloat(row[tail]);
      sum += delta * delta;
    }
    distances[lane] = sum;
  }
}

inline void PrefetchRow(const void *vector, size_t dimension) {
  const auto *row = static_cast<const char *>(vector);
  _mm_prefetch(row, _MM_HINT_T0);

  // A SIFT FP16 row is only 256 bytes and the four-way kernel below already
  // leaves enough sequential work for the hardware prefetchers.  GIST rows
  // are 1920 bytes, however, and candidate ids arrive in graph order, so the
  // first few demand loads otherwise serialize on unrelated cache lines.
  // Stage the beginning of long rows one batch ahead without trying to pull
  // the entire row into L1 (which would evict the batch currently in flight).
  if (dimension >= 512) {
    _mm_prefetch(row + 64, _MM_HINT_T0);
    _mm_prefetch(row + 128, _MM_HINT_T0);
    _mm_prefetch(row + 192, _MM_HINT_T0);
  }
}

template <size_t Batch>
void PrefetchAndCompute(const void **vectors, const uint16_t *query,
                        size_t count, size_t dimension, size_t *offset,
                        float *distances) {
  while (count - *offset >= Batch) {
    const size_t next = *offset + Batch;
    const size_t prefetch_count = std::min(Batch, count - next);
    if (prefetch_count) {
      DistanceBatch<Batch, true>(vectors + *offset, query, dimension,
                                 distances + *offset, vectors + next,
                                 prefetch_count);
    } else {
      DistanceBatch<Batch>(vectors + *offset, query, dimension,
                           distances + *offset);
    }
    *offset += Batch;
  }
}

}  // namespace

void squared_euclidean_fp16_batch_distance(const void **vectors,
                                           const void *query, size_t count,
                                           size_t dimension, float *distances,
                                           const void ** /*extra_values*/) {
  const auto *query_fp16 = static_cast<const uint16_t *>(query);
  if (count == 0) return;

  if (dimension < 512) {
    // Preserve the SIFT-tuned path: four rows provide sufficient MLP without
    // increasing register pressure for short vectors.
    constexpr size_t kBatch = 4;
    for (size_t i = 0; i < std::min(count, kBatch); ++i) {
      PrefetchRow(vectors[i], dimension);
    }
    size_t i = 0;
    for (; i + kBatch <= count; i += kBatch) {
      const size_t next = i + kBatch;
      for (size_t lane = 0; lane < std::min(kBatch, count - next); ++lane) {
        PrefetchRow(vectors[next + lane], dimension);
      }
      DistanceBatch<kBatch>(vectors + i, query_fp16, dimension, distances + i);
    }
    for (; i < count; ++i) {
      DistanceBatch<1>(vectors + i, query_fp16, dimension, distances + i);
    }
    return;
  }

  // Long rows are memory-latency dominated. Ten independent rows reuse each
  // expanded query cache line while exposing enough unrelated loads to keep
  // the Ice Lake memory pipeline busy. The smaller high-recall rerank sets use
  // a twelve-row first batch: this exactly covers candidate_topk=12 and leaves
  // one four-row batch for candidate_topk=16. Handle all other remainders with
  // 8/4/3/2/1 row kernels rather than a string of scalar calls.
  constexpr size_t kLongBatch = 10;
  constexpr size_t kSmallLongBatch = 12;
  const size_t first_batch = count < 20 ? kSmallLongBatch : kLongBatch;
  for (size_t i = 0; i < std::min(count, first_batch); ++i) {
    PrefetchRow(vectors[i], dimension);
  }
  size_t i = 0;
  if (count < 20) {
    PrefetchAndCompute<kSmallLongBatch>(vectors, query_fp16, count, dimension,
                                        &i, distances);
  } else {
    PrefetchAndCompute<kLongBatch>(vectors, query_fp16, count, dimension, &i,
                                   distances);
  }
  PrefetchAndCompute<8>(vectors, query_fp16, count, dimension, &i, distances);
  PrefetchAndCompute<4>(vectors, query_fp16, count, dimension, &i, distances);
  switch (count - i) {
    case 3:
      DistanceBatch<3>(vectors + i, query_fp16, dimension, distances + i);
      break;
    case 2:
      DistanceBatch<2>(vectors + i, query_fp16, dimension, distances + i);
      break;
    case 1:
      DistanceBatch<1>(vectors + i, query_fp16, dimension, distances + i);
      break;
    default:
      break;
  }
}

void fp32_to_fp16(const float *input, size_t dimension, uint16_t *output) {
  size_t d = 0;
  constexpr int kRounding = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  for (; d + 16 <= dimension; d += 16) {
    const __m512 values = _mm512_loadu_ps(input + d);
    const __m256i half = _mm512_cvtps_ph(values, kRounding);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + d), half);
  }
  if (d + 8 <= dimension) {
    const __m256 values = _mm256_loadu_ps(input + d);
    const __m128i half = _mm256_cvtps_ph(values, kRounding);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + d), half);
    d += 8;
  }
  for (; d < dimension; ++d) {
    output[d] = _cvtss_sh(input[d], kRounding);
  }
}

}  // namespace zvec::turbo::avx512_vnni
