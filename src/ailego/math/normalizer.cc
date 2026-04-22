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

#include "normalizer.h"

namespace zvec {
namespace ailego {

#if (defined(__ARM_NEON) && defined(__aarch64__))
static inline void NormalizeNEON(float *arr, size_t dim, float norm) {
  float *last = arr + dim;
  float *last_aligned = arr + ((dim >> 3) << 3);

  float32x4_t v_norm = vdupq_n_f32(norm);
  for (; arr != last_aligned; arr += 8) {
    vst1q_f32(arr + 0, vdivq_f32(vld1q_f32(arr + 0), v_norm));
    vst1q_f32(arr + 4, vdivq_f32(vld1q_f32(arr + 4), v_norm));
  }
  if (last >= last_aligned + 4) {
    vst1q_f32(arr, vdivq_f32(vld1q_f32(arr), v_norm));
    arr += 4;
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}

#endif  // __ARM_NEON && __aarch64__

#if defined(__AVX__)
#if defined(__AVX512F__)
static inline void NormalizeAVX512(float *arr, size_t dim, float norm) {
  float *last = arr + dim;
  float *last_aligned = arr + ((dim >> 4) << 4);

  __m512 zmm_norm = _mm512_set1_ps(norm);
  if (((uintptr_t)arr & 0x3f) == 0) {
    for (; arr != last_aligned; arr += 16) {
      _mm512_store_ps(arr, _mm512_div_ps(_mm512_load_ps(arr), zmm_norm));
    }
    if (last >= arr + 8) {
      __m256 ymm_norm = _mm256_set1_ps(norm);
      _mm256_store_ps(arr, _mm256_div_ps(_mm256_load_ps(arr), ymm_norm));
      arr += 8;
    }
    if (last >= arr + 4) {
      __m128 xmm_norm = _mm_set1_ps(norm);
      _mm_store_ps(arr, _mm_div_ps(_mm_load_ps(arr), xmm_norm));
      arr += 4;
    }
  } else {
    for (; arr != last_aligned; arr += 16) {
      _mm512_storeu_ps(arr, _mm512_div_ps(_mm512_loadu_ps(arr), zmm_norm));
    }
    if (last >= arr + 8) {
      __m256 ymm_norm = _mm256_set1_ps(norm);
      _mm256_storeu_ps(arr, _mm256_div_ps(_mm256_loadu_ps(arr), ymm_norm));
      arr += 8;
    }
    if (last >= arr + 4) {
      __m128 xmm_norm = _mm_set1_ps(norm);
      _mm_storeu_ps(arr, _mm_div_ps(_mm_loadu_ps(arr), xmm_norm));
      arr += 4;
    }
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}
#endif  // __AVX512F__

static inline void NormalizeAVX(float *arr, size_t dim, float norm) {
  float *last = arr + dim;
  float *last_aligned = arr + ((dim >> 4) << 4);

  __m256 ymm_norm = _mm256_set1_ps(norm);
  if (((uintptr_t)arr & 0x1f) == 0) {
    for (; arr != last_aligned; arr += 16) {
      _mm256_store_ps(arr + 0,
                      _mm256_div_ps(_mm256_load_ps(arr + 0), ymm_norm));
      _mm256_store_ps(arr + 8,
                      _mm256_div_ps(_mm256_load_ps(arr + 8), ymm_norm));
    }
    if (last >= arr + 8) {
      _mm256_store_ps(arr, _mm256_div_ps(_mm256_load_ps(arr), ymm_norm));
      arr += 8;
    }
    if (last >= arr + 4) {
      __m128 xmm_norm = _mm_set1_ps(norm);
      _mm_store_ps(arr, _mm_div_ps(_mm_load_ps(arr), xmm_norm));
      arr += 4;
    }
  } else {
    for (; arr != last_aligned; arr += 16) {
      _mm256_storeu_ps(arr + 0,
                       _mm256_div_ps(_mm256_loadu_ps(arr + 0), ymm_norm));
      _mm256_storeu_ps(arr + 8,
                       _mm256_div_ps(_mm256_loadu_ps(arr + 8), ymm_norm));
    }
    if (last >= arr + 8) {
      _mm256_storeu_ps(arr, _mm256_div_ps(_mm256_loadu_ps(arr), ymm_norm));
      arr += 8;
    }
    if (last >= arr + 4) {
      __m128 xmm_norm = _mm_set1_ps(norm);
      _mm_storeu_ps(arr, _mm_div_ps(_mm_loadu_ps(arr), xmm_norm));
      arr += 4;
    }
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}
#endif  // __AVX__

#if defined(__SSE__)
static inline void NormalizeSSE(float *arr, size_t dim, float norm) {
  float *last = arr + dim;
  float *last_aligned = arr + ((dim >> 3) << 3);

  __m128 xmm_norm = _mm_set1_ps(norm);
  if (((uintptr_t)arr & 0xf) == 0) {
    for (; arr != last_aligned; arr += 8) {
      _mm_store_ps(arr + 0, _mm_div_ps(_mm_load_ps(arr + 0), xmm_norm));
      _mm_store_ps(arr + 4, _mm_div_ps(_mm_load_ps(arr + 4), xmm_norm));
    }
    if (last >= last_aligned + 4) {
      _mm_store_ps(arr, _mm_div_ps(_mm_load_ps(arr), xmm_norm));
      arr += 4;
    }
  } else {
    for (; arr != last_aligned; arr += 8) {
      _mm_storeu_ps(arr + 0, _mm_div_ps(_mm_loadu_ps(arr + 0), xmm_norm));
      _mm_storeu_ps(arr + 4, _mm_div_ps(_mm_loadu_ps(arr + 4), xmm_norm));
    }
    if (last >= last_aligned + 4) {
      _mm_storeu_ps(arr, _mm_div_ps(_mm_loadu_ps(arr), xmm_norm));
      arr += 4;
    }
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}
#endif  // __SSE__

#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the norm of vector
void Normalizer<float>::Compute(ValueType *arr, size_t dim, float norm) {
#if defined(__ARM_NEON)
  NormalizeNEON(arr, dim, norm);
#else
#if defined(__AVX512F__)
  if (dim > 15) {
    NormalizeAVX512(arr, dim, norm);
    return;
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (dim > 7) {
    NormalizeAVX(arr, dim, norm);
    return;
  }
#endif  // __AVX__
  NormalizeSSE(arr, dim, norm);
#endif  // __ARM_NEON
}
#endif  // __SSE__ || (__ARM_NEON && __aarch64__)

}  // namespace ailego
}  // namespace zvec