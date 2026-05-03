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

// AVX-512 quantization for the uniform-int8 quantizer.
//
// Pipeline (16 floats per iteration):
//   1. Load 16 fp32 values                                   (vmovups)
//   2. Fused multiply-add:  v = in * scale + bias            (vfmadd)
//   3. Convert fp32 -> int32 with current rounding mode      (vcvtps2dq)
//      (default = round-to-nearest-even, matches std::round
//       for the magnitude range relevant to int8 output)
//   4. Saturating pack int32 -> int8                         (vpmovsdb)
//      Hardware saturation handles the clip to [-128, 127];
//      the upstream training code clamps the bias so the value
//      never reaches -128, keeping the symmetric [-127, 127]
//      contract identical to the scalar implementation.
//   5. Store 16 int8 values                                  (vmovdqu)
//
// Compiled with -march=avx512vnni (set per-file in src/turbo/CMakeLists.txt).

#include "avx512_vnni/uniform_quantized_int8/quantize.h"

#include <algorithm>
#include <cmath>

#if defined(__AVX512F__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>

namespace zvec::turbo::avx512_vnni {

void uniform_int8_quantize(const float *in, std::size_t dim, float scale,
                           float bias, std::int8_t *out) {
  const __m512 vscale = _mm512_set1_ps(scale);
  const __m512 vbias = _mm512_set1_ps(bias);

  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m512 v = _mm512_loadu_ps(in + i);
    v = _mm512_fmadd_ps(v, vscale, vbias);
    // fp32 -> int32 with current rounding mode (round-to-nearest-even).
    __m512i vi = _mm512_cvtps_epi32(v);
    // Saturating pack int32 -> int8 (clips to [-128, 127]).
    __m128i packed = _mm512_cvtsepi32_epi8(vi);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out + i), packed);
  }

  // Tail: scalar fallback (matches the scalar reference exactly).
  for (; i < dim; ++i) {
    float v = std::round(in[i] * scale + bias);
    v = std::max(-127.0f, std::min(127.0f, v));
    out[i] = static_cast<std::int8_t>(v);
  }
}

}  // namespace zvec::turbo::avx512_vnni

#else  // no AVX-512 support — provide a no-op stub so dispatch can fall back

namespace zvec::turbo::avx512_vnni {

void uniform_int8_quantize(const float * /*in*/, std::size_t /*dim*/,
                           float /*scale*/, float /*bias*/,
                           std::int8_t * /*out*/) {
  // Intentionally empty; turbo::get_quantize_func will return nullptr on
  // CPUs without AVX-512 support and the caller will use its scalar path.
}

}  // namespace zvec::turbo::avx512_vnni

#endif
