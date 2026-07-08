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

#include "avx512_vnni/uniform_uint8/quantize.h"
#include <algorithm>
#include <cmath>

namespace zvec::turbo::avx512_vnni {

void uniform_uint8_quantize(const float *in, std::size_t dim, float scale,
                            float bias, std::int8_t *out) {
  auto *u8_out = reinterpret_cast<std::uint8_t *>(out);
  for (std::size_t i = 0; i < dim; ++i) {
    float v = std::round(in[i] * scale + bias);
    v = std::max(0.0f, std::min(255.0f, v));
    u8_out[i] = static_cast<std::uint8_t>(v);
  }
}

}  // namespace zvec::turbo::avx512_vnni
