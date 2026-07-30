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

#pragma once

#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx512_vnni {

// Uniform uint8 quantization:
//   raw    = clip(round(in[i] * scale + bias), 0, 255)
//   out[i] = int8(raw - 128)
//
// The shifted value is the canonical representation stored in the index.
void uniform_uint8_quantize(const float *in, std::size_t dim, float scale,
                            float bias, std::int8_t *out);

}  // namespace zvec::turbo::avx512_vnni
