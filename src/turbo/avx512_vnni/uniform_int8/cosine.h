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

namespace zvec::turbo::avx512_vnni {

// Compute cosine distance (negative inner product) between two
// uniform-quantized INT8 vectors. Since all vectors share a single global
// scale/bias, the cosine distance reduces to -sum(a[i] * b[i]) on raw int8
// values. There is NO metadata tail — `dim` is the pure int8 vector length.
void uniform_cosine_int8_distance(const void *a, const void *b, size_t dim,
                                  float *distance);

// Batch version: compute cosine distance between `n` INT8 database vectors
// and a single INT8 query. No query preprocessing is required because values
// are in [0, 127] and can be used directly with the VNNI dpbusd instruction.
void uniform_cosine_int8_batch_distance(const void *const *vectors,
                                        const void *query, size_t n,
                                        size_t dim, float *distances);

}  // namespace zvec::turbo::avx512_vnni
