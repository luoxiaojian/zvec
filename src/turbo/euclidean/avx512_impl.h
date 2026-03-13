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

namespace zvec::turbo {

void l2_int8_distance_avx512_vnni(const void *a, const void *b, int dim,
                                  float *distance);

void l2_int8_batch_distance_avx512_vnni(const void *const *vectors,
                                        const void *query, int n, int dim,
                                        float *distances);

void l2_int8_query_preprocess_avx512_vnni(void *query, size_t dim);


}  // namespace zvec::turbo
