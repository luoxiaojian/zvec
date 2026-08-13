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

// Record layout:
//   [ original_dim bytes: int8 values, stored as uint8(value) - 128 ]
//   [ int32 sum_sq_u8 ]
//
// The index data type remains DT_INT8. Stored-to-stored functions always
// return true squared L2. Query functions include the query-only correction.
void uniform_squared_euclidean_uint8_distance(const void *a, const void *b,
                                              size_t dim, float *distance);

void uniform_squared_euclidean_uint8_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values = nullptr);

// Convert one canonical stored value into the batch-query representation:
//   body: int8(raw - 128) -> uint8(raw)
//   tail: sum(raw^2)      -> sum(raw^2) - 256 * sum(raw)
void uniform_squared_euclidean_uint8_query_preprocess(void *query, size_t dim);

}  // namespace zvec::turbo::avx512_vnni
