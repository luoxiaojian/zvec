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

#ifndef ZVEC_UNIFORM_UINT8_QUERY_PREPROCESS
#define ZVEC_UNIFORM_UINT8_QUERY_PREPROCESS 1
#endif

namespace zvec::turbo::avx512_vnni {

// Record layout:
//   [ original_dim bytes: uint8 values ]
//   [ int32 sum_u8 ]
//   [ int32 sum_sq_u8 ]
//
// The index data type remains DT_INT8; bytes are interpreted as uint8_t here.
void uniform_squared_euclidean_uint8_distance(const void *a, const void *b,
                                              size_t dim, float *distance);

void uniform_squared_euclidean_uint8_batch_distance(const void *const *vectors,
                                                    const void *query, size_t n,
                                                    size_t dim,
                                                    float *distances);

#if ZVEC_UNIFORM_UINT8_QUERY_PREPROCESS
void uniform_squared_euclidean_uint8_preprocessed_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances);

void uniform_squared_euclidean_uint8_query_preprocess(void *query, size_t dim);
#endif

}  // namespace zvec::turbo::avx512_vnni
