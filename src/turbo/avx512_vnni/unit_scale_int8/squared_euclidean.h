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

// Compute squared Euclidean distance for unit-scale quantized INT8 vectors
// (the scale=1 specialization of the Uniform Int8 Quantizer).
//
// Layout (database vector):
//   [ original_dim bytes: int8_t elements (value = round(float) + bias) ]
//   [ float sq_sum_half ]  (sum of original float squares / 2)
//
// Total: dim = original_dim + sizeof(float) = original_dim + 4.
//
// The query is stored as uint8 (value = round(float)), with no tail.
// Distance is computed as: sq_sum_half - dpbusd(query_uint8, data_int8),
// which serves as a monotonic proxy for the true L2 distance for ranking.
void unit_scale_squared_euclidean_int8_distance(const void *database_vec,
                                                const void *query, size_t dim,
                                                float *distance);

// Batch version: compute distances between `n` unit-scale quantized database
// vectors and a single uint8 query.  The query must already be in uint8
// format (preprocessed via +128 shift from int8 encoding).
void unit_scale_squared_euclidean_int8_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances);

// Split-layout batch kernel.
//
// Used when the streamer stores the int8 vector body and the side-data
// block in two separate flat arrays (cache-friendly, matches pyglass
// MyQuant layout).  Vectors passed here are pure int8 data of length
// (dim - 4) (no 4-byte tail).  `side_data[i]` points to the side-data
// block of vector i; this kernel only reads the first `float` from that
// block as `sq_sum_half`, but the pointer form allows future kernels to
// consume additional extra values without changing the interface.
// `dim` includes the 4-byte tail (matches the non-split variant), and
// the kernel internally derives original_dim.
//
// Result: distances[i] = *(const float*)side_data[i]
//                      - dpbusd(query_uint8, vectors[i]).
void unit_scale_squared_euclidean_int8_batch_distance_split(
    const void *const *vectors, const void *query, const void *const *side_data,
    size_t n, size_t dim, float *distances);

// Preprocess the query vector in-place: shift int8 -> uint8 by adding 128.
// Only the first (dim - 4) bytes (original_dim) are shifted; the 4-byte
// sq_sum_half tail is left intact.  `dim` includes the 4-byte tail.
void unit_scale_squared_euclidean_int8_query_preprocess(void *query,
                                                        size_t dim);

}  // namespace zvec::turbo::avx512_vnni
