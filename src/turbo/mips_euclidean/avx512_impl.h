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

// Compute MIPS squared Euclidean distance between a single quantized INT8
// vector pair. The query is NOT preprocessed for the single-vector path (both
// sides remain int8, using the sign+maddubs AVX2 kernel).
// `dim` includes the original vector bytes plus a 20-byte metadata tail
// (4 floats: scale_a, bias_a, sum_a, sum2_a).
void mips_l2_int8_distance_avx512_vnni(const void *a, const void *b, int dim,
                                       float *distance);

// Batch version. The query must have been preprocessed by
// mips_l2_int8_query_preprocess_avx512_vnni (int8 -> uint8 via +128 shift) so
// that the AVX512-VNNI dpbusd instruction can be used. The 128 * int8_sum bias
// introduced by the shift is corrected per-vector using the stored int8_sum.
void mips_l2_int8_batch_distance_avx512_vnni(const void *const *vectors,
                                             const void *query, int n, int dim,
                                             float *distances);

// Preprocess the query in-place (int8 -> uint8 via +128) for the batch path.
// Only the original_dim bytes are shifted; the metadata tail is left intact.
// `dim` includes the 20-byte metadata tail.
void mips_l2_int8_query_preprocess_avx512_vnni(void *query, size_t dim);

}  // namespace zvec::turbo
