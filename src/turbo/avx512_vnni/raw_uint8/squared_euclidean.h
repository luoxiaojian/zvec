// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace zvec::turbo::avx512_vnni {

void fp32_to_raw_uint8(const float *input, std::size_t dimension,
                       uint8_t *output);

// Raw homogeneous uint8_t squared L2. `dimension` is exactly the number of
// stored bytes; records contain no quantization parameters or correction tail.
void squared_euclidean_uint8_distance(const void *lhs, const void *rhs,
                                      std::size_t dimension, float *distance);

void squared_euclidean_uint8_batch_distance(
    const void **vectors, const void *query, std::size_t count,
    std::size_t dimension, float *distances, const void **extra_values);

}  // namespace zvec::turbo::avx512_vnni
