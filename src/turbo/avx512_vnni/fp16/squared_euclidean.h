// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace zvec::turbo::avx512_vnni {

// Squared L2 from one FP32 query to FP16 rows. Rows are independent pointers
// so callers can gather candidates from a graph result without copying them.
void fp32_fp16_squared_euclidean_batch_distance(
    const void **vectors, const float *query, std::size_t count,
    std::size_t dimension, float *distances);

}  // namespace zvec::turbo::avx512_vnni
