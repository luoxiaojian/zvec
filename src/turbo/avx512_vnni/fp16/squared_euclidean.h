// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace zvec::turbo::avx512_vnni {

// Squared L2 from one FP16 query to FP16 rows. Rows are independent pointers
// so callers can gather candidates from a graph result without copying them.
// This uses the common turbo BatchDistanceFunc ABI.
void squared_euclidean_fp16_batch_distance(
    const void **vectors, const void *query, std::size_t count,
    std::size_t dimension, float *distances, const void **extra_values);

}  // namespace zvec::turbo::avx512_vnni
