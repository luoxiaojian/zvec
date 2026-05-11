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

#include <functional>
#include <zvec/ailego/math_batch/utils.h>

namespace zvec::turbo {

using DistanceFunc =
    std::function<void(const void *m, const void *q, size_t dim, float *out)>;
using BatchDistanceFunc = std::function<void(
    const void **m, const void *q, size_t num, size_t dim, float *out)>;
using QueryPreprocessFunc =
    zvec::ailego::DistanceBatch::DistanceBatchQueryPreprocessFunc;

// Split-layout batch distance: vectors contain the core feature only,
// any per-vector side data (e.g. sq_sum_half, or multiple extra scalars)
// is provided via the `side_data` array, where `side_data[i]` points to
// the side-data block of vector i.  This allows kernels to consume one
// or more extra values per vector.  Returns nullptr if no split-layout
// kernel is available for the given (metric, data_type, quantize_type).
using BatchDistanceSplitFunc = std::function<void(
    const void **m, const void *q, const void *const *side_data, size_t num,
    size_t dim, float *out)>;

// Quantize fp32 -> int8 with a global affine transform:
//   out[i] = clip(round(in[i] * scale + bias), -127, 127)
// Raw function pointer (rather than std::function) to avoid indirect-call
// overhead on the per-record / per-query hot path.
using QuantizeFunc = void (*)(const float *in, size_t dim, float scale,
                              float bias, int8_t *out);

enum class MetricType {
  kSquaredEuclidean,
  kCosine,
  kMipsSquaredEuclidean,
  kUnknown,
};

enum class DataType {
  kInt8,
  kUnknown,
};

enum class QuantizeType {
  kDefault,
  kUniform,
  kUnitScale,
};

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type);

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type);

// Split-layout batch distance kernel (see BatchDistanceSplitFunc).
// Currently only implemented for (kSquaredEuclidean, kInt8, kUnitScale).
BatchDistanceSplitFunc get_batch_distance_split_func(
    MetricType metric_type, DataType data_type, QuantizeType quantize_type);

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type);

// Pairwise (data-to-data) distance: symmetric kernel for inter-candidate
// distance computation (e.g., robust_prune). Does not require query
// preprocessing.
DistanceFunc get_pairwise_distance_func(MetricType metric_type,
                                        DataType data_type,
                                        QuantizeType quantize_type);

BatchDistanceFunc get_pairwise_batch_distance_func(MetricType metric_type,
                                                   DataType data_type,
                                                   QuantizeType quantize_type);

// Returns a vectorized quantize kernel for (data_type, quantize_type), or
// nullptr if no SIMD implementation is available on the current CPU
// (callers must keep a scalar fallback).
QuantizeFunc get_quantize_func(DataType data_type, QuantizeType quantize_type);

}  // namespace zvec::turbo
