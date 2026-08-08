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
using BatchDistanceFunc =
    std::function<void(const void **m, const void *q, size_t num, size_t dim,
                       float *out, const void **extra_values)>;
using QueryPreprocessFunc =
    zvec::ailego::DistanceBatch::DistanceBatchQueryPreprocessFunc;

// Uniform quantize kernel: fp32 -> byte with a global affine transform:
//   out[i] = clip(round(in[i] * scale + bias), 0, 127)
// for UNIFORM_INT8. UNIFORM_UINT8 quantizes through [0, 255], then stores
// int8(value - 128), the canonical index representation. This signature is
// specific to the uniform quantizers and is NOT a generic quantize contract.
// Raw function pointer avoids indirect-call overhead on the hot path.
using UniformQuantizeFunc = void (*)(const float *in, size_t dim, float scale,
                                     float bias, int8_t *out);

// Packed global uint4 quantization. Two codes are stored per byte (low nibble
// first), and the logical dimension is padded to a multiple of 128.
using UniformUint4QuantizeFunc = void (*)(const float *in, size_t dim,
                                          float minimum, float range,
                                          uint8_t *out);

// Direct physical FP32 -> uint8_t conversion for values already represented
// in the unsigned-byte domain. This is not an affine quantizer.
using RawUint8ConvertFunc = void (*)(const float *in, size_t dim,
                                     uint8_t *out);

using Fp32ToFp16ConvertFunc = void (*)(const float *in, size_t dim,
                                       uint16_t *out);

enum class MetricType {
  kSquaredEuclidean,
  kCosine,
  kMipsSquaredEuclidean,
  kUnknown,
};

enum class DataType {
  kInt8,
  kInt4,
  kFp16,
  kUint8,
  kUnknown,
};

enum class QuantizeType {
  kDefault,
  kUniform,
  kUniformUint8,
  kUniformUint4,
};

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type);

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type);

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type);

// Returns the SIMD kernel for the uniform quantizer on the current CPU for
// the given output data_type, or nullptr if no SIMD implementation is
// available (callers must keep a scalar fallback). This is a
// uniform-specific accessor intentionally kept outside of the generic
// (metric/data/quantize) dispatch above; data_type is retained so the
// interface can grow to cover additional quantized output types.
UniformQuantizeFunc get_uniform_quantize_func(DataType data_type);

// Same dispatch shape as get_uniform_quantize_func, but quantizes through
// [0, 255] and emits the canonical shifted int8(value - 128) representation.
UniformQuantizeFunc get_uniform_uint8_quantize_func(DataType data_type);

UniformUint4QuantizeFunc get_uniform_uint4_quantize_func(DataType data_type);

RawUint8ConvertFunc get_raw_uint8_convert_func();

Fp32ToFp16ConvertFunc get_fp32_to_fp16_convert_func();

}  // namespace zvec::turbo
