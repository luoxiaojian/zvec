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

#include <ailego/internal/cpu_features.h>
#include <zvec/turbo/turbo.h>
#include "avx512_vnni/fp16/squared_euclidean.h"
#include "avx512_vnni/raw_uint8/squared_euclidean.h"
#include "avx512_vnni/record_quantized_int8/cosine.h"
#include "avx512_vnni/record_quantized_int8/squared_euclidean.h"
#include "avx512_vnni/uniform_int8/quantize.h"
#include "avx512_vnni/uniform_int8/squared_euclidean.h"
#include "avx512_vnni/uniform_uint8/quantize.h"
#include "avx512_vnni/uniform_uint8/squared_euclidean.h"
#include "avx512_vnni/uniform_uint4/quantize.h"
#include "avx512_vnni/uniform_uint4/squared_euclidean.h"

namespace zvec::turbo {

DistanceFunc get_distance_func(MetricType metric_type, DataType data_type,
                               QuantizeType quantize_type) {
  if (data_type == DataType::kUint8 &&
      quantize_type == QuantizeType::kDefault &&
      metric_type == MetricType::kSquaredEuclidean &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512BW) {
    return avx512_vnni::squared_euclidean_uint8_distance;
  }
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_distance;
        }
      }
    }
    if (quantize_type == QuantizeType::kUniform) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::uniform_squared_euclidean_int8_distance;
        }
      }
    }
    if (quantize_type == QuantizeType::kUniformUint8) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::uniform_squared_euclidean_uint8_distance;
        }
      }
    }
  }
  if (data_type == DataType::kInt4 &&
      quantize_type == QuantizeType::kUniformUint4 &&
      metric_type == MetricType::kSquaredEuclidean &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
    return avx512_vnni::uniform_squared_euclidean_uint4_distance;
  }
  return nullptr;
}

BatchDistanceFunc get_batch_distance_func(MetricType metric_type,
                                          DataType data_type,
                                          QuantizeType quantize_type) {
  if (data_type == DataType::kUint8 &&
      quantize_type == QuantizeType::kDefault &&
      metric_type == MetricType::kSquaredEuclidean &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512BW) {
    return avx512_vnni::squared_euclidean_uint8_batch_distance;
  }
  if (data_type == DataType::kFp16 &&
      quantize_type == QuantizeType::kDefault &&
      metric_type == MetricType::kSquaredEuclidean &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512DQ &&
      zvec::ailego::internal::CpuFeatures::static_flags_.F16C) {
    return avx512_vnni::squared_euclidean_fp16_batch_distance;
  }
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_batch_distance;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_batch_distance;
        }
      }
    }
    if (quantize_type == QuantizeType::kUniform) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::uniform_squared_euclidean_int8_batch_distance;
        }
      }
    }
    if (quantize_type == QuantizeType::kUniformUint8) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::uniform_squared_euclidean_uint8_batch_distance;
        }
      }
    }
  }
  if (data_type == DataType::kInt4 &&
      quantize_type == QuantizeType::kUniformUint4 &&
      metric_type == MetricType::kSquaredEuclidean &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
    return avx512_vnni::uniform_squared_euclidean_uint4_batch_distance;
  }
  return nullptr;
}

QueryPreprocessFunc get_query_preprocess_func(MetricType metric_type,
                                              DataType data_type,
                                              QuantizeType quantize_type) {
  if (data_type == DataType::kInt8) {
    if (quantize_type == QuantizeType::kDefault) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::squared_euclidean_int8_query_preprocess;
        }
        if (metric_type == MetricType::kCosine) {
          return avx512_vnni::cosine_int8_query_preprocess;
        }
      }
    }
    if (quantize_type == QuantizeType::kUniformUint8) {
      if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
        if (metric_type == MetricType::kSquaredEuclidean) {
          return avx512_vnni::uniform_squared_euclidean_uint8_query_preprocess;
        }
      }
    }
  }
  return nullptr;
}

UniformQuantizeFunc get_uniform_quantize_func(DataType data_type) {
  if (data_type == DataType::kInt8) {
    // Quantize uses AVX-512F (no VNNI required), but we gate on the same
    // AVX512_VNNI flag for now since the kernel lives in the avx512_vnni
    // directory and is compiled with the same march flag.
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
      return avx512_vnni::uniform_int8_quantize;
    }
  }
  return nullptr;
}

UniformQuantizeFunc get_uniform_uint8_quantize_func(DataType data_type) {
  if (data_type == DataType::kInt8) {
    if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
      return avx512_vnni::uniform_uint8_quantize;
    }
  }
  return nullptr;
}

UniformUint4QuantizeFunc get_uniform_uint4_quantize_func(DataType data_type) {
  if (data_type == DataType::kInt4 &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512_VNNI) {
    return avx512_vnni::uniform_uint4_quantize;
  }
  return nullptr;
}

RawUint8ConvertFunc get_raw_uint8_convert_func() {
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
      zvec::ailego::internal::CpuFeatures::static_flags_.AVX512BW) {
    return avx512_vnni::fp32_to_raw_uint8;
  }
  return nullptr;
}

Fp32ToFp16ConvertFunc get_fp32_to_fp16_convert_func() {
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F &&
      zvec::ailego::internal::CpuFeatures::static_flags_.F16C) {
    return avx512_vnni::fp32_to_fp16;
  }
  return nullptr;
}

}  // namespace zvec::turbo
