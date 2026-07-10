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

#include <cstdint>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "metric_params.h"

namespace zvec {
namespace core {

namespace {

constexpr size_t kTailBytes = sizeof(int32_t) * 2;

size_t original_dim(size_t dim) {
  return dim > kTailBytes ? dim - kTailBytes : 0;
}

const int32_t *tail(const void *ptr, size_t orig_dim) {
  return reinterpret_cast<const int32_t *>(
      reinterpret_cast<const uint8_t *>(ptr) + orig_dim);
}

void UniformUint8SquaredEuclidean(const void *a, const void *b, size_t dim,
                                  float *distance) {
  const size_t orig_dim = original_dim(dim);
  const auto *lhs = reinterpret_cast<const uint8_t *>(a);
  const auto *rhs = reinterpret_cast<const uint8_t *>(b);
  int64_t dot = 0;
  for (size_t i = 0; i < orig_dim; ++i) {
    dot += static_cast<int>(lhs[i]) * static_cast<int>(rhs[i]);
  }
  const int32_t *lhs_tail = tail(a, orig_dim);
  const int32_t *rhs_tail = tail(b, orig_dim);
  *distance = static_cast<float>(static_cast<int64_t>(lhs_tail[1]) +
                                 static_cast<int64_t>(rhs_tail[1]) - 2 * dot);
}

#if !ZVEC_UNIFORM_UINT8_QUERY_PREPROCESS
void UniformUint8SquaredEuclideanBatch(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances) {
  for (size_t i = 0; i < n; ++i) {
    UniformUint8SquaredEuclidean(vectors[i], query, dim, distances + i);
  }
}
#endif

#if ZVEC_UNIFORM_UINT8_QUERY_PREPROCESS
void UniformUint8QueryPreprocess(void *query, size_t dim) {
  const size_t orig_dim = original_dim(dim);
  auto *bytes = reinterpret_cast<uint8_t *>(query);
  for (size_t i = 0; i < orig_dim; ++i) {
    bytes[i] = static_cast<uint8_t>(static_cast<int>(bytes[i]) - 128);
  }
}

void UniformUint8SquaredEuclideanPreprocessedQueryBatch(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {
  for (size_t i = 0; i < n; ++i) {
    const size_t orig_dim = original_dim(dim);
    const auto *lhs = reinterpret_cast<const uint8_t *>(vectors[i]);
    const auto *rhs = reinterpret_cast<const int8_t *>(query);
    int64_t ip_shifted = 0;
    for (size_t d = 0; d < orig_dim; ++d) {
      ip_shifted += static_cast<int>(lhs[d]) * static_cast<int>(rhs[d]);
    }
    const int32_t *lhs_tail = tail(vectors[i], orig_dim);
    const int32_t *rhs_tail = tail(query, orig_dim);
    const int64_t dot = ip_shifted + 128 * static_cast<int64_t>(lhs_tail[0]);
    distances[i] =
        static_cast<float>(static_cast<int64_t>(lhs_tail[1]) +
                           static_cast<int64_t>(rhs_tail[1]) - 2 * dot);
  }
}
#endif

}  // namespace

class UniformUint8Metric : public IndexMetric {
 public:
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8) {
      LOG_ERROR("UniformUint8Metric: unsupported type %d", meta.data_type());
      return IndexError_Unsupported;
    }

    std::string metric_name;
    index_params.get(UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME, &metric_name);
    if (metric_name.empty()) {
      LOG_ERROR("UniformUint8Metric: param %s is required",
                UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME.c_str());
      return IndexError_InvalidArgument;
    }
    if (metric_name != "SquaredEuclidean") {
      LOG_ERROR("UniformUint8Metric: only SquaredEuclidean supported, got %s",
                metric_name.c_str());
      return IndexError_Unsupported;
    }

    meta_ = meta;
    params_ = index_params;
    return 0;
  }

  int cleanup(void) override {
    return 0;
  }

  bool is_matched(const IndexMeta &meta) const override {
    return meta.data_type() == meta_.data_type() &&
           meta.unit_size() == meta_.unit_size();
  }

  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return qmeta.data_type() == meta_.data_type() &&
           qmeta.unit_size() == meta_.unit_size() &&
           qmeta.dimension() == meta.dimension();
  }

  MatrixDistance distance(void) const override {
    return distance_matrix(1, 1);
  }

  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m == 1 && n == 1) {
      auto turbo_ret = turbo::get_distance_func(
          turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
          turbo::QuantizeType::kUniformUint8);
      if (turbo_ret) {
        return turbo_ret;
      }
      return UniformUint8SquaredEuclidean;
    }
    return nullptr;
  }

  MatrixBatchDistance batch_distance(void) const override {
    auto turbo_ret = turbo::get_batch_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniformUint8);
    if (turbo_ret) {
      return turbo_ret;
    }
#if ZVEC_UNIFORM_UINT8_QUERY_PREPROCESS
    return UniformUint8SquaredEuclideanPreprocessedQueryBatch;
#else
    return UniformUint8SquaredEuclideanBatch;
#endif
  }

  const ailego::Params &params(void) const override {
    return params_;
  }

  int train(const void * /*vec*/, size_t /*dim*/) override {
    return 0;
  }

  bool support_train(void) const override {
    return false;
  }

  void normalize(float * /*score*/) const override {}

  bool support_normalize(void) const override {
    return false;
  }

  Pointer query_metric(void) const override {
    return nullptr;
  }

  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
#if ZVEC_UNIFORM_UINT8_QUERY_PREPROCESS
    auto turbo_ret = turbo::get_query_preprocess_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniformUint8);
    if (turbo_ret) {
      return turbo_ret;
    }
    return UniformUint8QueryPreprocess;
#else
    return nullptr;
#endif
  }

 private:
  IndexMeta meta_{};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(UniformUint8, UniformUint8Metric);

}  // namespace core
}  // namespace zvec
