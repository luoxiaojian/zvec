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
#include <cstring>
#include <memory>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "metric_params.h"

namespace zvec {
namespace core {

namespace {

constexpr size_t kTailBytes = sizeof(int32_t);

size_t original_dim(size_t dim) {
  return dim > kTailBytes ? dim - kTailBytes : 0;
}

void UniformUint8StoredSquaredEuclidean(const void *a, const void *b,
                                        size_t dim, float *distance) {
  const size_t orig_dim = original_dim(dim);
  const auto *lhs = reinterpret_cast<const int8_t *>(a);
  const auto *rhs = reinterpret_cast<const int8_t *>(b);
  int64_t dist = 0;
  for (size_t i = 0; i < orig_dim; ++i) {
    int diff = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    dist += diff * diff;
  }
  *distance = static_cast<float>(dist);
}

IndexMetric::MatrixDistance UniformUint8StoredDistance() {
  static const IndexMetric::MatrixDistance distance = []() {
    auto turbo_distance = turbo::get_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniformUint8);
    return turbo_distance ? turbo_distance : UniformUint8StoredSquaredEuclidean;
  }();
  return distance;
}

void UniformUint8StoredQuerySquaredEuclidean(const void *stored,
                                             const void *query, size_t dim,
                                             float *distance) {
  const size_t orig_dim = original_dim(dim);
  const auto *lhs = reinterpret_cast<const int8_t *>(stored);
  const auto *rhs = reinterpret_cast<const uint8_t *>(query);
  int64_t dist = 0;
  for (size_t d = 0; d < orig_dim; ++d) {
    const int diff =
        static_cast<int>(lhs[d]) - (static_cast<int>(rhs[d]) - 128);
    dist += diff * diff;
  }
  *distance = static_cast<float>(dist);
}

void UniformUint8StoredQuerySquaredEuclideanBatch(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const * /*extra_values*/) {
  for (size_t i = 0; i < n; ++i) {
    UniformUint8StoredQuerySquaredEuclidean(vectors[i], query, dim,
                                            distances + i);
  }
}

void UniformUint8QueryPreprocess(void *query, size_t dim) {
  const size_t orig_dim = original_dim(dim);
  auto *stored = reinterpret_cast<int8_t *>(query);
  auto *raw = reinterpret_cast<uint8_t *>(query);
  int64_t sum = 0;
  for (size_t d = 0; d < orig_dim; ++d) {
    const int value = static_cast<int>(stored[d]) + 128;
    raw[d] = static_cast<uint8_t>(value);
    sum += value;
  }

  int32_t sum_sq = 0;
  std::memcpy(&sum_sq, raw + orig_dim, sizeof(sum_sq));
  const int32_t correction =
      static_cast<int32_t>(static_cast<int64_t>(sum_sq) - 256 * sum);
  std::memcpy(raw + orig_dim, &correction, sizeof(correction));
}

IndexMetric::DistanceBatchQueryPreprocessFunc
UniformUint8QueryPreprocessFunc() {
  static const IndexMetric::DistanceBatchQueryPreprocessFunc preprocess = []() {
    auto turbo_preprocess = turbo::get_query_preprocess_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniformUint8);
    return turbo_preprocess ? turbo_preprocess : UniformUint8QueryPreprocess;
  }();
  return preprocess;
}

}  // namespace

class UniformUint8QueryMetric : public IndexMetric {
 public:
  UniformUint8QueryMetric() = default;
  UniformUint8QueryMetric(const IndexMeta &meta, const ailego::Params &params)
      : meta_(meta), params_(params) {}

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
    return UniformUint8StoredQuerySquaredEuclidean;
  }

  MatrixBatchDistance batch_distance(void) const override {
    auto turbo_ret = turbo::get_batch_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniformUint8);
    if (turbo_ret) {
      return turbo_ret;
    }
    return UniformUint8StoredQuerySquaredEuclideanBatch;
  }

  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return UniformUint8QueryPreprocessFunc();
  }

  size_t extra_values_size_per_vector(void) const override {
    return kTailBytes;
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

 protected:
  IndexMeta meta_{};
  ailego::Params params_{};
};

class UniformUint8Metric : public UniformUint8QueryMetric {
 public:
  MatrixDistance distance(void) const override {
    return UniformUint8StoredDistance();
  }

  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m == 1 && n == 1) {
      return UniformUint8StoredDistance();
    }
    return nullptr;
  }

  // Extra values are a physical-storage optimization for online search.
  // The stored record remains self-contained before contiguous-memory
  // optimization, so the build metric itself does not require a side column.
  size_t extra_values_size_per_vector(void) const override {
    return 0;
  }

  Pointer query_metric(void) const override {
    return std::make_shared<UniformUint8QueryMetric>(meta_, params_);
  }
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(UniformUint8, UniformUint8Metric);

}  // namespace core
}  // namespace zvec
