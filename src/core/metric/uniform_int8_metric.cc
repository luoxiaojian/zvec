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

#include <ailego/math/euclidean_distance_matrix.h>
#include <ailego/math_batch/euclidean_distance_batch.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "metric_params.h"

namespace zvec {
namespace core {

/*! Index Metric for Uniform Int8 Quantization (Global Scale)
 *
 * Supports both SquaredEuclidean and Cosine distances. Since all vectors
 * share a single global scale/bias, no per-vector reconstruction is needed.
 *   SquaredEuclidean: distance = sum((a[i] - b[i])^2)
 *   Cosine:           distance = -sum(a[i] * b[i])
 */
class UniformInt8Metric : public IndexMetric {
 public:
  //! Initialize Metric
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8) {
      LOG_ERROR("UniformInt8Metric: unsupported type %d", meta.data_type());
      return IndexError_Unsupported;
    }

    std::string metric_name;
    index_params.get(UNIFORM_INT8_METRIC_ORIGIN_METRIC_NAME, &metric_name);
    if (metric_name.empty()) {
      LOG_ERROR("UniformInt8Metric: param %s is required",
                UNIFORM_INT8_METRIC_ORIGIN_METRIC_NAME.c_str());
      return IndexError_InvalidArgument;
    }

    if (metric_name == "SquaredEuclidean") {
      turbo_metric_type_ = turbo::MetricType::kSquaredEuclidean;
    } else if (metric_name == "Cosine") {
      turbo_metric_type_ = turbo::MetricType::kCosine;
    } else {
      LOG_ERROR(
          "UniformInt8Metric: only SquaredEuclidean and Cosine supported, "
          "got %s",
          metric_name.c_str());
      return IndexError_Unsupported;
    }

    meta_ = meta;
    params_ = index_params;

    LOG_INFO("UniformInt8Metric initialized: dimension=%u metric=%s",
             meta_.dimension(), metric_name.c_str());
    return 0;
  }

  //! Cleanup Metric
  int cleanup(void) override {
    return 0;
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta) const override {
    return meta.data_type() == meta_.data_type() &&
           meta.unit_size() == meta_.unit_size();
  }

  //! Retrieve if it matched
  bool is_matched(const IndexMeta &meta,
                  const IndexQueryMeta &qmeta) const override {
    return qmeta.data_type() == meta_.data_type() &&
           qmeta.unit_size() == meta_.unit_size() &&
           qmeta.dimension() == meta.dimension();
  }

  //! Retrieve distance function for query (1x1)
  MatrixDistance distance(void) const override {
    return distance_matrix(1, 1);
  }

  //! Retrieve matrix distance function
  //! Dispatches to the appropriate turbo kernel based on the configured metric.
  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m == 1 && n == 1) {
      auto turbo_ret = turbo::get_distance_func(
          turbo_metric_type_, turbo::DataType::kInt8,
          turbo::QuantizeType::kUniform);
      if (turbo_ret) {
        return turbo_ret;
      }
      // Scalar fallback only available for SquaredEuclidean
      if (turbo_metric_type_ == turbo::MetricType::kSquaredEuclidean) {
        return reinterpret_cast<MatrixDistanceHandle>(
            ailego::SquaredEuclideanDistanceMatrix<int8_t, 1, 1>::Compute);
      }
    }
    return nullptr;
  }

  //! Retrieve batch distance function
  //! Dispatches to the appropriate turbo batch kernel based on the configured
  //! metric.
  MatrixBatchDistance batch_distance(void) const override {
    auto turbo_ret = turbo::get_batch_distance_func(
        turbo_metric_type_, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniform);
    if (turbo_ret) {
      return turbo_ret;
    }
    // Scalar fallback only available for SquaredEuclidean
    if (turbo_metric_type_ == turbo::MetricType::kSquaredEuclidean) {
      return reinterpret_cast<IndexMetric::MatrixBatchDistanceHandle>(
          ailego::DistanceBatch::SquaredEuclideanDistanceBatch<
              int8_t, 12, 2>::ComputeBatch);
    }
    return nullptr;
  }

  //! Retrieve params of Metric
  const ailego::Params &params(void) const override {
    return params_;
  }

  //! Train the metric (no training needed)
  int train(const void * /*vec*/, size_t /*dim*/) override {
    return 0;
  }

  //! Retrieve if it supports training
  bool support_train(void) const override {
    return false;
  }

  //! Normalize result (no-op: normalization is handled by reformer)
  void normalize(float * /*score*/) const override {}

  //! Retrieve if it supports normalization
  bool support_normalize(void) const override {
    return false;
  }

  //! Retrieve query metric object of this index metric
  Pointer query_metric(void) const override {
    return nullptr;
  }

  //! No query preprocessing needed for direct int8 L2
  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return nullptr;
  }

 private:
  IndexMeta meta_{};
  ailego::Params params_{};
  turbo::MetricType turbo_metric_type_{turbo::MetricType::kSquaredEuclidean};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(UniformInt8, UniformInt8Metric);

}  // namespace core
}  // namespace zvec
