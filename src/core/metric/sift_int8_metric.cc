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

#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "metric_params.h"

namespace zvec {
namespace core {

/*! Index Metric for SIFT Int8 Quantization (scale=1, with sq_sum extra field)
 *
 * Uses dpbusd-based inner product with per-vector sq_sum_half initialization.
 * Distance = sq_sum_half - dpbusd(query_uint8, data_int8).
 * This is a monotonic proxy for L2 ranking.
 *
 * Vector layout: [ original_dim int8 ] + [ 4 bytes float: sq_sum_half ]
 * Query layout:  [ original_dim uint8 ] + [ 4 bytes unused ]
 * dim passed to distance functions = original_dim + 4.
 */
class SiftInt8Metric : public IndexMetric {
 public:
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8) {
      LOG_ERROR("SiftInt8Metric: unsupported type %d", meta.data_type());
      return IndexError_Unsupported;
    }

    std::string metric_name;
    index_params.get(SIFT_INT8_METRIC_ORIGIN_METRIC_NAME, &metric_name);
    if (metric_name.empty()) {
      LOG_ERROR("SiftInt8Metric: param %s is required",
                SIFT_INT8_METRIC_ORIGIN_METRIC_NAME.c_str());
      return IndexError_InvalidArgument;
    }

    if (metric_name != "SquaredEuclidean") {
      LOG_ERROR("SiftInt8Metric: only SquaredEuclidean supported, got %s",
                metric_name.c_str());
      return IndexError_Unsupported;
    }

    meta_ = meta;
    params_ = index_params;

    LOG_INFO("SiftInt8Metric initialized: dimension=%u", meta_.dimension());
    return 0;
  }

  int cleanup(void) override { return 0; }

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
          turbo::QuantizeType::kSift);
      if (turbo_ret) {
        return turbo_ret;
      }
    }
    return nullptr;
  }

  MatrixBatchDistance batch_distance(void) const override {
    auto turbo_ret = turbo::get_batch_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kSift);
    if (turbo_ret) {
      return turbo_ret;
    }
    return nullptr;
  }

  const ailego::Params &params(void) const override { return params_; }

  int train(const void * /*vec*/, size_t /*dim*/) override { return 0; }

  bool support_train(void) const override { return false; }

  void normalize(float * /*score*/) const override {}

  bool support_normalize(void) const override { return false; }

  Pointer query_metric(void) const override { return nullptr; }

  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return nullptr;
  }

 private:
  IndexMeta meta_{};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(SiftInt8, SiftInt8Metric);

}  // namespace core
}  // namespace zvec
