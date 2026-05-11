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

/*! Index Metric for Unit-Scale Int8 Quantization (scale=1, with sq_sum extra
 *  field)
 *
 * Vector layout: [ original_dim int8 ] + [ 4 bytes float: sq_sum_half ]
 * Total `dim` passed to distance functions = original_dim + 4.
 *
 * Single metric providing both search and pairwise distance functions:
 *
 * Search path (distance/batch_distance): dpbusd proxy distance.
 *   dist = sq_sum_half - dpbusd(query_uint8, data_int8)
 *   Query must be preprocessed (int8 -> uint8 via get_query_preprocess_func)
 *   before calling distance().
 *
 * Add path / pairwise (pairwise_distance/pairwise_batch_distance):
 *   Symmetric int8×int8 L2: sum((a[i] - b[i])^2) over original_dim elements,
 *   ignoring the 4-byte sq_sum_half tail. No preprocessing required.
 */
class UnitScaleInt8Metric : public IndexMetric {
 public:
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8) {
      LOG_ERROR("UnitScaleInt8Metric: unsupported type %d", meta.data_type());
      return IndexError_Unsupported;
    }

    std::string metric_name;
    index_params.get(UNIT_SCALE_INT8_METRIC_ORIGIN_METRIC_NAME, &metric_name);
    if (metric_name.empty()) {
      LOG_ERROR("UnitScaleInt8Metric: param %s is required",
                UNIT_SCALE_INT8_METRIC_ORIGIN_METRIC_NAME.c_str());
      return IndexError_InvalidArgument;
    }

    if (metric_name != "SquaredEuclidean") {
      LOG_ERROR("UnitScaleInt8Metric: only SquaredEuclidean supported, got %s",
                metric_name.c_str());
      return IndexError_Unsupported;
    }

    meta_ = meta;
    params_ = index_params;

    LOG_INFO("UnitScaleInt8Metric initialized: dimension=%u",
             meta_.dimension());
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

  //! Search-path distance: dpbusd proxy (needs preprocessed uint8 query).
  MatrixDistance distance(void) const override {
    return turbo::get_distance_func(turbo::MetricType::kSquaredEuclidean,
                                    turbo::DataType::kInt8,
                                    turbo::QuantizeType::kUnitScale);
  }

  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m == 1 && n == 1) {
      return distance();
    }
    return nullptr;
  }

  MatrixBatchDistance batch_distance(void) const override {
    return turbo::get_batch_distance_func(turbo::MetricType::kSquaredEuclidean,
                                          turbo::DataType::kInt8,
                                          turbo::QuantizeType::kUnitScale);
  }

  //! Side data size: the 4-byte `sq_sum_half` tail at the end of each vector.
  //! Storage backends may split this tail into a separate flat array for
  //! cache efficiency; the corresponding split-layout batch kernel will be
  //! used in that case.
  size_t side_data_size_per_vector(void) const override {
    return sizeof(float);
  }

  //! Split-layout batch kernel: vectors are pure int8 (length = original_dim),
  //! per-vector side data (here a single 4-byte `sq_sum_half`) is supplied
  //! through the `side_data` pointer array.
  MatrixBatchDistanceSplit batch_distance_split(void) const override {
    return turbo::get_batch_distance_split_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUnitScale);
  }

  //! Query preprocess: shift int8 -> uint8 (+128) for dpbusd distance.
  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return turbo::get_query_preprocess_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUnitScale);
  }

  //! Pairwise (data-to-data) distance: symmetric int8×int8 L2.
  //! Implemented in turbo. No preprocessing required.
  MatrixDistance pairwise_distance(void) const override {
    return turbo::get_pairwise_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUnitScale);
  }

  MatrixBatchDistance pairwise_batch_distance(void) const override {
    return turbo::get_pairwise_batch_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUnitScale);
  }

  const ailego::Params &params(void) const override {
    return params_;
  }
  int train(const void *, size_t) override {
    return 0;
  }
  bool support_train(void) const override {
    return false;
  }
  void normalize(float *) const override {}
  bool support_normalize(void) const override {
    return false;
  }

  //! No sub-metric (single metric design).
  Pointer query_metric(void) const override {
    return nullptr;
  }

 private:
  IndexMeta meta_{};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(UnitScaleInt8, UnitScaleInt8Metric);

}  // namespace core
}  // namespace zvec
