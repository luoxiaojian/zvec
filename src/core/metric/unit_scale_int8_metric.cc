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

/*! Query-side Metric for Unit-Scale Int8 Quantization.
 *
 * Used exclusively on the search path: computes the distance proxy
 *   dist = sq_sum_half - dpbusd(query_uint8, data_int8)
 * via `unit_scale_squared_euclidean_int8_distance`.  The first operand is
 * interpreted as uint8 (query), the second as int8 (database vector).
 *
 * This metric is NOT used for building the graph (add path), since both
 * operands during add are int8 database vectors — see UnitScaleInt8Metric
 * for the symmetric int8×int8 L2 kernel used in that path.
 */
class UnitScaleInt8QueryMetric : public IndexMetric {
 public:
  int init(const IndexMeta &meta, const ailego::Params &index_params) override {
    if (meta.data_type() != IndexMeta::DataType::DT_INT8) {
      LOG_ERROR("UnitScaleInt8QueryMetric: unsupported type %d",
                meta.data_type());
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
    return turbo::get_distance_func(turbo::MetricType::kSquaredEuclidean,
                                    turbo::DataType::kInt8,
                                    turbo::QuantizeType::kUnitScale);
  }

  MatrixBatchDistance batch_distance(void) const override {
    return turbo::get_batch_distance_func(turbo::MetricType::kSquaredEuclidean,
                                          turbo::DataType::kInt8,
                                          turbo::QuantizeType::kUnitScale);
  }

  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m == 1 && n == 1) {
      return distance();
    }
    return nullptr;
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
  Pointer query_metric(void) const override {
    return nullptr;
  }
  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return nullptr;
  }

 private:
  IndexMeta meta_{};
  ailego::Params params_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(UnitScaleInt8Query,
                                    UnitScaleInt8QueryMetric);

/*! Index Metric for Unit-Scale Int8 Quantization (scale=1, with sq_sum extra
 *  field)
 *
 * Vector layout: [ original_dim int8 ] + [ 4 bytes float: sq_sum_half ]
 * Total `dim` passed to distance functions = original_dim + 4.
 *
 * Add path (graph construction): both operands are int8 database vectors.
 *   Uses symmetric int8×int8 L2: sum((a[i] - b[i])^2) over original_dim
 *   elements, ignoring the 4-byte sq_sum_half tail.  Ordering is equivalent
 *   to the float L2 on the pre-quantized values (scale=1).
 *
 * Search path: the query is uint8 (via query_metric()).  Distance is
 *   dist = sq_sum_half - dpbusd(query_uint8, data_int8).
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

    // Resolve the symmetric int8×int8 L2 kernel (reused from uniform_int8)
    // for the add path.  `dim - 4` skips the sq_sum_half tail.
    add_distance_ = turbo::get_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniform);
    add_batch_distance_ = turbo::get_batch_distance_func(
        turbo::MetricType::kSquaredEuclidean, turbo::DataType::kInt8,
        turbo::QuantizeType::kUniform);

    // Create the query-side metric (dpbusd).
    query_metric_ = IndexFactory::CreateMetric("UnitScaleInt8Query");
    if (!query_metric_) {
      LOG_ERROR(
          "UnitScaleInt8Metric: failed to create UnitScaleInt8QueryMetric");
      return IndexError_NoExist;
    }
    int ret = query_metric_->init(meta, ailego::Params());
    if (ret != 0) {
      LOG_ERROR("UnitScaleInt8Metric: failed to init query metric");
      return ret;
    }

    LOG_INFO("UnitScaleInt8Metric initialized: dimension=%u", meta_.dimension());
    return 0;
  }

  int cleanup(void) override {
    query_metric_.reset();
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

  //! Add-path distance: symmetric int8×int8 L2 ignoring 4-byte tail.
  MatrixDistance distance(void) const override {
    return distance_matrix(1, 1);
  }

  MatrixDistance distance_matrix(size_t m, size_t n) const override {
    if (m == 1 && n == 1 && add_distance_) {
      auto kernel = add_distance_;
      return [kernel](const void *a, const void *b, size_t dim, float *out) {
        // Skip the sq_sum_half tail (4 bytes == 4 int8 slots).
        kernel(a, b, dim > sizeof(float) ? dim - sizeof(float) : dim, out);
      };
    }
    return nullptr;
  }

  MatrixBatchDistance batch_distance(void) const override {
    if (add_batch_distance_) {
      auto kernel = add_batch_distance_;
      return [kernel](const void **vectors, const void *q, size_t num,
                      size_t dim, float *out) {
        kernel(vectors, q, num, dim > sizeof(float) ? dim - sizeof(float) : dim,
               out);
      };
    }
    return nullptr;
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

  //! Search path: dpbusd-based distance.
  Pointer query_metric(void) const override {
    return query_metric_;
  }

  DistanceBatchQueryPreprocessFunc get_query_preprocess_func() const override {
    return nullptr;
  }

 private:
  IndexMeta meta_{};
  ailego::Params params_{};
  IndexMetric::Pointer query_metric_{};
  turbo::DistanceFunc add_distance_{};
  turbo::BatchDistanceFunc add_batch_distance_{};
};

INDEX_FACTORY_REGISTER_METRIC_ALIAS(UnitScaleInt8, UnitScaleInt8Metric);

}  // namespace core
}  // namespace zvec
