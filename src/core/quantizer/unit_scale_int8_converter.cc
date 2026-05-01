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

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <ailego/pattern/defer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "../metric/metric_params.h"

namespace zvec {
namespace core {

/*! Converter for Unit-Scale Int8 Quantization (scale=1, with sq_sum extra
 *  field)
 *
 * This is the internal fast-path specialization of the Uniform Int8 Quantizer
 * for the case where the data consists of non-negative integers that fit in
 * [0, 255] (e.g. SIFT-like feature descriptors). Because `scale = 1` is
 * lossless for such data, we drop the scale multiplication entirely and use
 * asymmetric dpbusd-based distance for higher throughput.
 *
 *   int8_val = round(float_val) + bias   (bias chosen so values map to int8)
 *
 * Each vector stores:
 *   [ original_dim int8 values ] + [ 1 float: sq_sum_half ]
 * where sq_sum_half = sum(float_val^2) / 2.
 *
 * The extra sq_sum_half field enables a fast distance proxy:
 *   dist = sq_sum_half - dpbusd(query_uint8, data_int8)
 * which is monotonic with true L2 for ranking when query norm is constant.
 */
class UnitScaleInt8StreamingConverter : public IndexConverter {
 public:
  UnitScaleInt8StreamingConverter(IndexMeta::DataType /*dst_type*/) {}

  ~UnitScaleInt8StreamingConverter() override {}

  int init(const IndexMeta &index_meta, const ailego::Params &params) override {
    meta_ = index_meta;
    original_dimension_ = index_meta.dimension();

    *stats_.mutable_trained_count() = 0;
    *stats_.mutable_transformed_count() = 0;

    meta_.set_converter("UnitScaleInt8StreamingConverter", 0, params);

    // Output dimension includes the extra float (4 bytes = 4 int8 slots)
    size_t output_dimension = original_dimension_ + sizeof(float);
    meta_.set_meta(IndexMeta::DataType::DT_INT8, output_dimension);

    ailego::Params metric_params;
    metric_params.set(UNIT_SCALE_INT8_METRIC_ORIGIN_METRIC_NAME,
                      index_meta.metric_name());
    meta_.set_metric("UnitScaleInt8", 0, metric_params);

    ailego::Params reformer_params;
    meta_.set_reformer("UnitScaleInt8StreamingReformer", 0, reformer_params);

    return 0;
  }

  int cleanup(void) override {
    *stats_.mutable_trained_count() = 0;
    *stats_.mutable_transformed_count() = 0;
    return 0;
  }

  int train(IndexHolder::Pointer holder) override {
    if (!holder) {
      LOG_ERROR("UnitScaleInt8StreamingConverter: null holder in train");
      return IndexError_InvalidArgument;
    }

    ailego::ElapsedTime timer;
    AILEGO_DEFER([&]() { stats_.set_trained_costtime(timer.milli_seconds()); });

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();

    auto iter = holder->create_iterator();
    if (!iter) {
      LOG_ERROR("UnitScaleInt8StreamingConverter: failed to create iterator");
      return IndexError_Runtime;
    }

    for (; iter->is_valid(); iter->next()) {
      const float *vec = reinterpret_cast<const float *>(iter->data());
      for (size_t i = 0; i < original_dimension_; ++i) {
        float v = vec[i];
        if (!std::isfinite(v)) {
          LOG_ERROR(
              "UnitScaleInt8StreamingConverter: non-finite value in training "
              "set (record_idx=%zu, dim_idx=%zu, value=%f)",
              (size_t)*stats_.mutable_trained_count(), i, v);
          return IndexError_InvalidArgument;
        }
        global_min = std::min(global_min, v);
        global_max = std::max(global_max, v);
      }
      (*stats_.mutable_trained_count())++;
    }

    if (*stats_.mutable_trained_count() == 0) {
      LOG_ERROR("UnitScaleInt8StreamingConverter: empty training set");
      return IndexError_InvalidArgument;
    }

    // scale=1, bias shifts global_min to -128 (or -127).
    // This maps [global_min, global_min+255] -> [-128, 127].
    bias_ = -std::round(global_min) - 128.0f;

    LOG_INFO(
        "UnitScaleInt8StreamingConverter train done: costtime %zums, "
        "global_min=%f, global_max=%f, bias=%f",
        (size_t)timer.milli_seconds(), global_min, global_max, bias_);

    ailego::Params reformer_params;
    reformer_params.set(UNIT_SCALE_INT8_REFORMER_BIAS, bias_);
    meta_.set_reformer("UnitScaleInt8StreamingReformer", 0, reformer_params);

    ailego::Params conv_params = meta_.converter_params();
    conv_params.set(UNIT_SCALE_INT8_REFORMER_BIAS, bias_);
    meta_.set_converter(meta_.converter_name(), 0, conv_params);

    return 0;
  }

  int transform(IndexHolder::Pointer holder) override {
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != original_dimension_) {
      return IndexError_Mismatch;
    }

    *stats_.mutable_transformed_count() += holder->count();
    holder_ = std::make_shared<UnitScaleInt8Holder>(holder, original_dimension_,
                                                    bias_);
    return 0;
  }

  int dump(const IndexDumper::Pointer & /*dumper*/) override {
    return 0;
  }

  const Stats &stats(void) const override {
    return stats_;
  }

  IndexHolder::Pointer result(void) const override {
    return holder_;
  }

  const IndexMeta &meta(void) const override {
    return meta_;
  }

 private:
  //! IndexHolder that applies unit-scale int8 quantization on-the-fly
  class UnitScaleInt8Holder : public IndexHolder {
   public:
    class Iterator : public IndexHolder::Iterator {
     public:
      Iterator(const UnitScaleInt8Holder *owner,
               IndexHolder::Iterator::Pointer &&iter)
          : owner_(owner),
            buffer_(owner->output_dimension(), 0),
            front_iter_(std::move(iter)) {
        this->encode_record();
      }

      ~Iterator(void) override {}

      const void *data(void) const override {
        return buffer_.data();
      }

      bool is_valid(void) const override {
        return front_iter_->is_valid();
      }

      uint64_t key(void) const override {
        return front_iter_->key();
      }

      void next(void) override {
        front_iter_->next();
        this->encode_record();
      }

     private:
      void encode_record(void) {
        if (!front_iter_->is_valid()) {
          return;
        }
        const float *vec = reinterpret_cast<const float *>(front_iter_->data());
        int8_t *out = buffer_.data();
        const float bias = owner_->bias_;
        const size_t dim = owner_->original_dim_;

        float sq_sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
          float v = vec[i];
          sq_sum += v * v;
          float quantized = std::round(v + bias);
          quantized = std::max(-128.0f, std::min(127.0f, quantized));
          out[i] = static_cast<int8_t>(quantized);
        }

        // Store sq_sum_half as extra field at the tail
        float *tail = reinterpret_cast<float *>(out + dim);
        tail[0] = sq_sum / 2.0f;
      }

      const UnitScaleInt8Holder *owner_{nullptr};
      std::vector<int8_t> buffer_{};
      IndexHolder::Iterator::Pointer front_iter_{};
    };

    UnitScaleInt8Holder(IndexHolder::Pointer front, size_t original_dim,
                        float bias)
        : front_(std::move(front)), original_dim_(original_dim), bias_(bias) {}

    size_t count(void) const override {
      return front_->count();
    }

    size_t dimension(void) const override {
      return original_dim_ + sizeof(float);
    }

    size_t output_dimension(void) const {
      return dimension();
    }

    IndexMeta::DataType data_type(void) const override {
      return IndexMeta::DataType::DT_INT8;
    }

    size_t element_size(void) const override {
      return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT8,
                                      dimension());
    }

    bool multipass(void) const override {
      return front_->multipass();
    }

    IndexHolder::Iterator::Pointer create_iterator(void) override {
      auto iter = front_->create_iterator();
      return iter
                 ? IndexHolder::Iterator::Pointer(
                       new UnitScaleInt8Holder::Iterator(this, std::move(iter)))
                 : IndexHolder::Iterator::Pointer();
    }

   private:
    IndexHolder::Pointer front_{};
    size_t original_dim_{0};
    float bias_{0.0f};
  };

  IndexMeta meta_{};
  Stats stats_{};
  IndexHolder::Pointer holder_{};
  size_t original_dimension_{0};
  float bias_{0.0f};
};

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(UnitScaleInt8StreamingConverter,
                                       UnitScaleInt8StreamingConverter,
                                       IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
