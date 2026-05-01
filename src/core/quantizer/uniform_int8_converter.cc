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
#include "../metric/metric_params.h"

namespace zvec {
namespace core {

/*! Converter for Uniform Int8 Quantization (Global Scale)
 *
 * Unlike IntegerStreamingConverter which uses per-vector scale/bias,
 * this converter computes a single global scale/bias from the entire dataset.
 * All vectors share the same quantization parameters, enabling direct int8
 * L2 distance computation without per-vector reconstruction.
 */
class UniformInt8StreamingConverter : public IndexConverter {
 public:
  //! Constructor
  UniformInt8StreamingConverter(IndexMeta::DataType /*dst_type*/) {}

  //! Destructor
  ~UniformInt8StreamingConverter() override {}

  //! Initialize Converter
  int init(const IndexMeta &index_meta, const ailego::Params &params) override {
    meta_ = index_meta;
    original_dimension_ = index_meta.dimension();

    // Store converter info in meta
    meta_.set_converter("UniformInt8StreamingConverter", 0, params);

    // Set data type to INT8, dimension stays the same (no per-vector extras)
    meta_.set_meta(IndexMeta::DataType::DT_INT8, original_dimension_);

    // Set metric to our direct int8 L2 metric
    ailego::Params metric_params;
    metric_params.set(UNIFORM_QUANTIZED_INT8_METRIC_ORIGIN_METRIC_NAME,
                      index_meta.metric_name());
    meta_.set_metric("UniformQuantizedInt8", 0, metric_params);

    // Set reformer name now (scale/bias will be updated in train()).
    // During search-only path, the stored meta provides the real params.
    ailego::Params reformer_params;
    meta_.set_reformer("UniformInt8StreamingReformer", 0, reformer_params);

    return 0;
  }

  //! Cleanup Converter
  int cleanup(void) override {
    *stats_.mutable_transformed_count() = 0;
    return 0;
  }

  //! Train: compute global min/max and derive scale/bias
  int train(IndexHolder::Pointer holder) override {
    if (!holder) {
      LOG_ERROR("UniformInt8StreamingConverter: null holder in train");
      return IndexError_InvalidArgument;
    }

    ailego::ElapsedTime timer;
    AILEGO_DEFER([&]() { stats_.set_trained_costtime(timer.milli_seconds()); });

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();

    auto iter = holder->create_iterator();
    if (!iter) {
      LOG_ERROR("UniformInt8StreamingConverter: failed to create iterator");
      return IndexError_Runtime;
    }

    for (; iter->is_valid(); iter->next()) {
      const float *vec = reinterpret_cast<const float *>(iter->data());
      for (size_t i = 0; i < original_dimension_; ++i) {
        global_min = std::min(global_min, vec[i]);
        global_max = std::max(global_max, vec[i]);
      }
      (*stats_.mutable_trained_count())++;
    }

    // Compute global scale and bias: maps [min, max] → [-127, 127]
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    float range = global_max - global_min;
    if (range <= 254.0f) {
      // Range fits within int8 dynamic range: use scale=1 for lossless
      // integer quantization (e.g. SIFT data with integer values in [0,218])
      scale_ = 1.0f;
      bias_ = -std::round(global_min) - 127.0f;
    } else {
      scale_ = 254.0f / std::max(range, epsilon);
      bias_ = -global_min * scale_ - 127.0f;
    }

    LOG_INFO(
        "UniformInt8StreamingConverter train done: costtime %zums, "
        "global_min=%f, global_max=%f, scale=%f, bias=%f",
        (size_t)timer.milli_seconds(), global_min, global_max, scale_, bias_);

    // Now configure the reformer with the computed scale/bias
    ailego::Params reformer_params;
    reformer_params.set(UNIFORM_INT8_REFORMER_SCALE, scale_);
    reformer_params.set(UNIFORM_INT8_REFORMER_BIAS, bias_);
    meta_.set_reformer("UniformInt8StreamingReformer", 0, reformer_params);

    // Also store scale/bias in converter params for persistence
    ailego::Params conv_params = meta_.converter_params();
    conv_params.set(UNIFORM_INT8_REFORMER_SCALE, scale_);
    conv_params.set(UNIFORM_INT8_REFORMER_BIAS, bias_);
    meta_.set_converter(meta_.converter_name(), 0, conv_params);

    return 0;
  }

  //! Transform: wrap holder to produce quantized int8 data
  int transform(IndexHolder::Pointer holder) override {
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != original_dimension_) {
      return IndexError_Mismatch;
    }

    *stats_.mutable_transformed_count() += holder->count();
    holder_ = std::make_shared<UniformInt8Holder>(holder, original_dimension_,
                                                  scale_, bias_);
    return 0;
  }

  //! Dump index into storage
  int dump(const IndexDumper::Pointer & /*dumper*/) override {
    return 0;
  }

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve a holder as result
  IndexHolder::Pointer result(void) const override {
    return holder_;
  }

  //! Retrieve Index Meta
  const IndexMeta &meta(void) const override {
    return meta_;
  }

 private:
  //! IndexHolder that applies uniform int8 quantization on-the-fly
  class UniformInt8Holder : public IndexHolder {
   public:
    class Iterator : public IndexHolder::Iterator {
     public:
      Iterator(const UniformInt8Holder *owner,
               IndexHolder::Iterator::Pointer &&iter)
          : owner_(owner),
            buffer_(owner->dimension(), 0),
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
        if (front_iter_->is_valid()) {
          const float *vec =
              reinterpret_cast<const float *>(front_iter_->data());
          int8_t *out = reinterpret_cast<int8_t *>(buffer_.data());
          float scale = owner_->scale_;
          float bias = owner_->bias_;
          for (size_t i = 0; i < owner_->original_dim_; ++i) {
            float v = std::round(vec[i] * scale + bias);
            v = std::max(-127.0f, std::min(127.0f, v));
            out[i] = static_cast<int8_t>(v);
          }
        }
      }

      const UniformInt8Holder *owner_{nullptr};
      std::vector<uint8_t> buffer_{};
      IndexHolder::Iterator::Pointer front_iter_{};
    };

    UniformInt8Holder(IndexHolder::Pointer front, size_t original_dim,
                      float scale, float bias)
        : front_(std::move(front)),
          original_dim_(original_dim),
          scale_(scale),
          bias_(bias) {}

    size_t count(void) const override {
      return front_->count();
    }

    size_t dimension(void) const override {
      return original_dim_;
    }

    IndexMeta::DataType data_type(void) const override {
      return IndexMeta::DataType::DT_INT8;
    }

    size_t element_size(void) const override {
      return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT8,
                                      original_dim_);
    }

    bool multipass(void) const override {
      return front_->multipass();
    }

    IndexHolder::Iterator::Pointer create_iterator(void) override {
      auto iter = front_->create_iterator();
      return iter ? IndexHolder::Iterator::Pointer(
                        new UniformInt8Holder::Iterator(this, std::move(iter)))
                  : IndexHolder::Iterator::Pointer();
    }

   private:
    IndexHolder::Pointer front_{};
    size_t original_dim_{0};
    float scale_{0.0f};
    float bias_{0.0f};
  };

  //! Members
  IndexMeta meta_{};
  Stats stats_{};
  IndexHolder::Pointer holder_{};
  size_t original_dimension_{0};
  float scale_{0.0f};
  float bias_{0.0f};
};

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(UniformInt8StreamingConverter,
                                       UniformInt8StreamingConverter,
                                       IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
