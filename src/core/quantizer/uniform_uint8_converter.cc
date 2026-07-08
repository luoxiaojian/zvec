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
#include <cstdint>
#include <limits>
#include <vector>
#include <ailego/pattern/defer.h>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>
#include "../metric/metric_params.h"

namespace zvec {
namespace core {

class UniformUint8StreamingConverter : public IndexConverter {
 public:
  UniformUint8StreamingConverter(IndexMeta::DataType /*dst_type*/) {}
  ~UniformUint8StreamingConverter() override {}

  int init(const IndexMeta &index_meta, const ailego::Params &params) override {
    meta_ = index_meta;
    original_dimension_ = index_meta.dimension();
    encoded_dimension_ = original_dimension_ + kTailBytes;
    *stats_.mutable_trained_count() = 0;
    *stats_.mutable_transformed_count() = 0;

    meta_.set_converter("UniformUint8StreamingConverter", 0, params);
    meta_.set_meta(IndexMeta::DataType::DT_INT8, encoded_dimension_);

    ailego::Params metric_params;
    metric_params.set(UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME,
                      index_meta.metric_name());
    meta_.set_metric("UniformUint8", 0, metric_params);

    params.get(UNIFORM_UINT8_REFORMER_SCALE, &scale_);
    params.get(UNIFORM_UINT8_REFORMER_BIAS, &bias_);
    if (scale_ != 0.0f) {
      ailego::Params reformer_params;
      reformer_params.set(UNIFORM_UINT8_REFORMER_SCALE, scale_);
      reformer_params.set(UNIFORM_UINT8_REFORMER_BIAS, bias_);
      meta_.set_reformer("UniformUint8StreamingReformer", 0, reformer_params);
    }
    return 0;
  }

  int cleanup(void) override {
    *stats_.mutable_trained_count() = 0;
    *stats_.mutable_transformed_count() = 0;
    return 0;
  }

  int train(IndexHolder::Pointer holder) override {
    if (!holder) {
      LOG_ERROR("UniformUint8StreamingConverter: null holder in train");
      return IndexError_InvalidArgument;
    }

    ailego::ElapsedTime timer;
    AILEGO_DEFER([&]() { stats_.set_trained_costtime(timer.milli_seconds()); });

    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();
    bool all_integer = true;

    auto iter = holder->create_iterator();
    if (!iter) {
      LOG_ERROR("UniformUint8StreamingConverter: failed to create iterator");
      return IndexError_Runtime;
    }

    for (; iter->is_valid(); iter->next()) {
      const float *vec = reinterpret_cast<const float *>(iter->data());
      for (size_t i = 0; i < original_dimension_; ++i) {
        float v = vec[i];
        if (!std::isfinite(v)) {
          LOG_ERROR(
              "UniformUint8StreamingConverter: non-finite value in training "
              "set (record_idx=%zu, dim_idx=%zu, value=%f)",
              (size_t)*stats_.mutable_trained_count(), i, v);
          return IndexError_InvalidArgument;
        }
        global_min = std::min(global_min, v);
        global_max = std::max(global_max, v);
        if (all_integer && std::floor(v) != v) {
          all_integer = false;
        }
      }
      (*stats_.mutable_trained_count())++;
    }

    if (*stats_.mutable_trained_count() == 0) {
      LOG_ERROR("UniformUint8StreamingConverter: empty training set");
      return IndexError_InvalidArgument;
    }

    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    float range = global_max - global_min;
    if (all_integer && range <= 255.0f) {
      scale_ = 1.0f;
      bias_ = -global_min;
    } else {
      scale_ = 255.0f / std::max(range, epsilon);
      bias_ = -global_min * scale_;
    }

    LOG_INFO(
        "UniformUint8StreamingConverter train done: costtime %zums, "
        "global_min=%f, global_max=%f, scale=%f, bias=%f",
        (size_t)timer.milli_seconds(), global_min, global_max, scale_, bias_);

    ailego::Params reformer_params;
    reformer_params.set(UNIFORM_UINT8_REFORMER_SCALE, scale_);
    reformer_params.set(UNIFORM_UINT8_REFORMER_BIAS, bias_);
    meta_.set_reformer("UniformUint8StreamingReformer", 0, reformer_params);

    ailego::Params conv_params = meta_.converter_params();
    conv_params.set(UNIFORM_UINT8_REFORMER_SCALE, scale_);
    conv_params.set(UNIFORM_UINT8_REFORMER_BIAS, bias_);
    meta_.set_converter(meta_.converter_name(), 0, conv_params);
    return 0;
  }

  int transform(IndexHolder::Pointer holder) override {
    if (holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != original_dimension_) {
      return IndexError_Mismatch;
    }
    *stats_.mutable_transformed_count() += holder->count();
    holder_ = std::make_shared<UniformUint8Holder>(
        holder, original_dimension_, encoded_dimension_, scale_, bias_);
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
  static constexpr size_t kTailBytes = sizeof(int32_t) * 2;

  class UniformUint8Holder : public IndexHolder {
   public:
    class Iterator : public IndexHolder::Iterator {
     public:
      Iterator(const UniformUint8Holder *owner,
               IndexHolder::Iterator::Pointer &&iter)
          : owner_(owner),
            buffer_(owner->encoded_dim(), 0),
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
        const size_t dim = owner_->original_dim_;

        if (owner_->quantize_func_ != nullptr) {
          owner_->quantize_func_(vec, dim, owner_->scale_, owner_->bias_, out);
        } else {
          auto *u8_out = reinterpret_cast<uint8_t *>(out);
          for (size_t i = 0; i < dim; ++i) {
            float v = std::round(vec[i] * owner_->scale_ + owner_->bias_);
            v = std::max(0.0f, std::min(255.0f, v));
            u8_out[i] = static_cast<uint8_t>(v);
          }
        }

        const auto *u8 = reinterpret_cast<const uint8_t *>(out);
        int64_t sum = 0;
        int64_t sum_sq = 0;
        for (size_t i = 0; i < dim; ++i) {
          int v = static_cast<int>(u8[i]);
          sum += v;
          sum_sq += v * v;
        }
        auto *tail = reinterpret_cast<int32_t *>(out + dim);
        tail[0] = static_cast<int32_t>(sum);
        tail[1] = static_cast<int32_t>(sum_sq);
      }

      const UniformUint8Holder *owner_{nullptr};
      std::vector<int8_t> buffer_{};
      IndexHolder::Iterator::Pointer front_iter_{};
    };

    UniformUint8Holder(IndexHolder::Pointer front, size_t original_dim,
                       size_t encoded_dim, float scale, float bias)
        : front_(std::move(front)),
          original_dim_(original_dim),
          encoded_dim_(encoded_dim),
          scale_(scale),
          bias_(bias),
          quantize_func_(
              turbo::get_uniform_uint8_quantize_func(turbo::DataType::kInt8)) {}

    size_t count(void) const override {
      return front_->count();
    }

    size_t dimension(void) const override {
      return encoded_dim_;
    }

    IndexMeta::DataType data_type(void) const override {
      return IndexMeta::DataType::DT_INT8;
    }

    size_t element_size(void) const override {
      return IndexMeta::ElementSizeof(IndexMeta::DataType::DT_INT8,
                                      encoded_dim_);
    }

    bool multipass(void) const override {
      return front_->multipass();
    }

    IndexHolder::Iterator::Pointer create_iterator(void) override {
      auto iter = front_->create_iterator();
      return iter ? IndexHolder::Iterator::Pointer(
                        new UniformUint8Holder::Iterator(this, std::move(iter)))
                  : IndexHolder::Iterator::Pointer();
    }

    size_t encoded_dim() const {
      return encoded_dim_;
    }

   private:
    IndexHolder::Pointer front_{};
    size_t original_dim_{0};
    size_t encoded_dim_{0};
    float scale_{0.0f};
    float bias_{0.0f};
    turbo::UniformQuantizeFunc quantize_func_{nullptr};
  };

  IndexMeta meta_{};
  Stats stats_{};
  IndexHolder::Pointer holder_{};
  size_t original_dimension_{0};
  size_t encoded_dimension_{0};
  float scale_{0.0f};
  float bias_{0.0f};
};

INDEX_FACTORY_REGISTER_CONVERTER_ALIAS(UniformUint8StreamingConverter,
                                       UniformUint8StreamingConverter,
                                       IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
