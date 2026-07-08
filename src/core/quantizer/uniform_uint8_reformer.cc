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
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace core {

class UniformUint8StreamingReformer : public IndexReformer {
 public:
  UniformUint8StreamingReformer(IndexMeta::DataType /*dst_type*/) {}
  ~UniformUint8StreamingReformer() override {}

  int init(const ailego::Params &params) override {
    bool has_scale = params.get(UNIFORM_UINT8_REFORMER_SCALE, &scale_);
    bool has_bias = params.get(UNIFORM_UINT8_REFORMER_BIAS, &bias_);
    if (!has_scale || !has_bias) {
      LOG_ERROR(
          "UniformUint8StreamingReformer init: missing required params "
          "(scale_present=%d, bias_present=%d)",
          (int)has_scale, (int)has_bias);
      initialized_ = false;
      return IndexError_InvalidArgument;
    }
    if (!std::isfinite(scale_) || scale_ == 0.0f || !std::isfinite(bias_)) {
      LOG_ERROR(
          "UniformUint8StreamingReformer: invalid params scale=%f, bias=%f",
          scale_, bias_);
      initialized_ = false;
      return IndexError_InvalidArgument;
    }
    scale_reciprocal_sq_ = 1.0f / (scale_ * scale_);
    quantize_func_ =
        turbo::get_uniform_uint8_quantize_func(turbo::DataType::kInt8);
    initialized_ = true;
    return 0;
  }

  int cleanup(void) override {
    return 0;
  }

  int load(IndexStorage::Pointer) override {
    return 0;
  }

  int unload(void) override {
    return 0;
  }

  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(query, qmeta, 1, out, ometa);
  }

  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(query, qmeta, count, out, ometa);
  }

  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    return do_quantize(record, rmeta, 1, out, ometa);
  }

  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    return do_quantize(records, rmeta, count, out, ometa);
  }

  int normalize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                IndexDocumentList &result) const override {
    if (!initialized_) {
      return IndexError_Runtime;
    }
    for (auto &it : result) {
      *it.mutable_score() *= scale_reciprocal_sq_;
    }
    return 0;
  }

  bool need_revert() const override {
    return true;
  }

  int revert(const void *in, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    if (!initialized_) {
      return IndexError_Runtime;
    }
    size_t dim = original_dim(qmeta.dimension());
    out->resize(dim * sizeof(float));
    float *out_buf = reinterpret_cast<float *>(out->data());
    const auto *buf = reinterpret_cast<const uint8_t *>(in);
    float inv_scale = 1.0f / scale_;
    for (size_t i = 0; i < dim; ++i) {
      out_buf[i] = (static_cast<float>(buf[i]) - bias_) * inv_scale;
    }
    return 0;
  }

 private:
  static constexpr size_t kTailBytes = sizeof(int32_t) * 2;

  static size_t original_dim(size_t encoded_dim) {
    return encoded_dim > kTailBytes ? encoded_dim - kTailBytes : 0;
  }

  int do_quantize(const void *src, const IndexQueryMeta &smeta, uint32_t count,
                  std::string *out, IndexQueryMeta *ometa) const {
    if (!initialized_) {
      LOG_ERROR("UniformUint8StreamingReformer: quantize called before init");
      return IndexError_Runtime;
    }
    if (smeta.data_type() != IndexMeta::DataType::DT_FP32 ||
        smeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    const size_t src_dim = smeta.dimension();
    const size_t encoded_dim = src_dim + kTailBytes;
    *ometa = smeta;
    ometa->set_meta(IndexMeta::DataType::DT_INT8, encoded_dim);
    const size_t out_stride = ometa->element_size();
    out->resize(static_cast<size_t>(count) * out_stride);

    const float *vec = reinterpret_cast<const float *>(src);
    int8_t *ovec = reinterpret_cast<int8_t *>(&(*out)[0]);
    for (uint32_t i = 0; i < count; ++i) {
      encode(vec + i * src_dim, src_dim, ovec + i * out_stride);
    }
    return 0;
  }

  void encode(const float *in, size_t dim, int8_t *out) const {
    if (quantize_func_ != nullptr) {
      quantize_func_(in, dim, scale_, bias_, out);
    } else {
      auto *u8_out = reinterpret_cast<uint8_t *>(out);
      for (size_t i = 0; i < dim; ++i) {
        float v = std::round(in[i] * scale_ + bias_);
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

  float scale_{0.0f};
  float bias_{0.0f};
  float scale_reciprocal_sq_{1.0f};
  bool initialized_{false};
  turbo::UniformQuantizeFunc quantize_func_{nullptr};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(UniformUint8StreamingReformer,
                                      UniformUint8StreamingReformer,
                                      IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
