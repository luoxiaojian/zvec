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
#include <cstring>
#include <limits>
#include <memory>
#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>

namespace zvec {
namespace core {

/*! Reformer for Uniform Int8 Quantization (Global Scale)
 *
 * Uses a global scale/bias (computed by UniformInt8StreamingConverter) to
 * quantize query vectors and build-time record vectors to int8.
 * No per-vector extras are appended — the output is pure int8.
 */
class UniformInt8StreamingReformer : public IndexReformer {
 public:
  //! Constructor
  UniformInt8StreamingReformer(IndexMeta::DataType /*dst_type*/) {}

  //! Initialize Reformer
  //! During build, scale/bias come from converter's train().
  //! During search, first init may have empty params (from converter init);
  //! the real params are provided via a second init after the index is opened.
  int init(const ailego::Params &params) override {
    params.get(UNIFORM_INT8_REFORMER_SCALE, &scale_);
    params.get(UNIFORM_INT8_REFORMER_BIAS, &bias_);
    // Precompute reciprocal for score normalization
    // int8_l2 = scale^2 * real_l2, so real_l2 = int8_l2 / scale^2
    scale_reciprocal_sq_ = (scale_ != 0.0f) ? (1.0f / (scale_ * scale_)) : 1.0f;

    LOG_INFO("UniformInt8StreamingReformer init: scale=%f, bias=%f", scale_,
             bias_);
    return 0;
  }

  //! Cleanup Reformer
  int cleanup(void) override {
    return 0;
  }

  //! Load index from container
  int load(IndexStorage::Pointer) override {
    return 0;
  }

  //! Unload index
  int unload(void) override {
    return 0;
  }

  //! Transform a single query: float → int8
  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = qmeta.data_type();
    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = qmeta;
    ometa->set_meta(IndexMeta::DataType::DT_INT8, qmeta.dimension());
    out->resize(ometa->element_size());

    const float *vec = reinterpret_cast<const float *>(query);
    int8_t *ovec = reinterpret_cast<int8_t *>(&(*out)[0]);
    quantize(vec, qmeta.dimension(), ovec);

    return 0;
  }

  //! Transform batch queries: float → int8
  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = qmeta.data_type();
    if (ft != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = qmeta;
    ometa->set_meta(IndexMeta::DataType::DT_INT8, qmeta.dimension());
    out->resize(count * ometa->element_size());

    const float *vec = reinterpret_cast<const float *>(query);
    for (size_t i = 0; i < count; ++i) {
      int8_t *ovec =
          reinterpret_cast<int8_t *>(&(*out)[i * ometa->element_size()]);
      quantize(&vec[i * qmeta.dimension()], qmeta.dimension(), ovec);
    }

    return 0;
  }

  //! Convert a single record: float → int8 (used during build)
  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = rmeta.data_type();
    if (ft != IndexMeta::DataType::DT_FP32 ||
        rmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = rmeta;
    ometa->set_meta(IndexMeta::DataType::DT_INT8, rmeta.dimension());
    out->resize(ometa->element_size());

    const float *vec = reinterpret_cast<const float *>(record);
    int8_t *ovec = reinterpret_cast<int8_t *>(&(*out)[0]);
    quantize(vec, rmeta.dimension(), ovec);

    return 0;
  }

  //! Convert batch records: float → int8
  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    IndexMeta::DataType ft = rmeta.data_type();
    if (ft != IndexMeta::DataType::DT_FP32 ||
        rmeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = rmeta;
    ometa->set_meta(IndexMeta::DataType::DT_INT8, rmeta.dimension());
    out->resize(count * ometa->element_size());

    const float *vec = reinterpret_cast<const float *>(records);
    for (size_t i = 0; i < count; ++i) {
      int8_t *ovec =
          reinterpret_cast<int8_t *>(&(*out)[i * ometa->element_size()]);
      quantize(&vec[i * rmeta.dimension()], rmeta.dimension(), ovec);
    }

    return 0;
  }

  //! Normalize results: convert int8 L2 distances back to float L2 distances
  int normalize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                IndexDocumentList &result) const override {
    for (auto &it : result) {
      *it.mutable_score() *= scale_reciprocal_sq_;
    }
    return 0;
  }

  //! Support revert (int8 → float)
  bool need_revert() const override {
    return true;
  }

  //! Revert: convert int8 vector back to float
  int revert(const void *in, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    size_t dim = qmeta.dimension();
    out->resize(dim * sizeof(float));
    float *out_buf = reinterpret_cast<float *>(out->data());
    const int8_t *buf = reinterpret_cast<const int8_t *>(in);

    // Reverse quantization: float_val = (int8_val - bias) / scale
    float inv_scale = (scale_ != 0.0f) ? (1.0f / scale_) : 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      out_buf[i] = (static_cast<float>(buf[i]) - bias_) * inv_scale;
    }

    return 0;
  }

 private:
  //! Quantize float vector to int8 using global scale/bias
  inline void quantize(const float *in, size_t dim, int8_t *out) const {
    for (size_t i = 0; i < dim; ++i) {
      float v = std::round(in[i] * scale_ + bias_);
      v = std::max(-127.0f, std::min(127.0f, v));
      out[i] = static_cast<int8_t>(v);
    }
  }

  //! Members
  float scale_{0.0f};
  float bias_{0.0f};
  float scale_reciprocal_sq_{1.0f};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(UniformInt8StreamingReformer,
                                      UniformInt8StreamingReformer,
                                      IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
