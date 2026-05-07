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

#include <core/quantizer/quantizer_params.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace core {

/*! Reformer for SIFT Int8 Quantization (scale=1, with sq_sum extra field)
 *
 * Query quantization: float → uint8 (value = clamp(round(float), 0, 255))
 * Record quantization: float → int8 (value = round(float) + bias) + sq_sum/2
 *
 * The query is stored as pure uint8 with no extra fields, since the query
 * norm is constant across all distance comparisons and cancels out in ranking.
 */
class SiftInt8StreamingReformer : public IndexReformer {
 public:
  SiftInt8StreamingReformer(IndexMeta::DataType /*dst_type*/) {}

  int init(const ailego::Params &params) override {
    bool has_bias = params.get(SIFT_INT8_REFORMER_BIAS, &bias_);

    if (!has_bias) {
      initialized_ = false;
      LOG_INFO(
          "SiftInt8StreamingReformer init: bias not ready yet, "
          "waiting for second init");
      return 0;
    }

    if (!std::isfinite(bias_)) {
      LOG_ERROR("SiftInt8StreamingReformer: invalid bias=%f", bias_);
      initialized_ = false;
      return IndexError_InvalidArgument;
    }

    initialized_ = true;
    LOG_INFO("SiftInt8StreamingReformer init: bias=%f", bias_);
    return 0;
  }

  bool requires_post_open_reinit() const override { return true; }

  int cleanup(void) override { return 0; }

  int load(IndexStorage::Pointer) override { return 0; }

  int unload(void) override { return 0; }

  //! Transform query: float → uint8 (stored in int8 buffer)
  //! Query is quantized as uint8 with no extra fields.
  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    return do_transform_query(query, qmeta, 1, out, ometa);
  }

  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    return do_transform_query(query, qmeta, count, out, ometa);
  }

  //! Convert record: float → int8 + sq_sum_half extra field
  int convert(const void *record, const IndexQueryMeta &rmeta, std::string *out,
              IndexQueryMeta *ometa) const override {
    return do_convert_record(record, rmeta, 1, out, ometa);
  }

  int convert(const void *records, const IndexQueryMeta &rmeta, uint32_t count,
              std::string *out, IndexQueryMeta *ometa) const override {
    return do_convert_record(records, rmeta, count, out, ometa);
  }

  //! Normalize results: no-op for SIFT distance proxy
  //! The distance is a monotonic ranking proxy (sq_sum_half - ip),
  //! not a true L2 distance, so no post-scaling is needed.
  int normalize(const void * /*query*/, const IndexQueryMeta & /*qmeta*/,
                IndexDocumentList & /*result*/) const override {
    return 0;
  }

  bool need_revert() const override { return true; }

  //! Revert: convert int8 record back to float (approximate)
  int revert(const void *in, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    if (!initialized_) {
      LOG_ERROR(
          "SiftInt8StreamingReformer::revert called before init "
          "with valid params");
      return IndexError_Runtime;
    }
    // The stored dimension includes the extra float tail.
    // Original dimension = stored_dim - sizeof(float).
    size_t stored_dim = qmeta.dimension();
    size_t original_dim =
        stored_dim > sizeof(float) ? stored_dim - sizeof(float) : stored_dim;
    out->resize(original_dim * sizeof(float));
    float *out_buf = reinterpret_cast<float *>(out->data());
    const int8_t *buf = reinterpret_cast<const int8_t *>(in);

    for (size_t i = 0; i < original_dim; ++i) {
      out_buf[i] = static_cast<float>(buf[i]) - bias_;
    }

    return 0;
  }

 private:
  //! Transform query: float → uint8, stored in the output dimension that
  //! matches database vectors (original_dim + sizeof(float)) for alignment,
  //! but the tail bytes are unused by the distance kernel.
  int do_transform_query(const void *src, const IndexQueryMeta &smeta,
                         uint32_t count, std::string *out,
                         IndexQueryMeta *ometa) const {
    if (!initialized_) {
      LOG_ERROR(
          "SiftInt8StreamingReformer: transform called before init "
          "with valid params");
      return IndexError_Runtime;
    }
    if (smeta.data_type() != IndexMeta::DataType::DT_FP32 ||
        smeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = smeta;
    // Output dimension = original_dim + sizeof(float) to match database layout
    size_t original_dim = smeta.dimension();
    size_t output_dim = original_dim + sizeof(float);
    ometa->set_meta(IndexMeta::DataType::DT_INT8, output_dim);
    const size_t out_stride = ometa->element_size();
    out->resize(static_cast<size_t>(count) * out_stride);

    const float *vec = reinterpret_cast<const float *>(src);
    int8_t *ovec = reinterpret_cast<int8_t *>(&(*out)[0]);
    for (uint32_t i = 0; i < count; ++i) {
      quantize_query(vec + i * original_dim, original_dim,
                     ovec + i * out_stride);
    }
    return 0;
  }

  //! Convert record: float → int8 + sq_sum_half tail
  int do_convert_record(const void *src, const IndexQueryMeta &smeta,
                        uint32_t count, std::string *out,
                        IndexQueryMeta *ometa) const {
    if (!initialized_) {
      LOG_ERROR(
          "SiftInt8StreamingReformer: convert called before init "
          "with valid params");
      return IndexError_Runtime;
    }
    if (smeta.data_type() != IndexMeta::DataType::DT_FP32 ||
        smeta.unit_size() !=
            IndexMeta::UnitSizeof(IndexMeta::DataType::DT_FP32)) {
      return IndexError_Unsupported;
    }

    *ometa = smeta;
    size_t original_dim = smeta.dimension();
    size_t output_dim = original_dim + sizeof(float);
    ometa->set_meta(IndexMeta::DataType::DT_INT8, output_dim);
    const size_t out_stride = ometa->element_size();
    out->resize(static_cast<size_t>(count) * out_stride);

    const float *vec = reinterpret_cast<const float *>(src);
    int8_t *ovec = reinterpret_cast<int8_t *>(&(*out)[0]);
    for (uint32_t i = 0; i < count; ++i) {
      quantize_record(vec + i * original_dim, original_dim,
                      ovec + i * out_stride);
    }
    return 0;
  }

  //! Quantize query: float → uint8 (stored in int8 buffer)
  //! Query uses uint8 representation for dpbusd (first operand = unsigned).
  inline void quantize_query(const float *in, size_t dim,
                             int8_t *out) const {
    for (size_t i = 0; i < dim; ++i) {
      float v = std::round(in[i]);
      v = std::max(0.0f, std::min(255.0f, v));
      reinterpret_cast<uint8_t *>(out)[i] = static_cast<uint8_t>(v);
    }
    // Zero the tail (sq_sum_half area) — unused by query
    float *tail = reinterpret_cast<float *>(out + dim);
    tail[0] = 0.0f;
  }

  //! Quantize record: float → int8 (value + bias) + sq_sum_half tail
  inline void quantize_record(const float *in, size_t dim,
                              int8_t *out) const {
    float sq_sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
      float v = in[i];
      sq_sum += v * v;
      float quantized = std::round(v + bias_);
      quantized = std::max(-128.0f, std::min(127.0f, quantized));
      out[i] = static_cast<int8_t>(quantized);
    }
    float *tail = reinterpret_cast<float *>(out + dim);
    tail[0] = sq_sum / 2.0f;
  }

  float bias_{0.0f};
  bool initialized_{false};
};

INDEX_FACTORY_REGISTER_REFORMER_ALIAS(SiftInt8StreamingReformer,
                                      SiftInt8StreamingReformer,
                                      IndexMeta::DataType::DT_INT8);

}  // namespace core
}  // namespace zvec
