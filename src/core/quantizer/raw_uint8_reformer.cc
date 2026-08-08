// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_reformer.h>
#include <zvec/turbo/turbo.h>

namespace zvec::core {
namespace {

inline void ConvertFp32ToRawUint8(const float *input, size_t count,
                                  uint8_t *output) {
  static const auto convert = turbo::get_raw_uint8_convert_func();
  if (convert) {
    convert(input, count, output);
  } else {
    for (size_t i = 0; i < count; ++i) {
      output[i] = static_cast<uint8_t>(input[i]);
    }
  }
}

}  // namespace

class RawUint8Reformer final : public IndexReformer {
 public:
  int init(const ailego::Params &) override { return 0; }
  int cleanup() override { return 0; }
  int load(IndexStorage::Pointer) override { return 0; }
  int unload() override { return 0; }

  int transform(const void *query, const IndexQueryMeta &qmeta,
                std::string *out, IndexQueryMeta *ometa) const override {
    return transform(query, qmeta, 1, out, ometa);
  }

  int transform(const void *query, const IndexQueryMeta &qmeta, uint32_t count,
                std::string *out, IndexQueryMeta *ometa) const override {
    if (!query || !out || !ometa) return IndexError_InvalidArgument;
    if (qmeta.data_type() == IndexMeta::DataType::DT_UINT8) {
      out->assign(static_cast<const char *>(query),
                  qmeta.element_size() * count);
      *ometa = qmeta;
      return 0;
    }
    if (qmeta.data_type() != IndexMeta::DataType::DT_FP32 ||
        qmeta.unit_size() != sizeof(float)) {
      return IndexError_Unsupported;
    }

    const size_t values = static_cast<size_t>(qmeta.dimension()) * count;
    out->resize(values);
    ConvertFp32ToRawUint8(static_cast<const float *>(query), values,
                          reinterpret_cast<uint8_t *>(out->data()));
    *ometa = qmeta;
    ometa->set_meta(IndexMeta::DataType::DT_UINT8, qmeta.dimension());
    return 0;
  }

  int normalize(const void *, const IndexQueryMeta &,
                IndexDocumentList &) const override {
    return 0;
  }

  bool need_revert() const override { return true; }

  int revert(const void *input, const IndexQueryMeta &qmeta,
             std::string *out) const override {
    if (!input || !out ||
        qmeta.data_type() != IndexMeta::DataType::DT_UINT8) {
      return IndexError_Unsupported;
    }
    out->resize(static_cast<size_t>(qmeta.dimension()) * sizeof(float));
    const auto *source = static_cast<const uint8_t *>(input);
    auto *target = reinterpret_cast<float *>(out->data());
    for (size_t i = 0; i < qmeta.dimension(); ++i) {
      target[i] = static_cast<float>(source[i]);
    }
    return 0;
  }
};

INDEX_FACTORY_REGISTER_REFORMER(RawUint8Reformer);

}  // namespace zvec::core
