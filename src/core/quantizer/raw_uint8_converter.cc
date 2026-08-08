// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include <zvec/core/framework/index_framework.h>
#include <zvec/turbo/turbo.h>

namespace zvec::core {
namespace {

inline void ConvertFp32ToRawUint8(const float *input, size_t dimension,
                                  uint8_t *output) {
  // This is a physical type conversion, not a quantizer: no scale, bias,
  // clipping range, or per-record tail is produced. SIFT values are already
  // integral and representable in uint8_t, so the cast preserves them exactly.
  static const auto convert = turbo::get_raw_uint8_convert_func();
  if (convert) {
    convert(input, dimension, output);
  } else {
    for (size_t i = 0; i < dimension; ++i) {
      output[i] = static_cast<uint8_t>(input[i]);
    }
  }
}

class RawUint8Holder final : public IndexHolder {
 public:
  class Iterator final : public IndexHolder::Iterator {
   public:
    Iterator(size_t dimension, IndexHolder::Iterator::Pointer &&front)
        : buffer_(dimension), front_(std::move(front)) {
      transform_record();
    }

    const void *data() const override { return buffer_.data(); }
    bool is_valid() const override { return front_->is_valid(); }
    uint64_t key() const override { return front_->key(); }

    void next() override {
      front_->next();
      transform_record();
    }

   private:
    void transform_record() {
      if (front_->is_valid()) {
        ConvertFp32ToRawUint8(static_cast<const float *>(front_->data()),
                              buffer_.size(), buffer_.data());
      }
    }

    std::vector<uint8_t> buffer_{};
    IndexHolder::Iterator::Pointer front_{};
  };

  explicit RawUint8Holder(IndexHolder::Pointer front)
      : front_(std::move(front)) {}

  size_t count() const override { return front_->count(); }
  size_t dimension() const override { return front_->dimension(); }
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DataType::DT_UINT8;
  }
  size_t element_size() const override { return dimension(); }
  bool multipass() const override { return front_->multipass(); }

  IndexHolder::Iterator::Pointer create_iterator() override {
    auto iterator = front_->create_iterator();
    if (!iterator) return nullptr;
    return std::make_unique<Iterator>(dimension(), std::move(iterator));
  }

 private:
  IndexHolder::Pointer front_{};
};

}  // namespace

class RawUint8Converter final : public IndexConverter {
 public:
  int init(const IndexMeta &meta, const ailego::Params &) override {
    if (meta.data_type() != IndexMeta::DataType::DT_FP32 ||
        meta.unit_size() != sizeof(float)) {
      LOG_ERROR("RawUint8Converter only supports FP32 input");
      return IndexError_Unsupported;
    }
    meta_ = meta;
    meta_.set_meta(IndexMeta::DataType::DT_UINT8, meta.dimension());
    meta_.set_converter("RawUint8Converter", 0, ailego::Params());
    meta_.set_reformer("RawUint8Reformer", 0, ailego::Params());
    return 0;
  }

  int cleanup() override { return 0; }
  int train(IndexHolder::Pointer) override { return 0; }

  int transform(IndexHolder::Pointer holder) override {
    if (!holder || holder->data_type() != IndexMeta::DataType::DT_FP32 ||
        holder->dimension() != meta_.dimension()) {
      return IndexError_Mismatch;
    }
    holder_ = std::make_shared<RawUint8Holder>(std::move(holder));
    return 0;
  }

  int dump(const IndexDumper::Pointer &) override { return 0; }
  const Stats &stats() const override { return stats_; }
  IndexHolder::Pointer result() const override { return holder_; }
  const IndexMeta &meta() const override { return meta_; }

 private:
  IndexMeta meta_{};
  IndexHolder::Pointer holder_{};
  Stats stats_{};
};

INDEX_FACTORY_REGISTER_CONVERTER(RawUint8Converter);

}  // namespace zvec::core
