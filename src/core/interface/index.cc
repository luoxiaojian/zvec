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

#include <magic_enum/magic_enum.hpp>
#include <zvec/core/framework/index_converter.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_storage.h>
#include <zvec/core/interface/index.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>
#include "algorithm/flat/flat_streamer.h"
#include "mixed_reducer/mixed_reducer_params.h"

namespace zvec::core_interface {

namespace {

int PrepareNativeFlatQuery(const VectorData &source,
                           const core::IndexQueryMeta &source_meta,
                           const core::IndexQueryMeta &flat_meta,
                           std::string *storage, VectorData *prepared) {
  if (!storage || !prepared ||
      !std::holds_alternative<DenseVector>(source.vector)) {
    return core::IndexError_InvalidArgument;
  }
  const auto &dense = std::get<DenseVector>(source.vector);
  if (source_meta.data_type() == flat_meta.data_type() &&
      source_meta.dimension() == flat_meta.dimension()) {
    *prepared = source;
    return core::IndexError_Success;
  }
  if (source_meta.data_type() != core::IndexMeta::DataType::DT_FP32 ||
      source_meta.dimension() != flat_meta.dimension()) {
    return core::IndexError_Unsupported;
  }

  const auto *input = static_cast<const float *>(dense.data);
  if (flat_meta.data_type() == core::IndexMeta::DataType::DT_FP16) {
    storage->resize(flat_meta.dimension() * sizeof(ailego::Float16));
    static const auto convert = turbo::get_fp32_to_fp16_convert_func();
    auto *output = reinterpret_cast<uint16_t *>(storage->data());
    if (convert) {
      convert(input, flat_meta.dimension(), output);
    } else {
      ailego::FloatHelper::ToFP16(input, flat_meta.dimension(), output);
    }
  } else if (flat_meta.data_type() == core::IndexMeta::DataType::DT_UINT8) {
    storage->resize(flat_meta.dimension());
    auto *output = reinterpret_cast<uint8_t *>(storage->data());
    const auto convert = turbo::get_raw_uint8_convert_func();
    if (convert) {
      convert(input, flat_meta.dimension(), output);
    } else {
      for (size_t i = 0; i < flat_meta.dimension(); ++i) {
        output[i] = static_cast<uint8_t>(input[i]);
      }
    }
  } else {
    return core::IndexError_Unsupported;
  }
  prepared->vector = DenseVector{storage->data()};
  return core::IndexError_Success;
}

int RestoreNativeFlatVectors(const core::IndexQueryMeta &public_meta,
                             const core::IndexQueryMeta &flat_meta,
                             SearchResult *result) {
  if (!result || public_meta.dimension() != flat_meta.dimension()) {
    return core::IndexError_InvalidArgument;
  }
  if (public_meta.data_type() == flat_meta.data_type()) {
    return core::IndexError_Success;
  }
  if (public_meta.data_type() != core::IndexMeta::DataType::DT_FP32) {
    return core::IndexError_Unsupported;
  }

  result->reverted_vector_list_.resize(result->doc_list_.size());
  for (size_t i = 0; i < result->doc_list_.size(); ++i) {
    const void *source = result->doc_list_[i].vector();
    if (!source) {
      return core::IndexError_Runtime;
    }
    auto &restored = result->reverted_vector_list_[i];
    restored.resize(public_meta.dimension() * sizeof(float));
    auto *output = reinterpret_cast<float *>(restored.data());
    if (flat_meta.data_type() == core::IndexMeta::DataType::DT_FP16) {
      ailego::FloatHelper::ToFP32(static_cast<const uint16_t *>(source),
                                 public_meta.dimension(), output);
    } else if (flat_meta.data_type() ==
               core::IndexMeta::DataType::DT_UINT8) {
      const auto *input = static_cast<const uint8_t *>(source);
      for (size_t d = 0; d < public_meta.dimension(); ++d) {
        output[d] = static_cast<float>(input[d]);
      }
    } else {
      result->reverted_vector_list_.clear();
      return core::IndexError_Unsupported;
    }
  }
  return core::IndexError_Success;
}

// A multipass view over merge-source providers. Unlike
// MultiPassIndexHolder this holder does not materialize a second in-memory
// copy of every source vector; each converter pass creates fresh provider
// iterators and walks sources in their original order.
class MergeSourceIndexHolder final : public core::IndexHolder {
 public:
  class Iterator final : public core::IndexHolder::Iterator {
   public:
    explicit Iterator(const std::vector<core::IndexHolder::Pointer> &sources)
        : sources_(sources) {
      advance_to_valid_source();
    }

    const void *data() const override {
      return source_iter_->data();
    }

    bool is_valid() const override {
      return source_iter_ != nullptr && source_iter_->is_valid();
    }

    uint64_t key() const override {
      return source_iter_->key();
    }

    void next() override {
      if (source_iter_ != nullptr) {
        source_iter_->next();
      }
      advance_to_valid_source();
    }

   private:
    void advance_to_valid_source() {
      while (source_iter_ == nullptr || !source_iter_->is_valid()) {
        source_iter_.reset();
        if (source_index_ >= sources_.size()) {
          return;
        }
        source_iter_ = sources_[source_index_++]->create_iterator();
      }
    }

    const std::vector<core::IndexHolder::Pointer> &sources_;
    size_t source_index_{0};
    core::IndexHolder::Iterator::Pointer source_iter_{};
  };

  MergeSourceIndexHolder(std::vector<core::IndexHolder::Pointer> sources,
                         core::IndexQueryMeta source_meta)
      : sources_(std::move(sources)),
        source_meta_(std::move(source_meta)) {
    for (const auto &source : sources_) {
      count_ += source->count();
    }
  }

  size_t count() const override {
    return count_;
  }

  size_t dimension() const override {
    return source_meta_.dimension();
  }

  core::IndexMeta::DataType data_type() const override {
    return source_meta_.data_type();
  }

  size_t element_size() const override {
    return source_meta_.element_size();
  }

  bool multipass() const override {
    return true;
  }

  core::IndexHolder::Iterator::Pointer create_iterator() override {
    return std::make_unique<Iterator>(sources_);
  }

 private:
  std::vector<core::IndexHolder::Pointer> sources_{};
  size_t count_{0};
  core::IndexQueryMeta source_meta_{};
};

int TrainConverterFromMergeSources(
    const core::IndexConverter::Pointer &converter,
    const std::vector<Index::Pointer> &indexes) {
  if (!converter || indexes.empty()) {
    return core::IndexError_InvalidArgument;
  }

  std::vector<core::IndexHolder::Pointer> sources;
  sources.reserve(indexes.size());
  core::IndexQueryMeta source_meta;
  bool has_source_meta = false;
  for (const auto &index : indexes) {
    auto provider = index->create_index_provider();
    if (!provider) {
      continue;
    }
    if (!has_source_meta) {
      source_meta.set_meta(provider->data_type(), provider->dimension());
      has_source_meta = true;
    } else if (provider->data_type() != source_meta.data_type() ||
               provider->dimension() != source_meta.dimension() ||
               provider->element_size() != source_meta.element_size()) {
      LOG_ERROR("Merge-source vector type mismatch");
      return core::IndexError_Mismatch;
    }
    sources.emplace_back(std::move(provider));
  }
  auto holder = std::make_shared<MergeSourceIndexHolder>(
      std::move(sources), source_meta);
  if (holder->count() == 0) {
    LOG_ERROR("No vectors available to train converter");
    return core::IndexError_InvalidArgument;
  }
  return core::IndexConverter::TrainAndTransform(converter, holder);
}

int CreateReformerFromConverterMeta(
    const core::IndexConverter::Pointer &converter,
    core::IndexMeta *proxima_index_meta,
    core::IndexQueryMeta *streamer_vector_meta,
    core::IndexReformer::Pointer *reformer) {
  if (!converter || !proxima_index_meta || !streamer_vector_meta || !reformer) {
    return core::IndexError_InvalidArgument;
  }
  if (*reformer != nullptr) {
    return core::IndexError_Success;
  }
  const auto &meta = converter->meta();
  if (meta.reformer_name().empty()) {
    return core::IndexError_Success;
  }
  *proxima_index_meta = meta;
  streamer_vector_meta->set_meta(proxima_index_meta->data_type(),
                                 proxima_index_meta->dimension());
  streamer_vector_meta->set_meta_type(proxima_index_meta->meta_type());
  *reformer = core::IndexFactory::CreateReformer(meta.reformer_name());
  if (!*reformer || (*reformer)->init(meta.reformer_params()) != 0) {
    LOG_ERROR("Failed to create reformer '%s' from converter meta",
              meta.reformer_name().c_str());
    return core::IndexError_Runtime;
  }
  return core::IndexError_Success;
}

void PersistTrainedMetaToStreamer(const core::IndexMeta &meta,
                                  core::IndexStreamer::Pointer &streamer) {
  if (!streamer) {
    return;
  }
  // Streamers that persist trained meta via flush (e.g. HnswStreamer) override
  // merge_trained_meta; others keep the base no-op and persist by other means.
  streamer->merge_trained_meta(meta);
}

}  // namespace

// eliminate the pre-alloc of the context pool
thread_local static std::array<core::IndexContext::Pointer,
                               (magic_enum::enum_count<IndexType>() - 1) * 2>
    _context_list;


bool Index::init_context() {
  context_index_ = (magic_enum::enum_integer(param_.index_type) - 1) * 2 +
                   static_cast<size_t>(is_sparse_);
  if (_context_list[context_index_] == nullptr) {
    if ((_context_list[context_index_] = streamer_->create_context()) ==
        nullptr) {
      LOG_ERROR("Failed to create context");
      return false;
    }
  }
  return true;
}

core::IndexContext::Pointer &Index::acquire_context() {
  init_context();
  return _context_list[context_index_];
}

int Index::ParseMetricName(const BaseIndexParam &param) {
  std::string metric_name;
  if (is_sparse_) {
    // only inner product is supported for sparse index
    switch (param.metric_type) {
      case MetricType::kInnerProduct:
        metric_name = "InnerProductSparse";
        break;
      case MetricType::kMIPSL2sq:
        metric_name = "MipsSquaredEuclideanSparse";
        break;
      default:
        LOG_ERROR("Unsupported metric type");
        return core::IndexError_Runtime;
    }
  } else {
    switch (param.metric_type) {
      case MetricType::kL2sq:
        metric_name = "SquaredEuclidean";
        break;
      case MetricType::kInnerProduct:
        metric_name = "InnerProduct";
        break;
      case MetricType::kCosine:
        metric_name = "Cosine";  // This is already the normalizedCosine
        break;
      case MetricType::kMIPSL2sq:
        metric_name = "MipsSquaredEuclidean";
        break;
      default:
        LOG_ERROR("Unsupported metric type");
        return core::IndexError_Runtime;
    }
  }
  // TODO: MIPS need to set some param
  // for streamer open()
  proxima_index_meta_.set_metric(metric_name, 0, ailego::Params());
  return 0;
}

int Index::CreateAndInitMetric(const BaseIndexParam & /*param*/) {
  auto &metric_name = proxima_index_meta_.metric_name();

  metric_ = core::IndexFactory::CreateMetric(metric_name);
  if (!metric_) {
    LOG_ERROR("Failed to create metric, name %s", metric_name.c_str());
    return core::IndexError_Runtime;
  }
  if (const auto ret = metric_->init(proxima_index_meta_,
                                     proxima_index_meta_.metric_params());
      ret != 0) {
    LOG_ERROR("Failed to create and init metric, name %s, code %d, desc: %s",
              metric_name.c_str(), ret, core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  if (metric_->query_metric()) {
    metric_ = metric_->query_metric();
  }

  return core::IndexError_Success;
}

int Index::CreateAndInitConverterReformer(const QuantizerParam &param,
                                          const BaseIndexParam &index_param) {
  ailego::Params converter_params;
  std::string converter_name;
  if (is_sparse_) {
    switch (param.type) {
      case QuantizerType::kNone:
        return core::IndexError_Success;
      case QuantizerType::kFP16:
        converter_name = "HalfFloatSparseConverter";
        break;
      default:
        LOG_ERROR("Unsupported quantizer type: ");
        return core::IndexError_Unsupported;
    }
  } else {
    if (index_param.metric_type == MetricType::kCosine) {
      switch (param.type) {
        case QuantizerType::kNone:
          if (index_param.data_type == DataType::DT_FP16) {
            converter_name = "CosineHalfFloatConverter";
          } else if (index_param.data_type == DataType::DT_FP32) {
            converter_name = "CosineNormalizeConverter";
          } else {
            LOG_ERROR("Unsupported data type: ");
            return core::IndexError_Unsupported;
          }
          break;
        case QuantizerType::kRabitq:
          if (index_param.data_type == DataType::DT_FP32) {
            converter_name = "CosineNormalizeConverter";
          } else {
            LOG_ERROR("Unsupported data type: ");
            return core::IndexError_Unsupported;
          }
          break;
        case QuantizerType::kFP16:
          converter_name = "CosineFp16Converter";
          break;
        case QuantizerType::kInt8:
          converter_name = "CosineInt8Converter";
          break;
        case QuantizerType::kInt4:
          converter_name = "CosineInt4Converter";
          break;
        default:
          LOG_ERROR("Unsupported quantizer type: ");
          return core::IndexError_Unsupported;
      }
    } else {
      switch (param.type) {
        case QuantizerType::kNone:
          return core::IndexError_Success;
        case QuantizerType::kFP16:
          converter_name = "HalfFloatConverter";
          break;
        case QuantizerType::kInt8:
          converter_name = "Int8StreamingConverter";
          break;
        case QuantizerType::kInt4:
          converter_name = "Int4StreamingConverter";
          break;
        case QuantizerType::kRabitq:
          // no converter here
          return 0;
        case QuantizerType::kUniformInt8:
          converter_name = "UniformInt8StreamingConverter";
          break;
        case QuantizerType::kUniformUint8:
          converter_name = "UniformUint8StreamingConverter";
          break;
        case QuantizerType::kUniformUint4:
          converter_name = "UniformUint4StreamingConverter";
          break;
        default:
          LOG_ERROR("Unsupported quantizer type: ");
          return core::IndexError_Unsupported;
      }
    }
  }

  proxima_index_meta_.set_converter(converter_name, 0, converter_params);
  converter_ = core::IndexFactory::CreateConverter(converter_name);
  if (converter_ == nullptr ||
      converter_->init(proxima_index_meta_, converter_params) != 0) {
    LOG_ERROR("Failed to create and init converter");
    return core::IndexError_Runtime;
  }

  proxima_index_meta_ = converter_->meta();

  if (!proxima_index_meta_.reformer_name().empty()) {
    reformer_ =
        core::IndexFactory::CreateReformer(proxima_index_meta_.reformer_name());
    if (reformer_ == nullptr ||
        reformer_->init(proxima_index_meta_.reformer_params()) != 0) {
      LOG_ERROR("Failed to create and init reformer");
      return core::IndexError_Runtime;
    }
  }

  streamer_vector_meta_.set_meta(proxima_index_meta_.data_type(),
                                 proxima_index_meta_.dimension());
  streamer_vector_meta_.set_meta_type(proxima_index_meta_.meta_type());

  return core::IndexError_Success;
}

int Index::Init(const BaseIndexParam &param) {
  param_ = param;  // will lose the original type info

  is_sparse_ = param.is_sparse;
  is_huge_page_ = param.is_huge_page;

  proxima_index_meta_.set_meta(param.data_type, param.dimension);
  proxima_index_meta_.set_meta_type(is_sparse_ ? IndexMeta::MetaType::MT_SPARSE
                                               : IndexMeta::MetaType::MT_DENSE);

  input_vector_meta_.set_meta(proxima_index_meta_.data_type(),
                              proxima_index_meta_.dimension());
  input_vector_meta_.set_meta_type(proxima_index_meta_.meta_type());
  streamer_vector_meta_ = input_vector_meta_;


  // when quantizer=int8/int4, the converter.init() will change the metric to
  // QuantizedInteger with params
  if (ParseMetricName(param) != 0) {
    LOG_ERROR("Failed to parse metric name");
    return core::IndexError_Runtime;
  }

  if (CreateAndInitConverterReformer(param.quantizer_param, param) != 0) {
    LOG_ERROR("Failed to create and init converter");
    return core::IndexError_Runtime;
  }

  // must after quantizer handled. e.g., cosine doesn't support int8 quantizer
  if (CreateAndInitMetric(param) != 0) {
    LOG_ERROR("Failed to create and init metric");
    return core::IndexError_Runtime;
  }

  if (CreateAndInitStreamer(param) != 0) {
    LOG_ERROR("Failed to create and init streamer");
    return core::IndexError_Runtime;
  }
  return 0;
}


int Index::Open(const std::string &file_path, StorageOptions storage_options) {
  ailego::Params storage_params;
  // storage_params.set("proxima.mmap_file.storage.memory_warmup", true);
  // storage_params.set("proxima.mmap_file.storage.segment_meta_capacity",
  // 1024);
  switch (storage_options.type) {
    case StorageOptions::StorageType::kMMAP: {
      storage_ = core::IndexFactory::CreateStorage("MMapFileStorage");
      if (storage_ == nullptr) {
        LOG_ERROR("Failed to create MMapFileStorage");
        return core::IndexError_Runtime;
      }
      int ret = storage_->init(storage_params);
      if (ret != 0) {
        LOG_ERROR("Failed to init MMapFileStorage, path: %s, err: %s",
                  file_path.c_str(), core::IndexError::What(ret));
        return ret;
      }
      break;
    }
    case StorageOptions::StorageType::kBufferPool: {
      storage_ = core::IndexFactory::CreateStorage("BufferStorage");
      if (storage_ == nullptr) {
        LOG_ERROR("Failed to create BufferStorage");
        return core::IndexError_Runtime;
      }
      int ret = storage_->init(storage_params);
      if (ret != 0) {
        LOG_ERROR("Failed to init BufferStorage, path: %s, err: %s",
                  file_path.c_str(), core::IndexError::What(ret));
        return ret;
      }
      break;
    }
    default:
      LOG_ERROR("Unsupported storage type");
      return core::IndexError_Unsupported;
  }

  // read_options.create_new
  int ret = storage_->open(file_path, storage_options.create_new);
  if (ret != 0) {
    LOG_ERROR("Failed to open storage, path: %s, err: %s", file_path.c_str(),
              core::IndexError::What(ret));
    return core::IndexError_Runtime;
  }
  if (streamer_ == nullptr || streamer_->open(storage_) != 0) {
    LOG_ERROR("Failed to open streamer, path: %s", file_path.c_str());
    return core::IndexError_Runtime;
  }

  // Restore reformer from persisted meta after reload. Training and in-process
  // reformer creation happen in Merge(); Search must not lazily init reformer.
  // Pre-optimize writer segments (UniformInt8) legitimately have no reformer.
  if (reformer_ == nullptr && streamer_ != nullptr) {
    const auto &streamer_meta = streamer_->meta();
    if (!streamer_meta.reformer_name().empty()) {
      reformer_ =
          core::IndexFactory::CreateReformer(streamer_meta.reformer_name());
      if (!reformer_ || reformer_->init(streamer_meta.reformer_params()) != 0) {
        LOG_ERROR("Failed to create reformer '%s' from persisted meta",
                  streamer_meta.reformer_name().c_str());
        return core::IndexError_Runtime;
      }
    } else if (converter_ != nullptr) {
      if (!converter_->meta().reformer_name().empty()) {
        if (int ret = CreateReformerFromConverterMeta(
                converter_, &proxima_index_meta_, &streamer_vector_meta_,
                &reformer_);
            ret != 0) {
          return ret;
        }
      } else if (!storage_options.create_new) {
        LOG_ERROR(
            "Index::Open: converter exists but reformer not initialized and "
            "no reformer in persisted meta");
        return core::IndexError_Runtime;
      }
      // New index: converters like UniformInt8 defer reformer creation until
      // train() runs (e.g. during optimize merge). Raw vectors may be ingested
      // before that point without a reformer.
    }
  }

  // converter/reformer/metric are created in IndexFactory::CreateIndex
  // TODO: init

  // TODO: context pool
  if (!init_context()) {  // to validate if any error, will be overwritten
    LOG_ERROR("Failed to init context");
    return core::IndexError_Runtime;
  }

  is_open_ = true;
  is_read_only_ = storage_options.read_only;
  return 0;
}

int Index::Close() {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (!is_read_only_) {
    if (ailego_unlikely(Flush() != 0)) {
      LOG_ERROR("Failed to cleanup streamer");
      return core::IndexError_Runtime;
    }
  }
  if (ailego_unlikely(streamer_->cleanup() != 0)) {
    LOG_ERROR("Failed to cleanup streamer");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(storage_->close() != 0)) {
    LOG_ERROR("Failed to close storage");
    return core::IndexError_Runtime;
  }
  is_open_ = false;
  return 0;
}

int Index::Flush() {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (is_read_only_) {
    LOG_ERROR("Cannot flush read-only index");
    return core::IndexError_Runtime;
  }
  // Only persist trained meta once the converter has actually produced a
  // reformer (i.e. training ran, e.g. during optimize). Before that the call
  // would be a no-op, so skip it on every pre-training flush.
  if (converter_ != nullptr && !converter_->meta().reformer_name().empty()) {
    PersistTrainedMetaToStreamer(converter_->meta(), streamer_);
  }
  if (ailego_unlikely(streamer_->flush(0) != 0)) {
    LOG_ERROR("Failed to flush streamer");
    return core::IndexError_Runtime;
  }
  if (ailego_unlikely(storage_->flush() != 0)) {
    LOG_ERROR("Failed to flush storage");
    return core::IndexError_Runtime;
  }
  return 0;
}

bool Index::IsDirty() const {
  if (!storage_) {
    return false;
  }
  return storage_->is_dirty();
}

int Index::Fetch(const uint32_t doc_id, VectorDataBuffer *vector_data_buffer) {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }
  if (is_sparse_) {
    return _sparse_fetch(doc_id, vector_data_buffer);
  }
  return _dense_fetch(doc_id, vector_data_buffer);
}

int Index::Add(const VectorData &vector_data, const uint32_t doc_id) {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (is_read_only_) {
    LOG_ERROR("Cannot add to read-only index");
    return core::IndexError_Runtime;
  }

  auto &context = acquire_context();
  if (!context) {
    LOG_ERROR("Failed to acquire context");
    return core::IndexError_Runtime;
  }

  int ret = 0;
  if (is_sparse_) {
    ret = _sparse_add(vector_data, doc_id, context);
  } else {
    ret = _dense_add(vector_data, doc_id, context);
  }
  context->reset();
  return ret;
}


int Index::Search(const VectorData &vector_data,
                  const BaseIndexQueryParam::Pointer &search_param,
                  SearchResult *result) {
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }

  if (!is_trained_ && this->Train() != 0) {
    LOG_ERROR("Failed to train index");
    return core::IndexError_Runtime;
  }

  auto &context = acquire_context();
  if (!context) {
    LOG_ERROR("Failed to acquire context");
    return core::IndexError_Runtime;
  }

  if (_prepare_for_search(vector_data, search_param, context) != 0) {
    LOG_ERROR("Failed to prepare for search");
    context->reset();
    return core::IndexError_Runtime;
  }

  if (is_sparse_) {
    int ret = _sparse_search(vector_data, search_param, result, context);
    context->reset();
    return ret;
  }

  // dense support refiner, but sparse doesn't
  int ret = 0;
  if (search_param->refiner_param == nullptr) {
    ret = _dense_search(vector_data, search_param, result, context);
    context->reset();
  } else {
    auto &reference_index = search_param->refiner_param->reference_index;
    if (reference_index == nullptr) {
      LOG_ERROR("Reference index is not set");
      context->reset();
      return core::IndexError_Runtime;
    }
    // TODO: tackle query_param's type info loss to loosen the constraint
    if (reference_index->param_.index_type != IndexType::kFlat) {
      LOG_ERROR("Reference index is not flat");
      context->reset();
      return core::IndexError_Runtime;
    }

    context->set_topk(_get_coarse_search_topk(search_param));
    context->set_fetch_vector(false);  // no need to fetch vector
    if (_dense_search(vector_data, search_param, result, context) != 0) {
      LOG_ERROR("Failed to search");
      context->reset();
      return core::IndexError_Runtime;
    }

    auto &base_result = context->result();
    std::vector<uint64_t> keys(base_result.size());
    for (size_t i = 0; i < base_result.size(); ++i) {
      keys[i] = base_result[i].key();
    }

    FlatQueryParam::Pointer flat_search_param =
        std::make_shared<FlatQueryParam>();
    flat_search_param->topk = search_param->topk;
    flat_search_param->fetch_vector = search_param->fetch_vector;
    flat_search_param->filter = search_param->filter;
    // TODO: should copy other params?
    flat_search_param->bf_pks = std::make_shared<std::vector<uint64_t>>(keys);

    std::string flat_query_storage;
    VectorData flat_query;
    ret = PrepareNativeFlatQuery(
        vector_data, input_vector_meta_, reference_index->input_vector_meta_,
        &flat_query_storage, &flat_query);
    if (ret == 0) {
      ret = reference_index->Search(flat_query, flat_search_param, result);
    }
    if (ret == 0 && flat_search_param->fetch_vector) {
      ret = RestoreNativeFlatVectors(input_vector_meta_,
                                     reference_index->input_vector_meta_,
                                     result);
    }
  }
  context->reset();
  return ret;
}


int Index::SearchDocIds(const VectorData &vector_data,
                        const BaseIndexQueryParam::Pointer &search_param,
                        int64_t *output_ids, int topk) {
  if (output_ids == nullptr || topk <= 0) {
    return core::IndexError_InvalidArgument;
  }
  if (!is_open_) {
    LOG_ERROR("Index is not open");
    return core::IndexError_Runtime;
  }
  if (!is_trained_ && this->Train() != 0) {
    LOG_ERROR("Failed to train index");
    return core::IndexError_Runtime;
  }
  if (is_sparse_) {
    LOG_ERROR("SearchDocIds does not support sparse indexes");
    return core::IndexError_Unsupported;
  }
  auto &context = acquire_context();
  if (!context) {
    LOG_ERROR("Failed to acquire context");
    return core::IndexError_Runtime;
  }

  if (_prepare_for_search(vector_data, search_param, context) != 0) {
    LOG_ERROR("Failed to prepare for search");
    context->reset();
    return core::IndexError_Runtime;
  }
  int ret = 0;
  // Vamana's contiguous fast path already owns a distance-sorted BlockHeap.
  // Reuse it for both coarse-only and refine searches so SearchDocIds does not
  // rebuild a TopkHeap and materialize IndexDocuments merely to recover keys.
  thread_local std::vector<uint64_t> direct_search_keys;
  auto search_direct_keys = [&]() -> int {
    if (!std::holds_alternative<DenseVector>(vector_data.vector)) {
      return core::IndexError_NotImplemented;
    }
    const DenseVector &dense = std::get<DenseVector>(vector_data.vector);
    const void *coarse_query = dense.data;
    std::string transformed_query;
    core::IndexQueryMeta transformed_meta = input_vector_meta_;
    if (reformer_ != nullptr) {
      if (reformer_->transform(dense.data, input_vector_meta_,
                               &transformed_query,
                               &transformed_meta) != 0) {
        LOG_ERROR("Failed to transform direct Vamana query");
        return core::IndexError_Runtime;
      }
      coarse_query = transformed_query.data();
    }
    return streamer_->search_keys_direct(coarse_query, transformed_meta,
                                         &direct_search_keys, context);
  };

  if (search_param->refiner_param == nullptr) {
    const int direct_ret = search_direct_keys();
    if (direct_ret == 0) {
      const int count = std::min(
          topk, static_cast<int>(direct_search_keys.size()));
      for (int i = 0; i < count; ++i) {
        output_ids[i] = static_cast<int64_t>(direct_search_keys[i]);
      }
      context->reset();
      return 0;
    }
    if (direct_ret != core::IndexError_NotImplemented) {
      LOG_ERROR("Failed direct Vamana doc-id search, error=%d", direct_ret);
      context->reset();
      return direct_ret;
    }
    ret = _dense_search_doc_ids(vector_data, search_param, output_ids, topk,
                                context);
    context->reset();
    return ret;
  }

  // Refine path: retain the same coarse candidate semantics as Search(), but
  // send the candidate keys directly to Flat::SearchDocIds so neither stage
  // materializes vectors/scores in the final result.
  auto &reference_index = search_param->refiner_param->reference_index;
  if (reference_index == nullptr) {
    LOG_ERROR("Reference index is not set");
    context->reset();
    return core::IndexError_Runtime;
  }
  if (reference_index->param_.index_type != IndexType::kFlat) {
    LOG_ERROR("Reference index is not flat");
    context->reset();
    return core::IndexError_Runtime;
  }

  context->set_topk(_get_coarse_search_topk(search_param));
  context->set_fetch_vector(false);
  // Reuse candidate storage on the query thread. The ann-bench path is
  // single-query/single-thread per Index instance, and retaining capacity
  // avoids one allocation at every refine call.
  bool used_direct_coarse = false;

  // Vamana's fast graph search already leaves a distance-sorted BlockHeap.
  // Feed those ids straight into Flat refine instead of rebuilding a
  // TopkHeap, sorting it, materializing IndexDocuments, then copying keys.
  const int coarse_ret = search_direct_keys();
  if (coarse_ret == 0) {
    used_direct_coarse = true;
  } else if (coarse_ret != core::IndexError_NotImplemented) {
    LOG_ERROR("Failed direct Vamana candidate search, error=%d", coarse_ret);
    context->reset();
    return coarse_ret;
  }

  if (!used_direct_coarse) {
    if (_execute_dense_search(vector_data, search_param, context) != 0) {
      LOG_ERROR("Failed to search");
      context->reset();
      return core::IndexError_Runtime;
    }
    const auto &base_result = context->result();
    direct_search_keys.resize(base_result.size());
    for (size_t i = 0; i < base_result.size(); ++i) {
      direct_search_keys[i] = base_result[i].key();
    }
  }

  // Keep the native Flat query buffer on the query thread so FP16/UINT8
  // conversion does not allocate for every request.
  thread_local std::string flat_query_storage;
  VectorData flat_query;
  ret = PrepareNativeFlatQuery(
      vector_data, input_vector_meta_, reference_index->input_vector_meta_,
      &flat_query_storage, &flat_query);
  if (ret != 0) {
    context->reset();
    return ret;
  }

  // Use Flat's generic candidate-limited doc-id operation when its rows are
  // contiguous.  This avoids copying candidate ids into query parameters and
  // avoids a recursive Index::SearchDocIds/result-heap/materialization pass.
  if (std::holds_alternative<DenseVector>(flat_query.vector)) {
    auto &reference_context = reference_index->acquire_context();
    if (reference_context) {
      const auto *flat_streamer =
          dynamic_cast<const core::FlatStreamer<32> *>(
              reference_index->streamer_.get());
      const auto &dense = std::get<DenseVector>(flat_query.vector);
      const int direct_ret =
          flat_streamer
              ? flat_streamer->search_doc_ids_by_p_keys(
                    dense.data, direct_search_keys.data(),
                    direct_search_keys.size(), output_ids,
                    static_cast<size_t>(topk),
                    reference_index->input_vector_meta_, reference_context)
              : core::IndexError_NotImplemented;
      reference_context->reset();
      if (direct_ret == 0) {
        context->reset();
        return 0;
      }
      if (direct_ret != core::IndexError_NotImplemented) {
        LOG_ERROR("Failed direct Flat candidate search, error=%d", direct_ret);
        context->reset();
        return direct_ret;
      }
    }
  }

  auto keys =
      std::make_shared<std::vector<uint64_t>>(direct_search_keys.begin(),
                                              direct_search_keys.end());
  auto flat_search_param = std::make_shared<FlatQueryParam>();
  flat_search_param->topk = static_cast<uint32_t>(topk);
  flat_search_param->fetch_vector = false;
  flat_search_param->filter = search_param->filter;
  flat_search_param->bf_pks = std::move(keys);
  ret = reference_index->SearchDocIds(flat_query, flat_search_param, output_ids,
                                      topk);
  context->reset();
  return ret;
}


int Index::_dense_fetch(const uint32_t doc_id,
                        VectorDataBuffer *vector_data_buffer) {
  core::IndexStorage::MemoryBlock vector_block;
  int ret = streamer_->get_vector_by_id(doc_id, vector_block);
  if (ret != 0) {
    LOG_ERROR("Failed to fetch vector, doc_id: %u", doc_id);
    return core::IndexError_Runtime;
  }
  const void *vector = vector_block.data();

  DenseVectorBuffer dense_vector_buffer;
  std::string &out_vector_buffer = dense_vector_buffer.data;
  // for int4, unit_size * dim != element_size
  out_vector_buffer.resize(input_vector_meta_.element_size());

  if (reformer_ != nullptr) {
    if (reformer_->revert(vector, streamer_vector_meta_, &out_vector_buffer) !=
        0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
  } else {
    out_vector_buffer = std::string(
        static_cast<const char *>(vector),
        input_vector_meta_.dimension() * input_vector_meta_.unit_size());
  }
  vector_data_buffer->vector_buffer = std::move(dense_vector_buffer);
  return 0;
}


int Index::_sparse_fetch(const uint32_t doc_id,
                         VectorDataBuffer *vector_data_buffer) {
  SparseVectorBuffer sparse_vector_buffer;

  if (0 != streamer_->get_sparse_vector_by_id(
               doc_id, &sparse_vector_buffer.count,
               &sparse_vector_buffer.indices, &sparse_vector_buffer.values)) {
    LOG_ERROR("Failed to fetch vector");
    return core::IndexError_Runtime;
  }

  if (reformer_ != nullptr) {
    std::string reverted_sparse_values_buffer;
    if (reformer_->revert(
            sparse_vector_buffer.count, sparse_vector_buffer.get_indices(),
            sparse_vector_buffer.get_values(), streamer_vector_meta_,
            &reverted_sparse_values_buffer) != 0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
    sparse_vector_buffer.values = std::move(reverted_sparse_values_buffer);
  }
  vector_data_buffer->vector_buffer = std::move(sparse_vector_buffer);
  return 0;
}

int Index::_dense_add(const VectorData &vector_data, const uint32_t doc_id,
                      core::IndexContext::Pointer &context) {
  if (!std::holds_alternative<DenseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector_data.vector);
  if (reformer_ != nullptr) {
    core::IndexQueryMeta new_meta;
    std::string new_vector;
    int ret;
    ret = reformer_->convert(dense_vector.data, input_vector_meta_, &new_vector,
                             &new_meta);
    if (ret != 0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
    ret = streamer_->add_with_id_impl(doc_id, new_vector.data(), new_meta,
                                      context);
    if (ret != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  } else {
    int ret = streamer_->add_with_id_impl(doc_id, dense_vector.data,
                                          input_vector_meta_, context);
    if (ret != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  }
  return 0;
}


int Index::_sparse_add(const VectorData &vector_data, const uint32_t doc_id,
                       core::IndexContext::Pointer &context) {
  if (!std::holds_alternative<SparseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const SparseVector &sparse_vector =
      std::get<SparseVector>(vector_data.vector);

  if (reformer_ != nullptr) {
    std::string converted_sparse_values_buffer;
    core::IndexQueryMeta new_meta;
    int ret;
    ret = reformer_->convert(sparse_vector.count, sparse_vector.get_indices(),
                             sparse_vector.get_values(), input_vector_meta_,
                             &converted_sparse_values_buffer, &new_meta);
    if (ret != 0) {
      LOG_ERROR("Failed to convert vector");
      return core::IndexError_Runtime;
    }
    ret = streamer_->add_with_id_impl(
        doc_id, sparse_vector.count, sparse_vector.get_indices(),
        converted_sparse_values_buffer.data(), new_meta, context);
    if (ret != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  } else {
    int ret = streamer_->add_with_id_impl(
        doc_id, sparse_vector.count, sparse_vector.get_indices(),
        sparse_vector.get_values(), input_vector_meta_, context);
    if (ret != 0) {
      LOG_ERROR("Failed to add vector");
      return core::IndexError_Runtime;
    }
  }
  return 0;
}


int Index::_execute_dense_search(
    const VectorData &vector_data,
    const BaseIndexQueryParam::Pointer &search_param,
    core::IndexContext::Pointer &context) {
  if (!std::holds_alternative<DenseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector_data.vector);
  auto vector = dense_vector.data;
  std::string new_vector;
  core::IndexQueryMeta new_meta = input_vector_meta_;
  if (reformer_ != nullptr) {
    if (reformer_->transform(dense_vector.data, input_vector_meta_, &new_vector,
                             &new_meta) != 0) {
      LOG_ERROR("Failed to transform vector");
      return core::IndexError_Runtime;
    }
    vector = new_vector.data();
  }
  if (search_param->bf_pks != nullptr) {
    if (streamer_->search_bf_by_p_keys_impl(
            vector, std::vector<std::vector<uint64_t>>{*search_param->bf_pks},
            new_meta, 1, context) != 0) {
      LOG_ERROR("Failed to search_bf_by_p_keys_impl vector");
      return core::IndexError_Runtime;
    }
  } else if (search_param->is_linear) {
    if (streamer_->search_bf_impl(vector, new_meta, 1, context) != 0) {
      LOG_ERROR("Failed to search vector");
      return core::IndexError_Runtime;
    }
  } else if (streamer_->search_impl(vector, new_meta, 1, context) != 0) {
    LOG_ERROR("Failed to search vector");
    return core::IndexError_Runtime;
  }
  return 0;
}

int Index::_dense_search_doc_ids(const VectorData &vector_data,
                                 const BaseIndexQueryParam::Pointer &search_param,
                                 int64_t *output_ids, int topk,
                                 core::IndexContext::Pointer &context) {
  if (0 != _execute_dense_search(vector_data, search_param, context)) {
    return core::IndexError_Runtime;
  }
  const auto &docs = context->result();
  const int n = std::min(topk, static_cast<int>(docs.size()));
  for (int i = 0; i < n; ++i) {
    output_ids[i] = static_cast<int64_t>(docs[i].key());
  }
  return 0;
}

int Index::_dense_search(const VectorData &vector_data,
                         const BaseIndexQueryParam::Pointer &search_param,
                         SearchResult *result,
                         core::IndexContext::Pointer &context) {
  if (!std::holds_alternative<DenseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const DenseVector &dense_vector = std::get<DenseVector>(vector_data.vector);
  if (0 != _execute_dense_search(vector_data, search_param, context)) {
    return core::IndexError_Runtime;
  }
  result->doc_list_ = std::move(context->result());

  if (metric_->support_normalize()) {
    for (uint32_t i = 0; i < result->doc_list_.size(); ++i) {
      metric_->normalize(result->doc_list_[i].mutable_score());
    }
  }
  if (reformer_) {
    if (reformer_->normalize(dense_vector.data, input_vector_meta_,
                             result->doc_list_) != 0) {
      LOG_ERROR("Failed to normalize vector");
      return core::IndexError_Runtime;
    }
    if (context->fetch_vector() && reformer_->need_revert()) {
      // TODO: use std::pmr to optimize memory allocation
      result->reverted_vector_list_.resize(context->result().size());
      for (uint32_t i = 0; i < context->result().size(); ++i) {
        std::string &reverted_vector = result->reverted_vector_list_[i];
        reverted_vector.resize(input_vector_meta_.dimension() *
                               input_vector_meta_.unit_size());
        if (reformer_->revert(context->result()[i].vector(), streamer_vector_meta_,
                              &reverted_vector) != 0) {
          LOG_ERROR("Failed to revert vector");
          return core::IndexError_Runtime;
        }
      }
    }
  }

  return 0;
}


int Index::_sparse_search(const VectorData &vector_data,
                          const BaseIndexQueryParam::Pointer &search_param,
                          SearchResult *result,
                          core::IndexContext::Pointer &context) {
  if (!std::holds_alternative<SparseVector>(vector_data.vector)) {
    LOG_ERROR("Invalid vector data");
    return core::IndexError_Runtime;
  }
  const SparseVector &sparse_vector =
      std::get<SparseVector>(vector_data.vector);
  auto indices = sparse_vector.get_indices();
  auto values = sparse_vector.get_values();

  std::string converted_sparse_values_buffer;
  core::IndexQueryMeta new_meta = input_vector_meta_;
  if (reformer_ != nullptr) {
    if (reformer_->transform(sparse_vector.count, indices, values,
                             input_vector_meta_,
                             &converted_sparse_values_buffer, &new_meta) != 0) {
      LOG_ERROR("Failed to transform vector");
      return core::IndexError_Runtime;
    }
    values = converted_sparse_values_buffer.data();
  }

  if (search_param->bf_pks != nullptr) {
    if (streamer_->search_bf_by_p_keys_impl(
            sparse_vector.count, indices, values,
            std::vector<std::vector<uint64_t>>{*search_param->bf_pks}, new_meta,
            context) != 0) {
      LOG_ERROR("Failed to search_bf_by_p_keys_impl vector");
      return core::IndexError_Runtime;
    }
  } else if (search_param->is_linear) {
    if (streamer_->search_bf_impl(sparse_vector.count, indices, values,
                                  new_meta, context) != 0) {
      LOG_ERROR("Failed to search vector");
      return core::IndexError_Runtime;
    }
  } else {
    if (streamer_->search_impl(sparse_vector.count, indices, values, new_meta,
                               context) != 0) {
      LOG_ERROR("Failed to search vector");
      return core::IndexError_Runtime;
    }
  }
  result->doc_list_ = std::move(context->result());

  if (metric_->support_normalize()) {
    for (uint32_t i = 0; i < result->doc_list_.size(); ++i) {
      metric_->normalize(result->doc_list_[i].mutable_score());
    }
  }
  if (reformer_) {
    // TODO: no need to call reformer_->normalize() when sparse?
    if (context->fetch_vector() && reformer_->need_revert()) {
      // TODO: use std::pmr to optimize memory allocation
      auto &result_doc_list = context->result();
      result->reverted_sparse_values_list_.resize(result_doc_list.size());
      for (uint32_t i = 0; i < result_doc_list.size(); ++i) {
        auto &result_doc = result_doc_list[i].sparse_doc();
        std::string &reverted_sparse_values =
            result->reverted_sparse_values_list_[i];
        reverted_sparse_values.resize(result_doc.sparse_count() *
                                      input_vector_meta_.unit_size());
        if (reformer_->revert(result_doc.sparse_count(),
                              reinterpret_cast<const uint32_t *>(
                                  result_doc.sparse_indices().data()),
                              reinterpret_cast<const void *>(
                                  result_doc.sparse_values().data()),
                              new_meta, &reverted_sparse_values) != 0) {
          LOG_ERROR("Failed to revert sparse vector");
          return core::IndexError_Runtime;
        }
      }
    }
  }
  return 0;
}


int Index::Merge(const std::vector<Index::Pointer> &indexes,
                 const IndexFilter &filter, const MergeOptions &options) {
  if (indexes.empty()) {
    return core::IndexError_Success;
  }
  // ivf need builder
  auto reducer =
      core::IndexFactory::CreateStreamerReducer("MixedStreamerReducer");
  if (reducer == nullptr) {
    LOG_ERROR("Failed to create reducer");
    return core::IndexError_Runtime;
  }

  if (options.write_concurrency == 0) {
    LOG_ERROR("Write concurrency must be greater than 0");
    return core::IndexError_InvalidArgument;
  }
  // must declare here to ensure its lifespan can cover reducer->reduce()
  std::unique_ptr<ailego::ThreadPool> local_thread_pool = nullptr;
  if (options.pool != nullptr) {
    reducer->set_thread_pool(options.pool);
  } else {
    local_thread_pool =
        std::make_unique<ailego::ThreadPool>(options.write_concurrency, false);
    reducer->set_thread_pool(local_thread_pool.get());
  }

  ailego::Params reducer_params;
  reducer_params.set(core::PARAM_MIXED_STREAMER_REDUCER_ENABLE_PK_REWRITE,
                     true);
  reducer_params.set(core::PARAM_MIXED_STREAMER_REDUCER_NUM_OF_ADD_THREADS,
                     options.write_concurrency);
  if (reducer->init(reducer_params) != 0) {
    LOG_ERROR("Failed to init reducer");
    return core::IndexError_Runtime;
  }

  if (converter_ != nullptr && reformer_ == nullptr) {
    if (int ret = TrainConverterFromMergeSources(converter_, indexes);
        ret != 0) {
      LOG_ERROR("Failed to train converter from merge sources");
      return ret;
    }
    if (int ret = CreateReformerFromConverterMeta(
            converter_, &proxima_index_meta_, &streamer_vector_meta_,
            &reformer_);
        ret != 0) {
      return ret;
    }
    PersistTrainedMetaToStreamer(proxima_index_meta_, streamer_);
  }

  // The reducer's holder describes the bytes emitted by the source indexes.
  // Native Flat sources emit their physical FP16/UINT8 bytes directly;
  // quantized sources emit public input bytes only when the reducer reverts
  // them first.
  core::IndexQueryMeta reducer_input_meta = input_vector_meta_;
  if (!is_sparse_) {
    bool found_source = false;
    for (const auto &index : indexes) {
      auto provider = index->create_index_provider();
      if (!provider) continue;
      const bool need_revert =
          index->reformer_ != nullptr &&
          ((streamer_->meta().reformer_name() !=
            index->streamer_->meta().reformer_name()) ||
           builder_ != nullptr);
      core::IndexQueryMeta candidate;
      if (need_revert) {
        candidate = index->input_vector_meta_;
      } else {
        candidate.set_meta(provider->data_type(), provider->dimension());
      }
      if (!found_source) {
        reducer_input_meta = candidate;
        found_source = true;
      } else if (candidate.data_type() != reducer_input_meta.data_type() ||
                 candidate.dimension() != reducer_input_meta.dimension() ||
                 candidate.element_size() !=
                     reducer_input_meta.element_size()) {
        LOG_ERROR("Merge sources expose different logical vector types");
        return core::IndexError_Mismatch;
      }
    }
  }

  if (reducer->set_target_streamer_wiht_info(
          builder_, streamer_, converter_, reformer_, reducer_input_meta) != 0) {
    LOG_ERROR("Failed to set target streamer");
    return core::IndexError_Runtime;
  }

  for (const auto &index : indexes) {
    if (reducer->feed_streamer_with_reformer(index->streamer_,
                                             index->reformer_) != 0) {
      LOG_ERROR("Failed to feed streamer");
      return core::IndexError_Runtime;
    }
  }
  if (streamer_ != nullptr) {
    int ret = streamer_->begin_bulk_build();
    if (ret != 0) {
      LOG_ERROR("Failed to begin bulk build, err: %s",
                core::IndexError::What(ret));
      return ret;
    }
  }
  if (reducer->reduce(filter) != 0) {
    LOG_ERROR("Failed to reduce");
    return core::IndexError_Runtime;
  }
  // All source vectors are now present in the target streamer. Give streaming
  // graph indexes a single pre-persistence finalization point (Vamana uses it
  // for the optional configured-alpha second graph pass).
  if (streamer_ != nullptr) {
    int ret = streamer_->finalize_build();
    if (ret != 0) {
      LOG_ERROR("Failed to finalize merged index, err: %s",
                core::IndexError::What(ret));
      return ret;
    }
  }
  PersistTrainedMetaToStreamer(proxima_index_meta_, streamer_);
  is_trained_ = true;
  return 0;
}

int Index::_get_coarse_search_topk(
    const BaseIndexQueryParam::Pointer &search_param) {
  float scale_factor = search_param->refiner_param->scale_factor_;
  if (scale_factor == 0) {
    scale_factor = 1;
  }
  return floor(search_param->topk * scale_factor);
}

std::string Index::get_metric_name(MetricType metric_type, bool is_sparse) {
  if (is_sparse) {
    switch (metric_type) {
      case MetricType::kInnerProduct:
        return "InnerProductSparse";
      case MetricType::kMIPSL2sq:
        return "MipsSquaredEuclideanSparse";
      default:
        return "";
    }
  } else {
    switch (metric_type) {
      case MetricType::kL2sq:
        return "SquaredEuclidean";
      case MetricType::kInnerProduct:
        return "InnerProduct";
      case MetricType::kCosine:
        return "Cosine";
      case MetricType::kMIPSL2sq:
        return "MipsSquaredEuclidean";
      default:
        return "";
    }
  }
}

}  // namespace zvec::core_interface
