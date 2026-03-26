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

#include <zvec/core/interface/index_bridge.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

#include <zvec/ailego/io/file.h>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/framework/index_filter.h>
#include <zvec/core/framework/index_logger.h>
#include <zvec/core/interface/index_factory.h>

namespace zvec::core_interface {

// ============================================================================
// Impl
// ============================================================================

struct IndexBridge::Impl {
  // Target index parameters (e.g., HNSW) - use pointer to avoid slicing
  BaseIndexParam::Pointer target_param;

  // Build phase: Target index (e.g., HNSW)
  Index::Pointer target_index;
  std::string target_temp_dir;
  std::string target_file_path;

  std::mutex mutex;
  uint32_t doc_count = 0;

  ~Impl() { Cleanup(); }

  void Cleanup() {
    if (target_index) {
      target_index->Close();
      target_index.reset();
    }
    if (!target_temp_dir.empty()) {
      ailego::File::RemoveDirectory(target_temp_dir);
      target_temp_dir.clear();
    }
  }

  static std::string CreateTempDir(const std::string& suffix) {
    std::string base_dir;
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMP");
    if (!tmp) tmp = ".";
    base_dir = tmp;
#else
    base_dir = "/tmp";
#endif
    static std::atomic<uint64_t> counter{0};
    std::ostringstream oss;
    oss << base_dir << "/zvec_batch_" << suffix << "_"
        << static_cast<uint64_t>(getpid()) << "_"
        << counter.fetch_add(1, std::memory_order_relaxed);
    std::string dir_path = oss.str();
    ailego::File::MakePath(dir_path);
    return dir_path;
  }

  // Clone the target parameter to preserve derived type
  static BaseIndexParam::Pointer CloneParam(const BaseIndexParam& param) {
    switch (param.index_type) {
      case IndexType::kHNSW: {
        auto* hnsw = dynamic_cast<const HNSWIndexParam*>(&param);
        if (hnsw) {
          return std::make_shared<HNSWIndexParam>(*hnsw);
        }
        break;
      }
      case IndexType::kFlat: {
        auto* flat = dynamic_cast<const FlatIndexParam*>(&param);
        if (flat) {
          return std::make_shared<FlatIndexParam>(*flat);
        }
        break;
      }
      case IndexType::kIVF: {
        auto* ivf = dynamic_cast<const IVFIndexParam*>(&param);
        if (ivf) {
          return std::make_shared<IVFIndexParam>(*ivf);
        }
        break;
      }
      case IndexType::kHNSWRabitq: {
        auto* rabitq = dynamic_cast<const HNSWRabitqIndexParam*>(&param);
        if (rabitq) {
          return std::make_shared<HNSWRabitqIndexParam>(*rabitq);
        }
        break;
      }
      default:
        break;
    }
    // Fallback: create base param (will likely fail for specialized indexes)
    return std::make_shared<BaseIndexParam>(param);
  }
};

// ============================================================================
// Lifecycle
// ============================================================================

IndexBridge::IndexBridge() : impl_(std::make_unique<Impl>()) {}

IndexBridge::~IndexBridge() = default;

IndexBridge::IndexBridge(IndexBridge&&) noexcept = default;
IndexBridge& IndexBridge::operator=(IndexBridge&&) noexcept =
    default;

IndexBridge::Pointer IndexBridge::Create(
    const BaseIndexParam& target_param) {
  auto bridge = Pointer(new IndexBridge());
  auto& impl = bridge->impl_;

  // Store target parameters (clone to preserve derived type)
  impl->target_param = Impl::CloneParam(target_param);
  if (!impl->target_param) {
    LOG_ERROR("IndexBridge::Create: failed to clone target param");
    return nullptr;
  }

  impl->target_index = IndexFactory::CreateAndInitIndex(*impl->target_param);
  if (!impl->target_index) {
    LOG_ERROR("IndexBridge::Build: failed to create target index");
    return nullptr;
  }

  impl->target_temp_dir = Impl::CreateTempDir("target");
  impl->target_file_path = impl->target_temp_dir + "/target.dat";

  StorageOptions storage_options;
  storage_options.type = StorageOptions::StorageType::kMMAP;
  storage_options.create_new = true;
  storage_options.read_only = false;

  int ret = impl->target_index->Open(impl->target_file_path, storage_options);
  if (ret != 0) {
    LOG_ERROR("IndexBridge::Build: failed to open target, err=%d", ret);
    return nullptr;
  }

  return bridge;
}

IndexBridge::Pointer IndexBridge::Deserialize(
    const std::string& param_json, const void* data, size_t size) {
  if (!data || size == 0) {
    LOG_ERROR("IndexBridge::Deserialize: empty data");
    return nullptr;
  }

  // Deserialize the index parameters from JSON
  auto param_ptr = IndexFactory::DeserializeIndexParamFromJson(param_json);
  if (!param_ptr) {
    LOG_ERROR("IndexBridge::Deserialize: failed to parse param JSON");
    return nullptr;
  }

  auto bridge = Pointer(new IndexBridge());
  auto& impl = bridge->impl_;
  impl->target_param = param_ptr;  // Store the shared_ptr directly

  // Create the target index directly (already built)
  impl->target_index = IndexFactory::CreateAndInitIndex(*impl->target_param);
  if (!impl->target_index) {
    LOG_ERROR("IndexBridge::Deserialize: failed to create target index");
    return nullptr;
  }

  // Write serialized data to temp file
  impl->target_temp_dir = Impl::CreateTempDir("target");
  impl->target_file_path = impl->target_temp_dir + "/target.dat";

  {
    std::ofstream ofs(impl->target_file_path,
                      std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      LOG_ERROR("IndexBridge::Deserialize: failed to create temp file");
      impl->Cleanup();
      return nullptr;
    }
    ofs.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(size));
    if (!ofs.good()) {
      LOG_ERROR("IndexBridge::Deserialize: failed to write temp file");
      impl->Cleanup();
      return nullptr;
    }
  }

  // Open target index
  StorageOptions storage_options;
  storage_options.type = StorageOptions::StorageType::kMMAP;
  storage_options.create_new = false;
  storage_options.read_only = false;

  int ret =
      impl->target_index->Open(impl->target_file_path, storage_options);
  if (ret != 0) {
    LOG_ERROR("IndexBridge::Deserialize: failed to open target, err=%d",
              ret);
    impl->Cleanup();
    return nullptr;
  }

  impl->doc_count = impl->target_index->GetDocCount();

  return bridge;
}

// ============================================================================
// Write Operations
// ============================================================================

int IndexBridge::Add(uint32_t doc_id, const float* vector,
                          uint32_t dimension) {
  if (!impl_ || !impl_->target_index) return -1;

  DenseVector dense_vec;
  dense_vec.data = vector;

  VectorData vector_data;
  vector_data.vector = dense_vec;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  int ret = impl_->target_index->Add(vector_data, doc_id);
  if (ret == 0) {
    impl_->doc_count++;
  }
  return ret;
}

int IndexBridge::Build(int concurrency) {
  if (!impl_) return -1;
  if (!impl_->target_index) {
    LOG_ERROR("IndexBridge::Build: failed to create target index");
    return -1;
  }
  LOG_INFO("IndexBridge::Build: target index created");
  
  int ret = impl_->target_index->Train();
  if (ret != 0) {
    LOG_ERROR("IndexBridge::Build: Train failed, err=%d", ret);
    return ret;
  }
  LOG_INFO("IndexBridge::Build: Train succeeded");

  // Flush target index
  ret = impl_->target_index->Flush();
  if (ret != 0) {
    LOG_ERROR("IndexBridge::Build: failed to flush target");
    return ret;
  }

  return 0;
}

// ============================================================================
// Query Operations
// ============================================================================

int IndexBridge::Search(const float* query, uint32_t dimension,
                             uint32_t topk,
                             const BaseIndexQueryParam* query_param,
                             std::vector<BridgeSearchResultItem>* results) {
  (void)dimension;  // Not used currently
  
  if (!impl_ || !results) return -1;

  Index::Pointer& index = impl_->target_index;
  if (!index) return -1;

  results->clear();

  DenseVector dense_vec;
  dense_vec.data = query;

  VectorData vector_data;
  vector_data.vector = dense_vec;

  // Build search param
  BaseIndexQueryParam::Pointer search_param;
  if (query_param) {
    search_param = query_param->Clone();
  } else {
    switch (impl_->target_param->index_type) {
      case IndexType::kFlat:
        search_param = std::make_shared<FlatQueryParam>();
        break;
      case IndexType::kHNSW:
        search_param = std::make_shared<HNSWQueryParam>();
        break;
      case IndexType::kIVF:
        search_param = std::make_shared<IVFQueryParam>();
        break;
      default:
        search_param = std::make_shared<FlatQueryParam>();
        break;
    }
  }
  search_param->topk = topk;

  SearchResult search_result;
  int ret;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ret = index->Search(vector_data, search_param, &search_result);
  }
  if (ret != 0) return ret;

  // Convert results
  results->reserve(search_result.doc_list_.size());
  for (const auto& doc : search_result.doc_list_) {
    BridgeSearchResultItem item;
    item.doc_id = static_cast<uint32_t>(doc.key());
    item.distance = doc.score();
    results->push_back(item);
  }

  return 0;
}

// ============================================================================
// Serialization
// ============================================================================

int IndexBridge::Serialize(std::string* output) {
  if (!impl_ || !output) return -1;

  if (!impl_->target_index) return -1;

  // Flush target index
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    int ret = impl_->target_index->Flush();
    if (ret != 0) {
      LOG_ERROR("IndexBridge::Serialize: flush failed, err=%d", ret);
      return ret;
    }
  }

  // Read target index file
  std::ifstream ifs(impl_->target_file_path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    LOG_ERROR("IndexBridge::Serialize: failed to open target file");
    return -1;
  }

  auto file_size = ifs.tellg();
  if (file_size <= 0) {
    LOG_ERROR("IndexBridge::Serialize: target file is empty");
    return -1;
  }

  ifs.seekg(0, std::ios::beg);
  output->resize(static_cast<size_t>(file_size));
  ifs.read(output->data(), file_size);

  if (!ifs.good()) {
    LOG_ERROR("IndexBridge::Serialize: failed to read target file");
    return -1;
  }

  return 0;
}

std::string IndexBridge::GetParamJson() const {
  if (!impl_ || !impl_->target_param) return "{}";
  return impl_->target_param->SerializeToJson();
}

// ============================================================================
// Metadata
// ============================================================================

uint32_t IndexBridge::DocCount() const {
  if (!impl_) return 0;
  return impl_->doc_count;
}

IndexType IndexBridge::GetIndexType() const {
  if (!impl_ || !impl_->target_param) return IndexType::kNone;
  return impl_->target_param->index_type;
}

MetricType IndexBridge::GetMetricType() const {
  if (!impl_ || !impl_->target_param) return MetricType::kNone;
  return impl_->target_param->metric_type;
}

uint32_t IndexBridge::GetDimension() const {
  if (!impl_ || !impl_->target_param) return 0;
  return static_cast<uint32_t>(impl_->target_param->dimension);
}

bool IndexBridge::IsBuilt() const {
  if (!impl_) return false;
  if (!impl_->target_index) return false;
  return true;
}

int IndexBridge::Flush() {
  if (!impl_) return -1;
  std::lock_guard<std::mutex> lock(impl_->mutex);

  if (impl_->target_index) {
    return impl_->target_index->Flush();
  }
  return 0;
}

}  // namespace zvec::core_interface
