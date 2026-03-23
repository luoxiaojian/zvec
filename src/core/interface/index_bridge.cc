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
#include <mutex>
#include <sstream>

#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec::core_interface {

// ============================================================================
// Impl
// ============================================================================

struct IndexBridge::Impl {
  Index::Pointer index;
  BaseIndexParam param;
  std::string temp_dir;
  std::string index_file_path;
  std::mutex mutex;
  uint32_t doc_count = 0;

  ~Impl() { Cleanup(); }

  void Cleanup() {
    if (index) {
      index->Close();
      index.reset();
    }
    if (!temp_dir.empty()) {
      ailego::File::RemoveDirectory(temp_dir);
      temp_dir.clear();
    }
  }

  static std::string CreateTempDir() {
    std::string base_dir;
#ifdef _WIN32
    const char* tmp = std::getenv("TEMP");
    if (!tmp) tmp = std::getenv("TMP");
    if (!tmp) tmp = ".";
    base_dir = tmp;
#else
    base_dir = "/tmp";
#endif
    // Generate a unique directory name using a combination of pid and counter
    static std::atomic<uint64_t> counter{0};
    std::ostringstream oss;
    oss << base_dir << "/zvec_bridge_"
        << static_cast<uint64_t>(getpid()) << "_"
        << counter.fetch_add(1, std::memory_order_relaxed);
    std::string dir_path = oss.str();
    ailego::File::MakePath(dir_path);
    return dir_path;
  }

  std::string GetIndexFilePath() const {
    return temp_dir + "/index.dat";
  }
};

// ============================================================================
// Lifecycle
// ============================================================================

IndexBridge::IndexBridge() : impl_(std::make_unique<Impl>()) {}

IndexBridge::~IndexBridge() = default;

IndexBridge::IndexBridge(IndexBridge&&) noexcept = default;
IndexBridge& IndexBridge::operator=(IndexBridge&&) noexcept = default;

IndexBridge::Pointer IndexBridge::Create(const BaseIndexParam& param) {
  auto bridge = Pointer(new IndexBridge());
  auto& impl = bridge->impl_;

  // Store a copy of the param
  impl->param = param;

  // Create the underlying Index
  impl->index = IndexFactory::CreateAndInitIndex(param);
  if (!impl->index) {
    LOG_ERROR("IndexBridge::Create: failed to create index");
    return nullptr;
  }

  // Create a temporary directory for the mmap storage
  impl->temp_dir = Impl::CreateTempDir();
  impl->index_file_path = impl->GetIndexFilePath();

  // Open the index with mmap storage in create-new mode
  StorageOptions storage_options;
  storage_options.type = StorageOptions::StorageType::kMMAP;
  storage_options.create_new = true;
  storage_options.read_only = false;

  int ret = impl->index->Open(impl->index_file_path, storage_options);
  if (ret != 0) {
    LOG_ERROR("IndexBridge::Create: failed to open index, err=%d", ret);
    impl->Cleanup();
    return nullptr;
  }

  return bridge;
}

IndexBridge::Pointer IndexBridge::Deserialize(const std::string& param_json,
                                              const void* data, size_t size) {
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
  impl->param = *param_ptr;

  // Create the underlying Index
  impl->index = IndexFactory::CreateAndInitIndex(impl->param);
  if (!impl->index) {
    LOG_ERROR("IndexBridge::Deserialize: failed to create index");
    return nullptr;
  }

  // Create a temporary directory and write the serialized data to a file
  impl->temp_dir = Impl::CreateTempDir();
  impl->index_file_path = impl->GetIndexFilePath();

  // Write the raw index data to the temp file
  {
    std::ofstream ofs(impl->index_file_path,
                      std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      LOG_ERROR("IndexBridge::Deserialize: failed to create temp file: %s",
                impl->index_file_path.c_str());
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

  // Open the index from the written file (not create_new, it already exists)
  StorageOptions storage_options;
  storage_options.type = StorageOptions::StorageType::kMMAP;
  storage_options.create_new = false;
  storage_options.read_only = false;

  int ret = impl->index->Open(impl->index_file_path, storage_options);
  if (ret != 0) {
    LOG_ERROR("IndexBridge::Deserialize: failed to open index, err=%d", ret);
    impl->Cleanup();
    return nullptr;
  }

  // Recover the doc count from the index
  impl->doc_count = impl->index->GetDocCount();

  return bridge;
}

// ============================================================================
// Write Operations
// ============================================================================

int IndexBridge::Add(uint32_t doc_id, const float* vector, uint32_t dimension) {
  if (!impl_ || !impl_->index) return -1;

  DenseVector dense_vec;
  dense_vec.data = vector;

  VectorData vector_data;
  vector_data.vector = dense_vec;

  std::lock_guard<std::mutex> lock(impl_->mutex);
  int ret = impl_->index->Add(vector_data, doc_id);
  if (ret == 0) {
    impl_->doc_count++;
  }
  return ret;
}

int IndexBridge::Remove(uint32_t doc_id) {
  if (!impl_ || !impl_->index) return -1;

  // zvec core Index does not expose a direct Remove(doc_id) at the Index level.
  // The streamer has remove_impl(key, context), but it's not exposed through
  // the Index public API. For now, we track deletions at the sqlite-zvec level
  // (shadow table deletes + rebuild on reconnect).
  // This is consistent with the architecture doc's design: _vectors is the
  // source of truth, and the index can be rebuilt from _vectors.
  (void)doc_id;
  return 0;
}

// ============================================================================
// Query Operations
// ============================================================================

int IndexBridge::Search(const float* query, uint32_t dimension, uint32_t topk,
                        const BaseIndexQueryParam* query_param,
                        std::vector<BridgeSearchResultItem>* results) {
  if (!impl_ || !impl_->index || !results) return -1;

  results->clear();

  DenseVector dense_vec;
  dense_vec.data = query;

  VectorData vector_data;
  vector_data.vector = dense_vec;

  // Build a query param if none provided; use defaults based on index type
  BaseIndexQueryParam::Pointer search_param;
  if (query_param) {
    search_param = query_param->Clone();
  } else {
    // Create a default query param based on index type
    switch (impl_->param.index_type) {
      case IndexType::kFlat:
        search_param = std::make_shared<FlatQueryParam>();
        break;
      case IndexType::kHNSW:
        search_param = std::make_shared<HNSWQueryParam>();
        break;
      case IndexType::kIVF:
        search_param = std::make_shared<IVFQueryParam>();
        break;
      case IndexType::kHNSWRabitq:
        search_param = std::make_shared<HNSWRabitqQueryParam>();
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
    ret = impl_->index->Search(vector_data, search_param, &search_result);
  }
  if (ret != 0) {
    return ret;
  }

  // Convert IndexDocumentList to BridgeSearchResultItem
  results->reserve(search_result.doc_list_.size());
  for (const auto& doc : search_result.doc_list_) {
    BridgeSearchResultItem item;
    // In zvec core, key() is uint64_t (the doc_id we passed to Add)
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
  if (!impl_ || !impl_->index || !output) return -1;

  // Flush the index to ensure all data is written to the mmap file
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    int ret = impl_->index->Flush();
    if (ret != 0) {
      LOG_ERROR("IndexBridge::Serialize: failed to flush index, err=%d", ret);
      return ret;
    }
  }

  // Read the entire mmap file into the output buffer
  std::ifstream ifs(impl_->index_file_path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    LOG_ERROR("IndexBridge::Serialize: failed to open index file: %s",
              impl_->index_file_path.c_str());
    return -1;
  }

  auto file_size = ifs.tellg();
  if (file_size <= 0) {
    LOG_ERROR("IndexBridge::Serialize: index file is empty");
    return -1;
  }

  ifs.seekg(0, std::ios::beg);
  output->resize(static_cast<size_t>(file_size));
  ifs.read(output->data(), file_size);

  if (!ifs.good()) {
    LOG_ERROR("IndexBridge::Serialize: failed to read index file");
    return -1;
  }

  return 0;
}

// ============================================================================
// Metadata
// ============================================================================

uint32_t IndexBridge::DocCount() const {
  if (!impl_) return 0;
  return impl_->doc_count;
}

std::string IndexBridge::GetParamJson() const {
  if (!impl_) return "{}";
  return impl_->param.SerializeToJson();
}

IndexType IndexBridge::GetIndexType() const {
  if (!impl_) return IndexType::kNone;
  return impl_->param.index_type;
}

MetricType IndexBridge::GetMetricType() const {
  if (!impl_) return MetricType::kNone;
  return impl_->param.metric_type;
}

uint32_t IndexBridge::GetDimension() const {
  if (!impl_) return 0;
  return static_cast<uint32_t>(impl_->param.dimension);
}

// ============================================================================
// Index Maintenance
// ============================================================================

int IndexBridge::Train() {
  if (!impl_ || !impl_->index) return -1;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->index->Train();
}

bool IndexBridge::IsTrained() const {
  if (!impl_ || !impl_->index) return false;
  return impl_->index->IsTrained();
}

int IndexBridge::Flush() {
  if (!impl_ || !impl_->index) return -1;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->index->Flush();
}

}  // namespace zvec::core_interface
