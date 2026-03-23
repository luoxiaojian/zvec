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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <zvec/core/interface/index.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/index_param.h>

namespace zvec::core_interface {

/// A single KNN search result entry.
struct BridgeSearchResultItem {
  uint32_t doc_id;
  float distance;
};

/// IndexBridge: a stable, filesystem-decoupled interface for embedded use.
///
/// It wraps zvec's core Index with a temporary mmap-backed storage directory,
/// and exposes Create / Add / Search / Serialize / Deserialize operations
/// without requiring the caller to manage file paths or storage details.
///
/// Designed as the sole dependency surface for sqlite-zvec's zvec0 module.
class IndexBridge {
 public:
  using Pointer = std::shared_ptr<IndexBridge>;

  /// Create a new empty index backed by a temporary directory.
  /// The caller specifies index parameters (type, dimension, metric, etc.).
  /// Returns nullptr on failure.
  static Pointer Create(const BaseIndexParam& param);

  /// Restore an index from a serialized byte buffer.
  /// @param param_json  JSON string of index parameters (from GetParamJson()).
  /// @param data        Pointer to the serialized index data.
  /// @param size        Size of the serialized data in bytes.
  /// Returns nullptr on failure.
  static Pointer Deserialize(const std::string& param_json,
                             const void* data, size_t size);

  ~IndexBridge();

  // Non-copyable, movable
  IndexBridge(const IndexBridge&) = delete;
  IndexBridge& operator=(const IndexBridge&) = delete;
  IndexBridge(IndexBridge&&) noexcept;
  IndexBridge& operator=(IndexBridge&&) noexcept;

  // ===== Write Operations =====

  /// Add a dense float32 vector with the given doc_id.
  /// The doc_id is managed by the caller (e.g., mapped from SQLite rowid).
  /// @return 0 on success, non-zero on error.
  int Add(uint32_t doc_id, const float* vector, uint32_t dimension);

  /// Mark a document as deleted. Space is not immediately reclaimed.
  /// @return 0 on success, non-zero on error.
  int Remove(uint32_t doc_id);

  // ===== Query Operations =====

  /// Perform a KNN search and return the top-k results.
  /// @param query       Query vector (float32).
  /// @param dimension   Dimension of the query vector.
  /// @param topk        Number of nearest neighbors to return.
  /// @param query_param Optional query parameters (ef_search, nprobe, etc.).
  ///                    Pass nullptr to use defaults.
  /// @param results     Output vector of search results.
  /// @return 0 on success, non-zero on error.
  int Search(const float* query, uint32_t dimension, uint32_t topk,
             const BaseIndexQueryParam* query_param,
             std::vector<BridgeSearchResultItem>* results);

  // ===== Serialization =====

  /// Serialize the entire index into a byte buffer.
  /// The output can later be passed to Deserialize() to restore the index.
  /// @return 0 on success, non-zero on error.
  int Serialize(std::string* output);

  // ===== Metadata =====

  /// Return the number of documents currently in the index.
  uint32_t DocCount() const;

  /// Return the index parameters as a JSON string.
  /// This JSON can be stored alongside the serialized data for later recovery.
  std::string GetParamJson() const;

  /// Return the index type.
  IndexType GetIndexType() const;

  /// Return the metric type.
  MetricType GetMetricType() const;

  /// Return the vector dimension.
  uint32_t GetDimension() const;

  // ===== Index Maintenance =====

  /// For indexes that require training (e.g., IVF), trigger training.
  /// @return 0 on success, non-zero on error.
  int Train();

  /// Check whether the index has been trained.
  bool IsTrained() const;

  /// Flush pending writes to the underlying storage.
  /// @return 0 on success, non-zero on error.
  int Flush();

 private:
  IndexBridge();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace zvec::core_interface
