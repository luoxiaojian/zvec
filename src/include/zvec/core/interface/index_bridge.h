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

#ifndef ZVEC_CORE_INTERFACE_INDEX_BRIDGE_H_
#define ZVEC_CORE_INTERFACE_INDEX_BRIDGE_H_

#include <memory>
#include <string>
#include <vector>

#include <zvec/core/interface/index_param.h>

namespace zvec::core_interface {

/// A single KNN search result entry.
struct BridgeSearchResultItem {
  uint32_t doc_id;
  float distance;
};

/**
 * @brief IndexBridge - Optimized for batch index building scenarios.
 *
 * Unlike IndexBridge which adds vectors directly to the target index (e.g., HNSW),
 * IndexBridge uses a two-phase approach:
 *   1. Collection phase: Vectors are added to a Flat index (O(1) per vector)
 *   2. Build phase: Batch-builds the target index using Merge (much faster)
 *
 * This is ~100-500x faster than IndexBridge for bulk loading scenarios like
 * vec0's build_index command.
 *
 * Usage:
 *   auto bridge = IndexBridge::Create(hnswParam);
 *   for (auto& vec : vectors) {
 *     bridge->Add(doc_id++, vec.data(), dim);
 *   }
 *   bridge->Build();  // Batch-build HNSW from collected vectors
 *   bridge->Serialize(&output);
 */
class IndexBridge {
 public:
  using Pointer = std::shared_ptr<IndexBridge>;

  ~IndexBridge();

  // Non-copyable
  IndexBridge(const IndexBridge&) = delete;
  IndexBridge& operator=(const IndexBridge&) = delete;

  // Movable
  IndexBridge(IndexBridge&&) noexcept;
  IndexBridge& operator=(IndexBridge&&) noexcept;

  /**
   * @brief Create a new IndexBridge for batch index building.
   * @param target_param Parameters for the target index (e.g., HNSW, IVF).
   *                     Vectors will be collected first, then batch-built.
   * @return Pointer to the bridge, or nullptr on failure.
   */
  static Pointer Create(const BaseIndexParam& target_param);

  /**
   * @brief Deserialize a previously serialized index.
   * @param param_json JSON string of index parameters.
   * @param data Serialized index data.
   * @param size Size of the serialized data.
   * @return Pointer to the bridge, or nullptr on failure.
   */
  static Pointer Deserialize(const std::string& param_json, const void* data,
                             size_t size);

  // ========== Write Operations (Collection Phase) ==========

  /**
   * @brief Add a vector to the collection (O(1) operation).
   * @param doc_id Document ID for this vector.
   * @param vector Pointer to the vector data.
   * @param dimension Dimension of the vector.
   * @return 0 on success, non-zero on error.
   */
  int Add(uint32_t doc_id, const float* vector, uint32_t dimension);

  /**
   * @brief Build the target index from collected vectors.
   *
   * This performs batch index construction using Merge, which is much faster
   * than adding vectors one by one to HNSW.
   *
   * @param concurrency Number of threads for building (0 = use default).
   * @return 0 on success, non-zero on error.
   */
  int Build(int concurrency = 0);

  // ========== Query Operations ==========

  /**
   * @brief Search the index for nearest neighbors.
   * @note Must call Build() before searching.
   */
  int Search(const float* query, uint32_t dimension, uint32_t topk,
             const BaseIndexQueryParam* query_param,
             std::vector<BridgeSearchResultItem>* results);

  // ========== Serialization ==========

  /**
   * @brief Serialize the built index to a string.
   * @note Must call Build() before serializing.
   */
  int Serialize(std::string* output);

  /**
   * @brief Get the index parameters as JSON string.
   */
  std::string GetParamJson() const;

  // ========== Metadata ==========

  uint32_t DocCount() const;
  IndexType GetIndexType() const;
  MetricType GetMetricType() const;
  uint32_t GetDimension() const;
  bool IsBuilt() const;

  // ========== Index Maintenance ==========

  int Flush();

 private:
  IndexBridge();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace zvec::core_interface

#endif  // ZVEC_CORE_INTERFACE_INDEX_BRIDGE_H_
