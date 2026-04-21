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
#include "vamana_streamer_entity.h"
#include <sys/mman.h>
#include <zvec/ailego/hash/crc32c.h>
#include <zvec/core/framework/index_stats.h>

namespace zvec {
namespace core {

VamanaStreamerEntity::VamanaStreamerEntity(IndexStreamer::Stats &stats)
    : stats_(stats) {
  keys_map_lock_ = std::make_shared<ailego::SharedMutex>();
  keys_map_ = std::make_shared<HashMap<key_t, node_id_t>>();
  keys_map_->set_empty_key(kInvalidKey);
  broker_ = std::make_shared<ChunkBroker>(stats);
}

VamanaStreamerEntity::~VamanaStreamerEntity() {}

int VamanaStreamerEntity::cleanup() {
  node_chunks_.clear();
  if (keys_map_) {
    keys_map_->clear();
  }
  header_.clear();
  return 0;
}

int VamanaStreamerEntity::init(size_t /*max_doc_cnt*/) {
  // node_size = vector_size + key_size + neighbors_size
  set_node_size(vector_size() + sizeof(key_t) + neighbors_size());
  neighbor_size_ = neighbors_size();
  return 0;
}

key_t VamanaStreamerEntity::get_key(node_id_t id) const {
  if (!use_key_info_map_) return id;
  auto loc = get_key_chunk_loc(id);
  if (ailego_unlikely(loc.first >= node_chunks_.size())) return kInvalidKey;
  const void *ptr = nullptr;
  size_t ret = node_chunks_[loc.first]->read(loc.second, &ptr, sizeof(key_t));
  if (ailego_unlikely(ret != sizeof(key_t))) {
    LOG_ERROR("Read key failed, ret=%zu", ret);
    return kInvalidKey;
  }
  return *reinterpret_cast<const key_t *>(ptr);
}

const void *VamanaStreamerEntity::get_vector(node_id_t id) const {
  auto loc = get_vector_chunk_loc(id);
  if (ailego_unlikely(loc.first >= node_chunks_.size())) return nullptr;
  const void *ptr = nullptr;
  size_t ret =
      node_chunks_[loc.first]->read(loc.second, &ptr, vector_size());
  if (ailego_unlikely(ret != vector_size())) {
    LOG_ERROR("Read vector failed, ret=%zu", ret);
    return nullptr;
  }
  return ptr;
}

int VamanaStreamerEntity::get_vector(const node_id_t id,
                                    IndexStorage::MemoryBlock &block) const {
  auto loc = get_vector_chunk_loc(id);
  if (ailego_unlikely(loc.first >= node_chunks_.size()))
    return IndexError_NoExist;
  size_t ret =
      node_chunks_[loc.first]->read(loc.second, block, vector_size());
  if (ailego_unlikely(ret != vector_size())) {
    LOG_ERROR("Read vector failed, ret=%zu", ret);
    return IndexError_ReadData;
  }
  return 0;
}

int VamanaStreamerEntity::get_vector(const node_id_t *ids, uint32_t count,
                                    const void **vecs) const {
  for (uint32_t i = 0; i < count; ++i) {
    vecs[i] = get_vector(ids[i]);
    if (ailego_unlikely(vecs[i] == nullptr)) {
      return IndexError_NoExist;
    }
  }
  return 0;
}

int VamanaStreamerEntity::get_vector(
    const node_id_t *ids, uint32_t count,
    std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
  vec_blocks.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    int ret = get_vector(ids[i], vec_blocks[i]);
    if (ailego_unlikely(ret != 0)) return ret;
  }
  return 0;
}

const Neighbors VamanaStreamerEntity::get_neighbors(node_id_t id) const {
  auto loc = get_neighbor_chunk_loc(id);
  IndexStorage::MemoryBlock mem_block;
  size_t ret = loc.first->read(loc.second, mem_block, neighbor_size_);
  if (ailego_unlikely(ret != neighbor_size_)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", ret);
    return Neighbors();
  }
  return Neighbors(mem_block);
}

int VamanaStreamerEntity::add_vector(key_t key, const void *vec,
                                    node_id_t *id) {
  Chunk::Pointer node_chunk;
  size_t chunk_offset = static_cast<size_t>(-1);

  std::lock_guard<std::mutex> lock(mutex_);

  node_id_t local_id = static_cast<node_id_t>(doc_cnt());
  uint32_t chunk_index = node_chunks_.size() - 1U;
  if (chunk_index == static_cast<uint32_t>(-1) ||
      (node_chunks_[chunk_index]->data_size() >=
       node_cnt_per_chunk_ * node_size())) {
    if (ailego_unlikely(node_chunks_.capacity() == node_chunks_.size())) {
      LOG_ERROR("add vector failed for no memory quota");
      return IndexError_IndexFull;
    }
    chunk_index++;
    auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NODE, chunk_index,
                                  chunk_size_);
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc data chunk failed");
      return p.first;
    }
    node_chunk = p.second;
    chunk_offset = 0UL;
    node_chunks_.emplace_back(node_chunk);
  } else {
    node_chunk = node_chunks_[chunk_index];
    chunk_offset = node_chunk->data_size();
  }

  // Write vector
  size_t size = node_chunk->write(chunk_offset, vec, vector_size());
  if (ailego_unlikely(size != vector_size())) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  // Write key
  size = node_chunk->write(chunk_offset + vector_size(), &key, sizeof(key_t));
  if (ailego_unlikely(size != sizeof(key_t))) {
    LOG_ERROR("Chunk write key failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  // Neighbors are initialized to zero by default (chunk is zero-filled)

  chunk_offset += node_size();
  if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
    LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
    return IndexError_Runtime;
  }

  if (use_key_info_map_) {
    keys_map_lock_->lock();
    (*keys_map_)[key] = local_id;
    keys_map_lock_->unlock();
  }

  *mutable_doc_cnt() += 1;
  broker_->mark_dirty();
  *id = local_id;

  return 0;
}

int VamanaStreamerEntity::add_vector_with_id(node_id_t id, const void *vec) {
  Chunk::Pointer node_chunk;
  size_t chunk_offset = static_cast<size_t>(-1);
  key_t key = id;

  std::lock_guard<std::mutex> lock(mutex_);

  auto func_get_node_chunk_and_offset = [&](node_id_t node_id) -> int {
    uint32_t chunk_idx = node_id >> node_index_mask_bits_;
    ailego_assert_with(chunk_idx <= node_chunks_.size(), "invalid chunk idx");
    if (chunk_idx == node_chunks_.size()) {
      if (ailego_unlikely(node_chunks_.capacity() == node_chunks_.size())) {
        LOG_ERROR("add vector failed for no memory quota");
        return IndexError_IndexFull;
      }
      auto p = broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_NODE, chunk_idx,
                                    chunk_size_);
      if (ailego_unlikely(p.first != 0)) {
        LOG_ERROR("Alloc data chunk failed");
        return p.first;
      }
      node_chunk = p.second;
      node_chunks_.emplace_back(node_chunk);
    }
    node_chunk = node_chunks_[chunk_idx];
    chunk_offset = (node_id & node_index_mask_) * node_size();
    return 0;
  };

  // Fill gaps with invalid keys
  for (size_t start_id = doc_cnt(); start_id < id; ++start_id) {
    if (auto ret = func_get_node_chunk_and_offset(start_id); ret != 0) {
      return ret;
    }
    size_t size = node_chunk->write(chunk_offset + vector_size(), &kInvalidKey,
                                    sizeof(key_t));
    if (ailego_unlikely(size != sizeof(key_t))) {
      LOG_ERROR("Chunk write key failed, ret=%zu", size);
      return IndexError_WriteData;
    }
    chunk_offset += node_size();
    if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
      return IndexError_Runtime;
    }
  }

  if (auto ret = func_get_node_chunk_and_offset(id); ret != 0) {
    return ret;
  }

  // Write vector
  size_t size = node_chunk->write(chunk_offset, vec, vector_size());
  if (ailego_unlikely(size != vector_size())) {
    LOG_ERROR("Chunk write vec failed, ret=%zu", size);
    return IndexError_WriteData;
  }
  // Write key
  size = node_chunk->write(chunk_offset + vector_size(), &key, sizeof(key_t));
  if (ailego_unlikely(size != sizeof(key_t))) {
    LOG_ERROR("Chunk write key failed, ret=%zu", size);
    return IndexError_WriteData;
  }

  if (*mutable_doc_cnt() <= id) {
    *mutable_doc_cnt() = id + 1;
    chunk_offset += node_size();
    if (ailego_unlikely(node_chunk->resize(chunk_offset) != chunk_offset)) {
      LOG_ERROR("Chunk resize to %zu failed", chunk_offset);
      return IndexError_Runtime;
    }
  }

  if (use_key_info_map_) {
    keys_map_lock_->lock();
    (*keys_map_)[key] = id;
    keys_map_lock_->unlock();
  }

  broker_->mark_dirty();
  return 0;
}

int VamanaStreamerEntity::update_neighbors(
    node_id_t id,
    const std::vector<std::pair<node_id_t, dist_t>> &neighbors) {
  auto loc = get_neighbor_chunk_loc(id);
  uint32_t count =
      std::min(static_cast<uint32_t>(neighbors.size()),
               static_cast<uint32_t>(max_degree()));

  // Build neighbor data in a local buffer
  size_t nbr_size = neighbors_size();
  std::vector<uint8_t> buffer(nbr_size, 0);
  auto *hd = reinterpret_cast<NeighborsHeader *>(buffer.data());
  hd->neighbor_cnt = count;
  for (uint32_t i = 0; i < count; ++i) {
    hd->neighbors[i] = neighbors[i].first;
  }

  size_t ret = loc.first->write(loc.second, buffer.data(), nbr_size);
  if (ailego_unlikely(ret != nbr_size)) {
    LOG_ERROR("Write neighbors failed, ret=%zu", ret);
    return IndexError_WriteData;
  }
  return 0;
}

void VamanaStreamerEntity::add_neighbor(node_id_t id, uint32_t size,
                                       node_id_t neighbor_id) {
  auto loc = get_neighbor_chunk_loc(id);
  if (size >= max_degree()) return;

  // Read current neighbors
  IndexStorage::MemoryBlock mem_block;
  size_t ret = loc.first->read(loc.second, mem_block, neighbor_size_);
  if (ailego_unlikely(ret != neighbor_size_)) {
    LOG_ERROR("Read neighbor header failed, ret=%zu", ret);
    return;
  }

  // Copy to mutable buffer, update, and write back
  std::vector<uint8_t> buffer(neighbor_size_);
  memcpy(buffer.data(), mem_block.data(), neighbor_size_);
  auto *hd = reinterpret_cast<NeighborsHeader *>(buffer.data());
  hd->neighbors[size] = neighbor_id;
  hd->neighbor_cnt = size + 1;

  ret = loc.first->write(loc.second, buffer.data(), neighbor_size_);
  if (ailego_unlikely(ret != neighbor_size_)) {
    LOG_ERROR("Write neighbor failed, ret=%zu", ret);
  }
}

void VamanaStreamerEntity::update_entry_point(node_id_t ep) {
  VamanaEntity::update_entry_point(ep);
  flush_header();
}

int VamanaStreamerEntity::open(IndexStorage::Pointer stg,
                               uint64_t max_index_size, bool check_crc) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool huge_page = stg->isHugePage();
  int ret = init_chunk_params(max_index_size, huge_page);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("init_chunk_params failed: %s", IndexError::What(ret));
    return ret;
  }

  ret = broker_->open(std::move(stg), max_index_size_, chunk_size_, check_crc);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Open index failed: %s", IndexError::What(ret));
    return ret;
  }

  // Init header
  auto header_chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                                         ChunkBroker::kDefaultChunkSeqId);
  if (!header_chunk) {
    // Open empty index, create header
    auto p =
        broker_->alloc_chunk(ChunkBroker::CHUNK_TYPE_HEADER,
                             ChunkBroker::kDefaultChunkSeqId, header_size());
    if (ailego_unlikely(p.first != 0)) {
      LOG_ERROR("Alloc header chunk failed");
      return p.first;
    }
    size_t size = p.second->write(0UL, &header(), header_size());
    if (ailego_unlikely(size != header_size())) {
      LOG_ERROR("Write header chunk failed");
      return IndexError_WriteData;
    }
    return 0;
  }

  // Open existing index
  ret = init_chunks(header_chunk);
  if (ailego_unlikely(ret != 0)) return ret;

  // Verify total docs
  node_id_t total_vecs = 0;
  if (!node_chunks_.empty()) {
    size_t last_idx = node_chunks_.size() - 1;
    if (node_chunks_[last_idx]->data_size() % node_size()) {
      LOG_WARN("The index may be broken");
      return IndexError_InvalidFormat;
    }
    total_vecs = last_idx * node_cnt_per_chunk_ +
                 node_chunks_[last_idx]->data_size() / node_size();
  }

  LOG_INFO("Open Vamana index, maxDegree=%zu docCnt=%u totalVecs=%u",
           max_degree(), doc_cnt(), total_vecs);

  if (doc_cnt() != total_vecs) {
    LOG_WARN("Index closed abnormally, using totalVecs as curDocCnt");
    *mutable_doc_cnt() = total_vecs;
  }

  // Rebuild key map
  if (use_key_info_map_) {
    for (node_id_t i = 0; i < doc_cnt(); ++i) {
      key_t k = get_key(i);
      if (k != kInvalidKey) {
        (*keys_map_)[k] = i;
      }
    }
  }

  stats_.set_loaded_count(doc_cnt());
  return 0;
}

int VamanaStreamerEntity::init_chunks(const Chunk::Pointer &header_chunk) {
  // Read header from chunk
  const void *hd_ptr = nullptr;
  size_t ret = header_chunk->read(0UL, &hd_ptr, header_size());
  if (ailego_unlikely(ret != header_size())) {
    LOG_ERROR("Read header chunk failed");
    return IndexError_ReadData;
  }
  auto *hd = reinterpret_cast<const VamanaHeader *>(hd_ptr);

  // Validate
  if (vector_size() != hd->vector_size()) {
    LOG_ERROR("vector size %zu mismatch index previous %zu", vector_size(),
              hd->vector_size());
    return IndexError_Mismatch;
  }
  if (max_degree() != hd->max_degree()) {
    LOG_ERROR("max_degree %zu mismatch index previous %zu", max_degree(),
              hd->max_degree());
    return IndexError_Mismatch;
  }

  *mutable_header() = *hd;

  // Load node chunks
  size_t chunk_cnt = broker_->get_chunk_cnt(ChunkBroker::CHUNK_TYPE_NODE);
  for (size_t i = 0; i < chunk_cnt; ++i) {
    auto chunk = broker_->get_chunk(ChunkBroker::CHUNK_TYPE_NODE, i);
    if (ailego_unlikely(!chunk)) {
      LOG_ERROR("Get node chunk %zu failed", i);
      return IndexError_ReadData;
    }
    node_chunks_.emplace_back(std::move(chunk));
  }

  return 0;
}

int VamanaStreamerEntity::close() {
  LOG_DEBUG("close Vamana index");
  std::lock_guard<std::mutex> lock(mutex_);
  flush_header();
  mutable_header()->reset();
  keys_map_->clear();
  header_.clear();
  node_chunks_.clear();
  return broker_->close();
}

int VamanaStreamerEntity::flush(uint64_t checkpoint) {
  LOG_INFO("Flush Vamana index, curDocs=%u", doc_cnt());
  std::lock_guard<std::mutex> lock(mutex_);
  flush_header();
  return broker_->flush(checkpoint);
}

int VamanaStreamerEntity::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("Dump Vamana index, curDocs=%u", doc_cnt());

  std::vector<key_t> keys(doc_cnt());
  auto ret = dump_segments(dumper, keys.data());
  if (ailego_unlikely(ret < 0)) {
    return static_cast<int>(ret);
  }
  *stats_.mutable_dumped_size() += ret;
  return 0;
}

const VamanaEntity::Pointer VamanaStreamerEntity::clone() const {
  std::vector<Chunk::Pointer> cloned_chunks;
  cloned_chunks.reserve(node_chunks_.size());
  for (size_t i = 0; i < node_chunks_.size(); ++i) {
    cloned_chunks.emplace_back(node_chunks_[i]->clone());
    if (ailego_unlikely(!cloned_chunks[i])) {
      LOG_ERROR("VamanaStreamerEntity get chunk failed in clone");
      return VamanaEntity::Pointer();
    }
  }

  auto *entity = new (std::nothrow) VamanaStreamerEntity(
      stats_, header(), chunk_size_, node_index_mask_bits_,
      get_vector_enabled_, use_key_info_map_, keys_map_lock_, keys_map_,
      std::move(cloned_chunks), broker_);
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("VamanaStreamerEntity new failed");
  }
  return VamanaEntity::Pointer(entity);
}

// ============================================================================
// VamanaContiguousStreamerEntity implementation
// ============================================================================

char *VamanaContiguousStreamerEntity::allocate_contiguous(size_t size) {
  if (size == 0) return nullptr;
#if defined(__linux__)
  void *ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    LOG_ERROR("mmap failed for contiguous memory, size=%zu", size);
    return nullptr;
  }
  ::madvise(ptr, size, MADV_HUGEPAGE);
  return static_cast<char *>(ptr);
#elif defined(__APPLE__)
  void *ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
  if (ptr == MAP_FAILED) {
    LOG_ERROR("mmap failed for contiguous memory, size=%zu", size);
    return nullptr;
  }
  return static_cast<char *>(ptr);
#else
  void *ptr = std::aligned_alloc(ailego::MemoryHelper::PageSize(), size);
  if (!ptr) {
    LOG_ERROR("aligned_alloc failed, size=%zu", size);
    return nullptr;
  }
  return static_cast<char *>(ptr);
#endif
}

void VamanaContiguousStreamerEntity::release_contiguous_memory() {
  if (node_base_) {
#if defined(__linux__) || defined(__APPLE__)
    ::munmap(node_base_, node_memory_size_);
#else
    std::free(node_base_);
#endif
    node_base_ = nullptr;
    node_memory_size_ = 0;
  }
}

int VamanaContiguousStreamerEntity::build_contiguous_memory() {
  release_contiguous_memory();

  const uint32_t total_docs = doc_cnt();
  if (total_docs == 0) return 0;

  const size_t per_node = node_size();
  const size_t total_node_data = static_cast<size_t>(total_docs) * per_node;
  node_memory_size_ = AlignHugePageSize(total_node_data);
  node_base_ = allocate_contiguous(node_memory_size_);
  if (!node_base_) return IndexError_Runtime;

  // Copy node data from chunks into contiguous memory
  const auto &chunks = node_chunks();
  const uint32_t nodes_per_chunk = 1U << node_index_mask_bits();
  for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
    const void *chunk_data = nullptr;
    size_t data_size = chunks[chunk_idx]->data_size();
    chunks[chunk_idx]->read(0, &chunk_data, data_size);

    uint32_t base_id = chunk_idx * nodes_per_chunk;
    uint32_t count_in_chunk =
        std::min(nodes_per_chunk, total_docs - base_id);

    const char *src = static_cast<const char *>(chunk_data);
    char *dst = node_base_ + static_cast<size_t>(base_id) * per_node;
    std::memcpy(dst, src, static_cast<size_t>(count_in_chunk) * per_node);
  }

  LOG_INFO(
      "Built Vamana contiguous memory: node_size=%zu total_docs=%u "
      "node_chunks=%zu",
      node_memory_size_, total_docs, chunks.size());

  return 0;
}

}  // namespace core
}  // namespace zvec
