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

#include <type_traits>
#include <variant>
#include "hnsw_context.h"
#include "hnsw_streamer_entity.h"

namespace zvec {
namespace core {

// ============================================================================
// dual_heap_search_neighbors: shared core for the fallback dual-heap path.
//
// Maintains a candidate min-heap + topk heap + VisitFilter.  Supports
// arbitrary levels, filters, and MemoryBlock types (BufferPool/Mmap).
// Also updates entry_point/dist for next-level continuation.
// ============================================================================
template <typename EntityType, typename MemBlockType, typename FilterFn>
void dual_heap_search_neighbors(const EntityType &entity, level_t level,
                                node_id_t *entry_point, dist_t *dist,
                                TopkHeap &topk, HnswContext *ctx,
                                HnswDistCalculator &dc, FilterFn &&filter,
                                std::mutex *node_locks, uint32_t lock_mask) {
  const uint32_t prefetch_offset = ctx->po();
  const size_t vector_lines = ailego_cache_line_count(entity.vector_size());
  const uint32_t prefetch_lines =
      ctx->pl() > 0 ? std::min<size_t>(ctx->pl(), vector_lines) : vector_lines;

  uint32_t buf_capacity = entity.max_degree(level);
  std::vector<node_id_t> neighbor_ids(buf_capacity);
  std::vector<MemBlockType> neighbor_vec_blocks;
  neighbor_vec_blocks.reserve(buf_capacity);
  std::vector<IndexStorage::MemoryBlock> provider_vec_blocks;
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);
  const bool has_extra_values = ctx->has_extra_values();
  std::vector<const void *> neighbor_extra_values(
      has_extra_values ? buf_capacity : 0);

  const bool use_provider = dc.has_provider();
  if (ailego_unlikely(use_provider)) {
    provider_vec_blocks.reserve(buf_capacity);
  }

  VisitFilter &visit = ctx->visit_filter();
  CandidateHeap &candidates = ctx->candidates();

  candidates.clear();
  visit.clear();
  visit.set_visited(*entry_point);
  if (!filter(*entry_point)) {
    topk.emplace(*entry_point, *dist);
  }

  candidates.emplace(*entry_point, *dist);
  while (!candidates.empty() && !ctx->reach_scan_limit()) {
    auto top = candidates.begin();
    node_id_t main_node = top->first;
    dist_t main_dist = top->second;

    if (topk.full() && main_dist > topk[0].second) {
      break;
    }

    candidates.pop();
    uint32_t size = 0;
    {
      std::lock_guard<std::mutex> node_lock(node_locks[main_node & lock_mask]);
      const auto neighbors = entity.get_neighbors_typed(level, main_node);
      ailego_prefetch(neighbors.data);
      if (ailego_unlikely(ctx->debugging())) {
        (*ctx->mutable_stats_get_neighbors())++;
      }

      if (neighbors.size() > buf_capacity) {
        buf_capacity = neighbors.size();
        neighbor_ids.resize(buf_capacity);
        neighbor_vec_blocks.resize(buf_capacity);
        dists.resize(buf_capacity);
        neighbor_vecs.resize(buf_capacity);
        if (has_extra_values) {
          neighbor_extra_values.resize(buf_capacity);
        }
      }

      for (uint32_t i = 0; i < neighbors.size(); ++i) {
        node_id_t node = neighbors[i];
        if (visit.visited(node)) {
          if (ailego_unlikely(ctx->debugging())) {
            (*ctx->mutable_stats_visit_dup_cnt())++;
          }
          continue;
        }
        visit.set_visited(node);
        neighbor_ids[size++] = node;
      }
    }
    if (size == 0) {
      continue;
    }

    neighbor_vec_blocks.clear();
    provider_vec_blocks.clear();
    int ret;
    if (ailego_unlikely(use_provider)) {
      ret = dc.get_vector(neighbor_ids.data(), size, provider_vec_blocks);
    } else {
      ret = entity.get_vector_typed(neighbor_ids.data(), size,
                                    neighbor_vec_blocks);
    }
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_vector())++;
    }
    if (ailego_unlikely(ret != 0)) {
      break;
    }

    if (ailego_unlikely(use_provider)) {
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = provider_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
    } else {
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = neighbor_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
    }

    // do prefetch
    for (uint32_t i = 0; i < std::min(prefetch_offset, size); ++i) {
      if (ailego_likely(neighbor_vecs[i] != nullptr)) {
        ailego_prefetch_lines(neighbor_vecs[i], prefetch_lines);
      }
    }

    dc.batch_dist(neighbor_vecs.data(), size, dists.data(),
                  has_extra_values ? neighbor_extra_values.data() : nullptr);
    if (ailego_unlikely(dc.error())) return;

    for (uint32_t i = 0; i < size; ++i) {
      node_id_t node = neighbor_ids[i];
      dist_t cur_dist = dists[i];

      if ((!topk.full()) || cur_dist < topk[0].second) {
        candidates.emplace(node, cur_dist);
        // update entry_point for next level scan
        if (cur_dist < *dist) {
          *entry_point = node;
          *dist = cur_dist;
        }
        if (!filter(node)) {
          topk.emplace(node, cur_dist);
        }
      }
    }
  }
}

// A borrowed predicate with two entry points. Other searches filter by index
// key. Identity-key HNSW graphs can bind the concrete node predicate once for
// their entire level-zero traversal.
class HnswNodeFilter {
 public:
  using Entity = std::variant<const HnswMmapStreamerEntity *,
                              const HnswBufferPoolStreamerEntity *,
                              const HnswContiguousStreamerEntity *,
                              const HnswExternalStreamerEntity *>;

  struct Operations {
    bool (*is_filtered)(const void *, uint64_t);
    void (*search)(const void *, Entity, node_id_t *, dist_t *, TopkHeap &,
                   HnswContext *, std::mutex *, uint32_t);
  };

  HnswNodeFilter(const void *state, const Operations *ops)
      : state_(state), ops_(ops) {}

  bool operator()(uint64_t key) const {
    return ops_->is_filtered(state_, key);
  }

  void search(Entity entity, node_id_t *entry_point, dist_t *dist,
              TopkHeap &topk, HnswContext *ctx, std::mutex *locks,
              uint32_t lock_mask) const {
    ops_->search(state_, entity, entry_point, dist, topk, ctx, locks, lock_mask);
  }

 private:
  const void *state_;
  const Operations *ops_;
};

// The caller owns the state through the synchronous search, including any
// fallback that needs the original index key.
template <typename State>
HnswNodeFilter make_hnsw_node_filter(State &state) {
  static const HnswNodeFilter::Operations ops{
      [](const void *p, uint64_t key) {
        return static_cast<const State *>(p)->is_filtered(key);
      },
      [](const void *p, HnswNodeFilter::Entity source, node_id_t *entry_point,
         dist_t *dist, TopkHeap &topk, HnswContext *ctx, std::mutex *locks,
         uint32_t lock_mask) {
        const auto &state = *static_cast<const State *>(p);
        auto filter = [&](node_id_t node) { return state.contains(node); };
        std::visit(
            [&](const auto *entity) {
              using Entity = std::decay_t<decltype(*entity)>;
              dual_heap_search_neighbors<Entity, typename Entity::MemoryBlock>(
                  *entity, 0, entry_point, dist, topk, ctx,
                  ctx->dist_calculator(), filter, locks, lock_mask);
            },
            source);
      }};
  return {&state, &ops};
}

}  // namespace core
}  // namespace zvec
