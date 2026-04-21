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

#include "vamana_algorithm.h"

namespace zvec {
namespace core {

// ============================================================================
// add_node: Insert a new node into the Vamana graph.
//
// Algorithm (from DiskANN / Vamana paper):
//   1. GreedySearch from entry_point to find candidate neighbors
//   2. RobustPrune to select diverse neighbors for the new node
//   3. Update the new node's neighbor list
//   4. For each new neighbor, add reverse link; if over-degree, RobustPrune
//   5. If this is the first node, set it as entry point
// ============================================================================
template <typename EntityType>
int VamanaAlgorithm<EntityType>::add_node(node_id_t id, VamanaContext *ctx) {
  spin_lock_.lock();
  auto entry_point = entity_.entry_point();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    entity_.update_entry_point(id);
    spin_lock_.unlock();
    return 0;
  }
  spin_lock_.unlock();

  // Step 1: GreedySearch to find candidate neighbors
  uint32_t search_list_size = entity_.search_list_size();
  ctx->topk_heap().clear();
  ctx->topk_heap().limit(search_list_size);
  ctx->dist_calculator().clear_compare_cnt();

  // Set query to the new node's vector
  const void *query_vec = entity_.get_vector(id);
  if (ailego_unlikely(query_vec == nullptr)) {
    LOG_ERROR("Failed to get vector for node %u", id);
    return IndexError_ReadData;
  }
  ctx->reset_query(query_vec);

  greedy_search(entry_point, search_list_size, ctx);

  // Step 2: RobustPrune to select diverse neighbors
  auto &topk_heap = ctx->topk_heap();
  auto pruned_neighbors = robust_prune(
      id, topk_heap, entity_.alpha(), entity_.max_degree(), ctx);

  // Step 3: Update the new node's neighbor list
  entity_.update_neighbors(id, pruned_neighbors);

  // Step 4: Reverse-link updates
  update_neighbors_and_reverse_links(id, pruned_neighbors, ctx);

  return 0;
}

// ============================================================================
// search: Greedy search for approximate nearest neighbors.
// ============================================================================
template <typename EntityType>
int VamanaAlgorithm<EntityType>::search(VamanaContext *ctx) const {
  spin_lock_.lock();
  auto entry_point = entity_.entry_point();
  spin_lock_.unlock();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    return 0;
  }

  auto &topk_heap = ctx->topk_heap();
  topk_heap.clear();

  uint32_t search_list_size = std::max(static_cast<size_t>(ctx->topk()), entity_.search_list_size());
  topk_heap.limit(search_list_size);

  greedy_search(entry_point, search_list_size, ctx);

  return 0;
}

// ============================================================================
// greedy_search: Beam search from entry_point.
//
// Maintains a candidate min-heap (ordered by distance) and a visited set.
// At each step, pops the closest unvisited candidate, expands its neighbors,
// and adds unvisited neighbors to both the candidate heap and the topk heap.
// Stops when the closest candidate is farther than the worst in topk, or
// when the scan limit is reached.
// ============================================================================
template <typename EntityType>
void VamanaAlgorithm<EntityType>::greedy_search(
    node_id_t entry_point, uint32_t search_list_size,
    VamanaContext *ctx) const {
  const auto &entity = entity_;
  VamanaDistCalculator &dc = ctx->dist_calculator();
  VisitFilter &visit = ctx->visit_filter();
  CandidateHeap &candidates = ctx->candidates();
  auto &topk_heap = ctx->topk_heap();

  const IndexFilter &index_filter =
      static_cast<const IndexContext *>(ctx)->filter();
  std::function<bool(node_id_t)> filter = [](node_id_t) { return false; };
  if (index_filter.is_valid()) {
    filter = [&](node_id_t id) {
      return index_filter(entity.get_key_typed(id));
    };
  }

  candidates.clear();
  visit.clear();

  // Initialize with entry point
  dist_t entry_dist = dc.dist(entry_point);
  visit.set_visited(entry_point);
  if (!filter(entry_point)) {
    topk_heap.emplace(entry_point, entry_dist);
  }
  candidates.emplace(entry_point, entry_dist);

  while (!candidates.empty() && !ctx->reach_scan_limit()) {
    auto top = candidates.begin();
    node_id_t current_node = top->first;
    dist_t current_dist = top->second;

    // Early termination: if the closest candidate is worse than the worst
    // result in a full topk heap, we won't find anything better.
    if (topk_heap.full() && current_dist > topk_heap[0].second) {
      break;
    }

    candidates.pop();

    // Expand neighbors using typed access (no virtual dispatch)
    const auto neighbors = entity.get_neighbors_typed(current_node);
    ailego_prefetch(neighbors.data);

    // Collect unvisited neighbors
    std::vector<node_id_t> neighbor_ids(neighbors.size());
    uint32_t unvisited_count = 0;
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      node_id_t node = neighbors[i];
      if (visit.visited(node)) continue;
      visit.set_visited(node);
      neighbor_ids[unvisited_count++] = node;
    }
    if (unvisited_count == 0) continue;

    // Batch fetch vectors using typed access
    std::vector<MemBlockType> neighbor_vec_blocks;
    int ret = entity.get_vector_typed(neighbor_ids.data(), unvisited_count,
                                      neighbor_vec_blocks);
    if (ailego_unlikely(ret != 0)) break;

    // Prefetch for better cache performance
    static constexpr uint32_t PREFETCH_BATCH = 12;
    static constexpr uint32_t PREFETCH_STEP = 2;
    for (uint32_t i = 0;
         i < std::min(PREFETCH_BATCH * PREFETCH_STEP, unvisited_count); ++i) {
      ailego_prefetch(neighbor_vec_blocks[i].data());
    }

    // Batch distance computation
    std::vector<float> dists(unvisited_count);
    std::vector<const void *> neighbor_vecs(unvisited_count);
    for (uint32_t i = 0; i < unvisited_count; ++i) {
      neighbor_vecs[i] = neighbor_vec_blocks[i].data();
    }
    dc.batch_dist(neighbor_vecs.data(), unvisited_count, dists.data());

    // Update candidates and topk
    for (uint32_t i = 0; i < unvisited_count; ++i) {
      node_id_t node = neighbor_ids[i];
      dist_t node_dist = dists[i];

      if (!topk_heap.full() || node_dist < topk_heap[0].second) {
        candidates.emplace(node, node_dist);
        if (!filter(node)) {
          topk_heap.emplace(node, node_dist);
        }
      }
    }
  }
}

// ============================================================================
// robust_prune: Select up to max_degree diverse neighbors from candidates.
//
// Algorithm (Vamana / DiskANN RobustPrune):
//   1. Sort candidates by distance (ascending)
//   2. Greedily select: for each candidate c (closest first),
//      add c to result if for all already-selected neighbors p:
//        alpha * dist(c, p) > dist(query, c)
//      This ensures angular diversity — neighbors are spread out.
//   3. If result is still under max_degree, fill with remaining candidates.
// ============================================================================
template <typename EntityType>
std::vector<std::pair<node_id_t, dist_t>>
VamanaAlgorithm<EntityType>::robust_prune(
    node_id_t id, TopkHeap &candidates, float alpha, uint32_t max_degree,
    VamanaContext *ctx) const {
  std::vector<std::pair<node_id_t, dist_t>> result;
  result.reserve(max_degree);

  if (candidates.size() == 0) return result;

  // Sort candidates by distance (ascending — closest first)
  candidates.sort();

  // Build sorted candidate list, excluding self
  std::vector<std::pair<node_id_t, dist_t>> sorted_candidates;
  sorted_candidates.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].first != id) {
      sorted_candidates.emplace_back(candidates[i].first,
                                     candidates[i].second);
    }
  }

  // Track which candidates are still active (not pruned)
  std::vector<bool> active(sorted_candidates.size(), true);

  VamanaDistCalculator &dc = ctx->dist_calculator();

  for (size_t i = 0; i < sorted_candidates.size() &&
                     result.size() < max_degree;
       ++i) {
    if (!active[i]) continue;

    node_id_t candidate_id = sorted_candidates[i].first;
    dist_t candidate_dist = sorted_candidates[i].second;

    // Add this candidate as a neighbor
    result.emplace_back(candidate_id, candidate_dist);

    // Prune remaining candidates that are too close to the selected one
    // (alpha-based occlusion check)
    const void *selected_vec = entity_.get_vector(candidate_id);
    if (ailego_unlikely(selected_vec == nullptr)) continue;

    for (size_t j = i + 1; j < sorted_candidates.size(); ++j) {
      if (!active[j]) continue;

      const void *other_vec = entity_.get_vector(sorted_candidates[j].first);
      if (ailego_unlikely(other_vec == nullptr)) continue;

      dist_t inter_dist = dc.dist(selected_vec, other_vec);

      // If alpha * dist(selected, other) <= dist(query, other),
      // then `other` is occluded by `selected` — prune it.
      if (alpha * inter_dist <= sorted_candidates[j].second) {
        active[j] = false;
      }
    }
  }

  return result;
}

// ============================================================================
// update_neighbors_and_reverse_links: For each new neighbor of `id`,
// add a reverse link from neighbor back to `id`. If the neighbor's degree
// exceeds max_degree, prune it using RobustPrune.
// ============================================================================
template <typename EntityType>
void VamanaAlgorithm<EntityType>::update_neighbors_and_reverse_links(
    node_id_t id,
    const std::vector<std::pair<node_id_t, dist_t>> &new_neighbors,
    VamanaContext *ctx) {
  for (const auto &[neighbor_id, dist] : new_neighbors) {
    reverse_update_neighbor(id, neighbor_id, dist, ctx);
  }
}

// ============================================================================
// reverse_update_neighbor: Add `id` as a neighbor of `neighbor_id`.
// If neighbor_id already has max_degree neighbors, collect all neighbors
// + the new one into a candidate set and RobustPrune.
// ============================================================================
template <typename EntityType>
void VamanaAlgorithm<EntityType>::reverse_update_neighbor(
    node_id_t id, node_id_t neighbor_id, dist_t dist, VamanaContext *ctx) {
  std::lock_guard<std::mutex> lock(lock_pool_[neighbor_id & kLockMask]);

  const Neighbors current_neighbors = entity_.get_neighbors(neighbor_id);
  uint32_t current_size = current_neighbors.size();
  uint32_t max_deg = entity_.max_degree();

  // Check if `id` is already a neighbor
  for (uint32_t i = 0; i < current_size; ++i) {
    if (current_neighbors[i] == id) return;
  }

  if (current_size < max_deg) {
    // Simply append
    entity_.add_neighbor(neighbor_id, current_size, id);
    return;
  }

  // Need to prune: collect current neighbors + new node into candidates
  VamanaDistCalculator &dc = ctx->dist_calculator();
  const void *neighbor_vec = entity_.get_vector(neighbor_id);
  if (ailego_unlikely(neighbor_vec == nullptr)) return;

  TopkHeap prune_candidates;
  prune_candidates.limit(max_deg + 1);

  // Add existing neighbors
  for (uint32_t i = 0; i < current_size; ++i) {
    node_id_t nbr = current_neighbors[i];
    const void *nbr_vec = entity_.get_vector(nbr);
    if (ailego_unlikely(nbr_vec == nullptr)) continue;
    dist_t nbr_dist = dc.dist(neighbor_vec, nbr_vec);
    prune_candidates.emplace(nbr, nbr_dist);
  }

  // Add the new reverse link
  prune_candidates.emplace(id, dist);

  // RobustPrune from neighbor_id's perspective
  // Temporarily set query to neighbor_vec for distance calculations
  const void *saved_query = nullptr;
  // We use dc.dist(vec_a, vec_b) directly in robust_prune, so no need
  // to change query. Just call robust_prune with neighbor_id.
  auto pruned = robust_prune(neighbor_id, prune_candidates, entity_.alpha(),
                             max_deg, ctx);

  // Update neighbor_id's neighbor list
  entity_.update_neighbors(neighbor_id, pruned);
}

// Explicit template instantiation for all entity types
template class VamanaAlgorithm<VamanaMmapStreamerEntity>;
template class VamanaAlgorithm<VamanaBufferPoolStreamerEntity>;
template class VamanaAlgorithm<VamanaContiguousStreamerEntity>;

}  // namespace core
}  // namespace zvec
