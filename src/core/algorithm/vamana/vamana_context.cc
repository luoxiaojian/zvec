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
#include "vamana_context.h"
#include <algorithm>
#include <random>
#include <zvec/core/interface/constants.h>
#include "vamana_params.h"

namespace zvec {
namespace core {

VamanaContext::VamanaContext(size_t dimension,
                             const IndexMetric::Pointer &metric,
                             const VamanaEntity::Pointer &entity)
    : IndexContext(metric),
      entity_(entity),
      dc_(entity.get(), metric, dimension),
      metric_(metric),
      stored_query_build_enabled_(metric &&
                                  metric->use_stored_query_for_build()) {
  if (metric) {
    build_distance_offset_ = metric->build_distance_offset();
  }
}

VamanaContext::VamanaContext(const IndexMetric::Pointer &metric,
                             const VamanaEntity::Pointer &entity)
    : IndexContext(metric),
      entity_(entity),
      dc_(entity.get(), metric),
      metric_(metric),
      stored_query_build_enabled_(metric &&
                                  metric->use_stored_query_for_build()) {
  if (metric) {
    build_distance_offset_ = metric->build_distance_offset();
  }
}

VamanaContext::~VamanaContext() {
  visit_filter_.destroy();
}

int VamanaContext::init(ContextType type) {
  return init(type, entity_->doc_cnt());
}

int VamanaContext::init(ContextType type, uint32_t streamer_doc_cnt) {
  int ret;
  uint32_t doc_cnt;

  type_ = type;
  results_.resize(1);
  topk_heap_.limit(std::max(topk_, ef_));
  update_heap_.limit(entity_->max_degree());

  switch (type) {
    case kBuilderContext:
      ret = visit_filter_.init(VisitFilter::ByteMap, entity_->doc_cnt(),
                               max_scan_num_, filter_negative_prob_);
      if (ret != 0) {
        LOG_ERROR("Create visit filter failed, mode %d", filter_mode_);
        return ret;
      }
      candidates_.limit(max_scan_num_);
      break;

    case kSearcherContext:
      ret = visit_filter_.init(filter_mode_, entity_->doc_cnt(), max_scan_num_,
                               filter_negative_prob_);
      if (ret != 0) {
        LOG_ERROR("Create visit filter failed, mode %d", filter_mode_);
        return ret;
      }
      candidates_.limit(max_scan_num_);
      break;

    case kStreamerContext:
      doc_cnt = streamer_doc_cnt;
      max_scan_num_ = compute_max_scan_num(doc_cnt);
      reserve_max_doc_cnt_ = doc_cnt + compute_reserve_cnt(doc_cnt);
      ret = visit_filter_.init(filter_mode_, reserve_max_doc_cnt_,
                               max_scan_num_, filter_negative_prob_);
      if (ret != 0) {
        LOG_ERROR("Create visit filter failed, mode %d", filter_mode_);
        return ret;
      }
      candidates_.limit(max_scan_num_);
      check_need_adjuct_ctx(doc_cnt);
      break;

    default:
      break;
  }

  return 0;
}

int VamanaContext::update_context(ContextType type, const IndexMeta &meta,
                                  const IndexMetric::Pointer &metric,
                                  const VamanaEntity::Pointer &entity,
                                  uint32_t magic_num) {
  if (magic_ == magic_num) {
    return 0;
  }
  type_ = type;
  entity_ = entity;
  metric_ = metric;
  stored_query_build_enabled_ =
      metric && metric->use_stored_query_for_build();
  magic_ = magic_num;
  if (metric) {
    build_distance_offset_ = metric->build_distance_offset();
  }
  dc_.update(entity.get(), metric, meta.dimension());
  return 0;
}

int VamanaContext::update(const ailego::Params &params) {
  uint32_t ef = ef_;
  params.get(PARAM_VAMANA_STREAMER_EF, &ef);
  ef_ = ef;
  topk_heap_.limit(std::max(topk_, ef_));
  uint32_t po = po_;
  params.get(PARAM_VAMANA_STREAMER_PO, &po);
  uint32_t pl = pl_;
  params.get(PARAM_VAMANA_STREAMER_PL, &pl);
  const auto resolved = resolve_query_prefetch(
      entity_->vector_data_size(), static_cast<uint32_t>(entity_->max_degree()),
      po, pl);
  po_ = resolved.first;
  pl_ = resolved.second;
  return 0;
}

std::pair<uint32_t, uint32_t> VamanaContext::resolve_query_prefetch(
    size_t vector_data_size, uint32_t max_degree, uint32_t requested_offset,
    uint32_t requested_lines) {
  using namespace core_interface;

  if (vector_data_size == 0 || max_degree == 0) {
    return {0U, 0U};
  }

  const uint32_t body_lines = static_cast<uint32_t>(
      (vector_data_size + kVamanaQueryPrefetchCacheLineBytes - 1) /
      kVamanaQueryPrefetchCacheLineBytes);

  uint32_t resolved_lines;
  if (requested_lines == kVamanaQueryPrefetchAuto) {
    resolved_lines = std::min(kVamanaQueryPrefetchTargetLines, body_lines);
  } else if (requested_lines == 0) {
    // Preserve the established explicit-zero behavior: prefetch the complete
    // stored graph vector (the refine payload is not part of vector_data_size).
    resolved_lines = body_lines;
  } else {
    resolved_lines =
        std::min({requested_lines, body_lines, kVamanaQueryPrefetchMaxValue});
  }

  uint32_t resolved_offset;
  if (requested_offset == kVamanaQueryPrefetchAuto) {
    const uint64_t bytes_per_neighbor =
        static_cast<uint64_t>(kVamanaQueryPrefetchCacheLineBytes) *
        resolved_lines;
    const uint32_t budget_offset = static_cast<uint32_t>(std::max<uint64_t>(
        1, kVamanaQueryPrefetchBudgetBytes / bytes_per_neighbor));
    resolved_offset =
        std::min({budget_offset, max_degree, kVamanaQueryPrefetchMaxValue});
  } else {
    resolved_offset =
        std::min({requested_offset, max_degree, kVamanaQueryPrefetchMaxValue});
  }

  return {resolved_offset, resolved_lines};
}

void VamanaContext::topk_to_result(uint32_t idx) {
  if (force_padding_topk_ && !topk_heap_.full() &&
      topk_heap_.size() < entity_->doc_cnt()) {
    this->fill_random_to_topk_full();
  }
  if (ailego_unlikely(topk_heap_.size() == 0)) {
    return;
  }

  ailego_assert_with(idx < results_.size(), "invalid idx");
  int size = std::min(topk_, static_cast<uint32_t>(topk_heap_.size()));
  topk_heap_.sort();
  results_[idx].clear();

  for (int i = 0; i < size; ++i) {
    auto score = topk_heap_[i].second;
    if (score > this->threshold()) {
      break;
    }
    node_id_t id = topk_heap_[i].first;
    if (fetch_vector_) {
      results_[idx].emplace_back(entity_->get_key(id), score, id,
                                 entity_->get_vector(id));
    } else {
      results_[idx].emplace_back(entity_->get_key(id), score, id);
    }
  }
}

void VamanaContext::fill_random_to_topk_full() {
  std::mt19937 rng(42);
  uint32_t doc_cnt = entity_->doc_cnt();
  uint32_t max_attempts = doc_cnt * 2;
  uint32_t attempts = 0;
  while (!topk_heap_.full() && doc_cnt > 0 && attempts < max_attempts) {
    node_id_t random_id = rng() % doc_cnt;
    if (entity_->get_key(random_id) != kInvalidKey) {
      dist_t random_dist = dc_.batch_dist(random_id);
      topk_heap_.emplace_back(random_id, random_dist);
    }
    ++attempts;
  }
}

}  // namespace core
}  // namespace zvec
