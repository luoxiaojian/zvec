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
#include <random>
#include "vamana_params.h"

namespace zvec {
namespace core {

VamanaContext::VamanaContext(size_t dimension,
                            const IndexMetric::Pointer &metric,
                            const VamanaEntity::Pointer &entity)
    : IndexContext(metric),
      entity_(entity),
      dc_(entity.get(), metric, dimension),
      metric_(metric) {}

VamanaContext::VamanaContext(const IndexMetric::Pointer &metric,
                            const VamanaEntity::Pointer &entity)
    : IndexContext(metric),
      entity_(entity),
      dc_(entity.get(), metric),
      metric_(metric) {}

VamanaContext::~VamanaContext() {}

int VamanaContext::init(ContextType type) {
  type_ = type;
  results_.resize(1);
  topk_heap_.limit(std::max(topk_, ef_));
  update_heap_.limit(entity_->max_degree());

  uint32_t max_scan_cnt = compute_max_scan_num(reserve_max_doc_cnt_);
  max_scan_num_ = max_scan_cnt;
  visit_filter_.reset(reserve_max_doc_cnt_, max_scan_cnt);
  candidates_.limit(max_scan_num_);

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
  magic_ = magic_num;
  dc_.update(entity.get(), metric, meta.dimension());
  return 0;
}

int VamanaContext::update(const ailego::Params &params) {
  uint32_t ef = ef_;
  params.get(PARAM_VAMANA_STREAMER_EF, &ef);
  ef_ = ef;
  topk_heap_.limit(std::max(topk_, ef_));
  return 0;
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
  while (!topk_heap_.full() && doc_cnt > 0) {
    node_id_t random_id = rng() % doc_cnt;
    if (entity_->get_key(random_id) != kInvalidKey) {
      dist_t random_dist = dc_.dist(random_id);
      topk_heap_.emplace_back(random_id, random_dist);
    }
  }
}

}  // namespace core
}  // namespace zvec
