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
#include "vamana_streamer.h"
#include <iostream>
#include <ailego/internal/cpu_features.h>
#include <ailego/pattern/defer.h>
#include <ailego/utility/memory_helper.h>
#include "vamana_algorithm.h"
#include "vamana_context.h"
#include "vamana_dist_calculator.h"
#include "vamana_index_provider.h"

namespace zvec {
namespace core {

VamanaStreamer::VamanaStreamer() = default;

VamanaStreamer::~VamanaStreamer() {
  if (state_ == STATE_INITED || state_ == STATE_OPENED) {
    this->cleanup();
  }
}

int VamanaStreamer::init(const IndexMeta &imeta, const ailego::Params &params) {
  meta_ = imeta;
  meta_.set_streamer("VamanaStreamer", VamanaEntity::kRevision, params);

  params.get(PARAM_VAMANA_STREAMER_MAX_INDEX_SIZE, &max_index_size_);
  params.get(PARAM_VAMANA_STREAMER_MAX_DEGREE, &max_degree_);
  params.get(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, &search_list_size_);
  params.get(PARAM_VAMANA_STREAMER_ALPHA, &alpha_);
  params.get(PARAM_VAMANA_STREAMER_MAX_OCCLUSION_SIZE, &max_occlusion_size_);
  params.get(PARAM_VAMANA_STREAMER_EF, &ef_);
  params.get(PARAM_VAMANA_STREAMER_PO, &build_prefetch_offset_);
  params.get(PARAM_VAMANA_STREAMER_PL, &build_prefetch_lines_);
  params.get(PARAM_VAMANA_STREAMER_REVERSE_PRUNE_BATCH_SIZE,
             &reverse_prune_batch_size_);
  if (reverse_prune_batch_size_ == 0 || reverse_prune_batch_size_ > 64) {
    LOG_ERROR("Vamana reverse prune batch size must be in [1, 64], got %u",
              reverse_prune_batch_size_);
    return IndexError_InvalidArgument;
  }
  if (build_prefetch_offset_ > 256 || build_prefetch_lines_ > 256) {
    LOG_ERROR(
        "Vamana build prefetch offset/lines must be in [0, 256], got "
        "offset=%u lines=%u",
        build_prefetch_offset_, build_prefetch_lines_);
    return IndexError_InvalidArgument;
  }
  params.get(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD,
             &bruteforce_threshold_);
  params.get(PARAM_VAMANA_STREAMER_MAX_SCAN_RATIO, &max_scan_ratio_);
  params.get(PARAM_VAMANA_STREAMER_MAX_SCAN_LIMIT, &max_scan_limit_);
  params.get(PARAM_VAMANA_STREAMER_MIN_SCAN_LIMIT, &min_scan_limit_);
  params.get(PARAM_VAMANA_STREAMER_CHUNK_SIZE, &chunk_size_);
  params.get(PARAM_VAMANA_STREAMER_GET_VECTOR_ENABLE, &get_vector_enabled_);
  params.get(PARAM_VAMANA_STREAMER_FORCE_PADDING_RESULT_ENABLE,
             &force_padding_topk_enabled_);
  params.get(PARAM_VAMANA_STREAMER_USE_ID_MAP, &use_id_map_);
  params.get(PARAM_VAMANA_STREAMER_DOCS_HARD_LIMIT, &docs_hard_limit_);
  params.get(PARAM_VAMANA_STREAMER_SATURATE_GRAPH, &saturate_graph_);
  params.get(PARAM_VAMANA_STREAMER_USE_CONTIGUOUS_MEMORY,
             &use_contiguous_memory_);
  params.get(PARAM_VAMANA_STREAMER_TWO_PASS_BUILD_ENABLE,
             &two_pass_build_enabled_);
  params.get(PARAM_VAMANA_STREAMER_USE_BULK_BUILD, &use_bulk_build_);

  size_t docs_soft_limit = 0;
  params.get(PARAM_VAMANA_STREAMER_DOCS_SOFT_LIMIT, &docs_soft_limit);
  if (docs_soft_limit > 0 && docs_soft_limit > docs_hard_limit_) {
    LOG_ERROR("[%s] must be >= [%s]",
              PARAM_VAMANA_STREAMER_DOCS_HARD_LIMIT.c_str(),
              PARAM_VAMANA_STREAMER_DOCS_SOFT_LIMIT.c_str());
    return IndexError_InvalidArgument;
  } else if (docs_soft_limit == 0UL) {
    docs_soft_limit_ =
        docs_hard_limit_ * VamanaEntity::kDefaultDocsSoftLimitRatio;
  } else {
    docs_soft_limit_ = docs_soft_limit;
  }

  // Validate parameters
  if (max_degree_ == 0U) max_degree_ = VamanaEntity::kDefaultMaxDegree;
  if (search_list_size_ == 0U)
    search_list_size_ = VamanaEntity::kDefaultSearchListSize;
  if (max_occlusion_size_ == 0U)
    max_occlusion_size_ = VamanaEntity::kDefaultMaxOcclusionSize;
  if (alpha_ <= 0.0f) alpha_ = VamanaEntity::kDefaultAlpha;
  if (ef_ == 0U) ef_ = VamanaEntity::kDefaultEf;
  if (chunk_size_ == 0UL) chunk_size_ = VamanaEntity::kDefaultChunkSize;
  if (chunk_size_ > VamanaEntity::kMaxChunkSize) {
    LOG_ERROR("[%s] must be < %zu", PARAM_VAMANA_STREAMER_CHUNK_SIZE.c_str(),
              VamanaEntity::kMaxChunkSize);
    return IndexError_InvalidArgument;
  }
  if (max_scan_ratio_ <= 0.0f || max_scan_ratio_ > 1.0f) {
    LOG_ERROR("[%s] must be in range (0.0f,1.0f]",
              PARAM_VAMANA_STREAMER_MAX_SCAN_RATIO.c_str());
    return IndexError_InvalidArgument;
  }
  if (max_scan_limit_ < min_scan_limit_) {
    LOG_ERROR("[%s] must be >= [%s]",
              PARAM_VAMANA_STREAMER_MAX_SCAN_LIMIT.c_str(),
              PARAM_VAMANA_STREAMER_MIN_SCAN_LIMIT.c_str());
    return IndexError_InvalidArgument;
  }

  LOG_DEBUG(
      "Vamana init params: maxIndexSize=%zu docsHardLimit=%zu "
      "docsSoftLimit=%zu maxDegree=%u searchListSize=%u alpha=%.2f "
      "maxOcclusionSize=%u ef=%u maxScanRatio=%.3f minScanLimit=%zu "
      "maxScanLimit=%zu bruteForceThreshold=%zu chunkSize=%zu "
      "getVectorEnabled=%u forcePadding=%u twoPassBuild=%u "
      "useBulkBuild=%u "
      "reversePruneBatchSize=%u buildPrefetchOffset=%u "
      "buildPrefetchLines=%u",
      max_index_size_, docs_hard_limit_, docs_soft_limit_, max_degree_,
      search_list_size_, alpha_, max_occlusion_size_, ef_, max_scan_ratio_,
      min_scan_limit_, max_scan_limit_, bruteforce_threshold_, chunk_size_,
      get_vector_enabled_, force_padding_topk_enabled_, two_pass_build_enabled_,
      use_bulk_build_, reverse_prune_batch_size_, build_prefetch_offset_,
      build_prefetch_lines_);

  const float initial_build_alpha = two_pass_build_enabled_ ? 1.0f : alpha_;
  bulk_build_active_ = false;
  build_finalized_.store(false);
  stats_.mutable_attributes()->set("vamana_bulk_build_used", uint32_t{0});
  stats_.mutable_attributes()->set("vamana_bulk_fast_search_used", uint32_t{0});
  stats_.mutable_attributes()->set("vamana_use_bulk_build",
                                   static_cast<uint32_t>(use_bulk_build_));
  stats_.mutable_attributes()->set("vamana_build_prefetch_offset",
                                   build_prefetch_offset_);
  stats_.mutable_attributes()->set("vamana_build_prefetch_lines",
                                   build_prefetch_lines_);
  stats_.mutable_attributes()->set("vamana_refine_pass_count", uint32_t{0});
  stats_.mutable_attributes()->set("vamana_build_pass_count", uint32_t{1});
  stats_.mutable_attributes()->set("vamana_initial_build_alpha",
                                   initial_build_alpha);
  state_ = STATE_INITED;
  return 0;
}

int VamanaStreamer::cleanup(void) {
  if (state_ == STATE_OPENED) {
    this->close();
  }

  LOG_INFO("VamanaStreamer cleanup");

  meta_.clear();
  metric_.reset();
  stats_.clear();
  if (entity_) entity_->cleanup();
  if (alg_) alg_->cleanup();

  max_index_size_ = 0UL;
  docs_hard_limit_ = VamanaEntity::kDefaultDocsHardLimit;
  docs_soft_limit_ = 0UL;
  max_degree_ = VamanaEntity::kDefaultMaxDegree;
  search_list_size_ = VamanaEntity::kDefaultSearchListSize;
  max_occlusion_size_ = VamanaEntity::kDefaultMaxOcclusionSize;
  alpha_ = VamanaEntity::kDefaultAlpha;
  ef_ = VamanaEntity::kDefaultEf;
  build_prefetch_offset_ = VamanaEntity::kDefaultBuildPrefetchOffset;
  build_prefetch_lines_ = VamanaEntity::kDefaultBuildPrefetchLines;
  reverse_prune_batch_size_ = 1;
  bruteforce_threshold_ = VamanaEntity::kDefaultBruteForceThreshold;
  max_scan_limit_ = VamanaEntity::kDefaultMaxScanLimit;
  min_scan_limit_ = VamanaEntity::kDefaultMinScanLimit;
  chunk_size_ = VamanaEntity::kDefaultChunkSize;
  max_scan_ratio_ = VamanaEntity::kDefaultScanRatio;
  state_ = STATE_INIT;
  check_crc_enabled_ = false;
  get_vector_enabled_ = false;
  two_pass_build_enabled_ = false;
  use_bulk_build_ = true;
  bulk_build_active_ = false;
  build_finalized_.store(false);

  return 0;
}

int VamanaStreamer::setup_entity() {
  entity_->set_use_key_info_map(use_id_map_);
  entity_->set_vector_size(meta_.element_size());
  entity_->set_chunk_size(chunk_size_);
  entity_->set_get_vector(get_vector_enabled_);

  // Set Vamana-specific parameters via public setters
  entity_->set_max_degree(max_degree_);
  entity_->set_search_list_size(search_list_size_);
  entity_->set_max_occlusion_size(max_occlusion_size_);
  // Standard Vamana two-pass construction builds the graph from scratch with
  // alpha=1.0, then revisits every point once with the configured target alpha.
  // Single-pass construction continues to use the configured alpha directly.
  const float initial_build_alpha = two_pass_build_enabled_ ? 1.0f : alpha_;
  entity_->set_alpha(initial_build_alpha);
  entity_->set_saturate_graph(saturate_graph_);

  int ret = entity_->init(docs_hard_limit_);
  if (ret != 0) {
    LOG_ERROR("Vamana entity init failed: %s", IndexError::What(ret));
  }
  return ret;
}

int VamanaStreamer::open(IndexStorage::Pointer stg) {
  LOG_INFO("VamanaStreamer open");

  if (ailego_unlikely(state_ != STATE_INITED)) {
    LOG_ERROR("Open storage failed, init streamer first!");
    return IndexError_NoReady;
  }

  // Create entity based on storage type
  switch (stg->memory_block_type()) {
    case IndexStorage::MemoryBlock::MBT_BUFFERPOOL: {
      entity_ = std::make_unique<VamanaBufferPoolStreamerEntity>(stats_);
      break;
    }
    default: {
      if (use_contiguous_memory_) {
        entity_ = std::make_unique<VamanaContiguousStreamerEntity>(stats_);
      } else {
        entity_ = std::make_unique<VamanaMmapStreamerEntity>(stats_);
      }
      break;
    }
  }

  auto cleanup_on_error = [this]() {
    if (entity_) {
      entity_->close();
      entity_.reset();
    }
    alg_.reset();
    metric_.reset();
  };

  int ret = setup_entity();
  if (ret != 0) {
    cleanup_on_error();
    return ret;
  }

  ret = entity_->open(std::move(stg), max_index_size_, check_crc_enabled_);
  if (ret != 0) {
    cleanup_on_error();
    return ret;
  }
  if (entity_->doc_cnt() == 0) {
    LOG_INFO("Vamana initial graph build configured with alpha=%.2f%s",
             entity_->alpha(),
             two_pass_build_enabled_ ? " (two-pass first pass)" : "");
  }

  // Handle IndexMeta
  IndexMeta index_meta;
  ret = entity_->get_index_meta(&index_meta);
  if (ret == IndexError_NoExist) {
    ret = entity_->set_index_meta(meta_);
    if (ret != 0) {
      LOG_ERROR("Failed to set index meta: %s", IndexError::What(ret));
      cleanup_on_error();
      return ret;
    }
  } else if (ret != 0) {
    LOG_ERROR("Failed to get index meta: %s", IndexError::What(ret));
    cleanup_on_error();
    return ret;
  } else {
    if (index_meta.dimension() != meta_.dimension() ||
        index_meta.element_size() != meta_.element_size() ||
        index_meta.metric_name() != meta_.metric_name() ||
        index_meta.data_type() != meta_.data_type()) {
      LOG_ERROR("IndexMeta mismatch from the previous in index");
      cleanup_on_error();
      return IndexError_Mismatch;
    }
    auto metric_params = index_meta.metric_params();
    metric_params.merge(meta_.metric_params());
    meta_.set_metric(index_meta.metric_name(), 0, metric_params);
    // Propagate reformer info from stored meta (needed for quantizers
    // whose reformer params are computed during training, e.g. UniformInt8)
    if (!index_meta.reformer_name().empty()) {
      meta_.set_reformer(index_meta.reformer_name(), 0,
                         index_meta.reformer_params());
    }
  }

  // Create metric
  metric_ = IndexFactory::CreateMetric(meta_.metric_name());
  if (!metric_) {
    LOG_ERROR("Failed to create metric %s", meta_.metric_name().c_str());
    cleanup_on_error();
    return IndexError_NoExist;
  }
  ret = metric_->init(meta_, meta_.metric_params());
  if (ret != 0) {
    LOG_ERROR("Failed to init metric, ret=%d", ret);
    cleanup_on_error();
    return ret;
  }
  if (!metric_->distance() || !metric_->batch_distance()) {
    LOG_ERROR("Invalid metric distance functions");
    cleanup_on_error();
    return IndexError_InvalidArgument;
  }
  add_distance_ = metric_->distance();
  add_batch_distance_ = metric_->batch_distance();
  add_stored_batch_distance_ = metric_->stored_batch_distance();
  size_t extra_values_size = metric_->extra_values_size_per_vector();
  search_distance_ = add_distance_;
  search_batch_distance_ = add_batch_distance_;

  auto query_metric = metric_->query_metric();
  if (query_metric && query_metric->distance() &&
      query_metric->batch_distance()) {
    search_distance_ = query_metric->distance();
    search_batch_distance_ = query_metric->batch_distance();
    extra_values_size = std::max(extra_values_size,
                                 query_metric->extra_values_size_per_vector());
  }
  entity_->set_extra_values_size(extra_values_size);

  // Create algorithm based on entity storage mode
  switch (entity_->storage_mode()) {
    case VamanaStorageMode::kBufferPool:
      alg_ = VamanaAlgorithmBase::UPointer(
          new VamanaAlgorithm<VamanaBufferPoolStreamerEntity>(
              static_cast<VamanaBufferPoolStreamerEntity &>(*entity_)));
      break;
    case VamanaStorageMode::kContiguous: {
      auto &contiguous_entity =
          static_cast<VamanaContiguousStreamerEntity &>(*entity_);
      int build_ret = contiguous_entity.build_contiguous_memory();
      if (build_ret != 0) {
        LOG_ERROR("Failed to build contiguous memory, ret=%d", build_ret);
        cleanup_on_error();
        return build_ret;
      }
      alg_ = VamanaAlgorithmBase::UPointer(
          new VamanaAlgorithm<VamanaContiguousStreamerEntity>(
              contiguous_entity));
      break;
    }
    default:
      alg_ = VamanaAlgorithmBase::UPointer(
          new VamanaAlgorithm<VamanaMmapStreamerEntity>(
              static_cast<VamanaMmapStreamerEntity &>(*entity_)));
      break;
  }

  ret = alg_->init();
  if (ret != 0) {
    cleanup_on_error();
    return ret;
  }

  state_ = STATE_OPENED;
  magic_ = IndexContext::GenerateMagic();

  return 0;
}

int VamanaStreamer::close(void) {
  LOG_INFO("VamanaStreamer close");

  stats_.clear();
  meta_.set_metric(metric_->name(), 0, metric_->params());
  entity_->set_index_meta(meta_);
  int ret = entity_->close();
  if (ret != 0) return ret;
  state_ = STATE_INITED;
  return 0;
}

int VamanaStreamer::flush(uint64_t checkpoint) {
  LOG_INFO("VamanaStreamer flush checkpoint=%zu", (size_t)checkpoint);

  meta_.set_metric(metric_->name(), 0, metric_->params());
  entity_->set_index_meta(meta_);
  return entity_->flush(checkpoint);
}

void VamanaStreamer::merge_trained_meta(const IndexMeta &trained_meta) {
  if (!trained_meta.reformer_name().empty()) {
    meta_.set_reformer(trained_meta.reformer_name(),
                       trained_meta.reformer_revision(),
                       trained_meta.reformer_params());
  }
  if (!trained_meta.converter_name().empty()) {
    meta_.set_converter(trained_meta.converter_name(),
                        trained_meta.converter_revision(),
                        trained_meta.converter_params());
  }
}

int VamanaStreamer::dump(const IndexDumper::Pointer &dumper) {
  LOG_INFO("VamanaStreamer dump");

  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });

  const bool regular_one_pass_build = !two_pass_build_enabled_ &&
                                      !bulk_build_active_ &&
                                      !build_finalized_.load();
  int ret = finalize_build_locked(true);
  if (ret != 0) {
    LOG_ERROR("Vamana two-pass graph refine failed: %s", IndexError::What(ret));
    return ret;
  }
  if (regular_one_pass_build) {
    update_entry_point_to_medoid();
  }

  meta_.set_searcher("VamanaSearcher", VamanaEntity::kRevision,
                     ailego::Params());

  ret = IndexHelper::SerializeToDumper(meta_, dumper.get());
  if (ret != 0) {
    LOG_ERROR("Failed to serialize meta into dumper.");
    return ret;
  }

  return entity_->dump(dumper);
}

void VamanaStreamer::update_entry_point_to_medoid() {
  // Calculate medoid (DiskANN standard: entry point = closest to centroid).
  // At dump time, data_type and dimension are fully known from meta_.
  // UNIFORM_UINT8 excludes its int32 sum_sq tail. UNIFORM_UINT4 asks the
  // entity to decode the two packed nibbles in each physical byte.
  if (entity_->doc_cnt() > 0) {
    const bool is_uniform_uint8 = (meta_.metric_name() == "UniformUint8");
    const bool is_uniform_uint4 = (meta_.metric_name() == "UniformUint4");
    uint32_t medoid_dim = meta_.dimension();
    constexpr uint32_t kUniformUint8TailBytes = sizeof(int32_t);
    if (is_uniform_uint8 && medoid_dim > kUniformUint8TailBytes) {
      medoid_dim -= kUniformUint8TailBytes;
    }
    if (is_uniform_uint4) medoid_dim *= 2U;
    node_id_t medoid = entity_->calculate_medoid(
        medoid_dim, static_cast<uint32_t>(meta_.data_type()), is_uniform_uint4);
    if (medoid != kInvalidNodeId && medoid != entity_->entry_point()) {
      LOG_INFO("Updating entry point from %u to medoid %u",
               entity_->entry_point(), medoid);
      entity_->update_entry_point(medoid);
    }
  }
}

int VamanaStreamer::begin_bulk_build() {
  if (state_ != STATE_OPENED) {
    LOG_ERROR("Cannot begin Vamana bulk build before opening the streamer");
    return IndexError_NoReady;
  }

  // begin_bulk_build() is called for every merge. Always leave an ineligible
  // target in normal streaming mode, including a second merge into an index
  // that previously used the bulk path.
  bulk_build_active_ = false;
  if (!use_bulk_build_) {
    LOG_INFO("Vamana bulk build and construction optimizations are disabled");
    stats_.mutable_attributes()->set("vamana_bulk_build_used", uint32_t{0});
    stats_.mutable_attributes()->set("vamana_bulk_fast_search_used",
                                     uint32_t{0});
    return 0;
  }
  if (entity_->storage_mode() != VamanaStorageMode::kContiguous ||
      entity_->doc_cnt() != 0) {
    return 0;
  }

  bulk_build_active_ = true;
  stats_.mutable_attributes()->set("vamana_bulk_build_used", uint32_t{1});
  stats_.mutable_attributes()->set("vamana_bulk_fast_search_used", uint32_t{1});
  LOG_INFO(
      "Vamana bulk build selected for empty contiguous merge target "
      "(two_pass=%u greedy_search=fast po=%u pl=%u)",
      two_pass_build_enabled_, build_prefetch_offset_, build_prefetch_lines_);
  return 0;
}

int VamanaStreamer::finalize_build() {
  if (!two_pass_build_enabled_ && !bulk_build_active_) {
    return 0;
  }

  shared_mutex_.lock();
  AILEGO_DEFER([&]() { shared_mutex_.unlock(); });
  return finalize_build_locked(true);
}

int VamanaStreamer::build_bulk_graph_locked() {
  if (!bulk_build_active_) {
    return 0;
  }

  auto &contiguous_entity =
      static_cast<VamanaContiguousStreamerEntity &>(*entity_);
  contiguous_entity.set_reverse_prune_batch_size(reverse_prune_batch_size_);
  int ret = contiguous_entity.build_contiguous_records_for_build();
  if (ret != 0) {
    LOG_ERROR("Failed to materialize Vamana bulk-build records, ret=%d", ret);
    return ret;
  }

  VamanaEntity::Pointer entity_ref(entity_.get(), [](VamanaEntity *) {});
  auto ctx =
      std::make_unique<VamanaContext>(meta_.dimension(), metric_, entity_ref);
  ctx->set_ef(ef_);
  ctx->set_po(build_prefetch_offset_);
  ctx->set_pl(build_prefetch_lines_);
  ctx->set_build_fast_search(true);
  ctx->set_build_optimization_enabled(true);
  ctx->set_reverse_prune_batch_size(reverse_prune_batch_size_);
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);

  // The physical entity already contains every transformed vector, but graph
  // construction must retain the same logical prefix growth as streaming
  // insertion.
  ret = ctx->init(VamanaContext::kStreamerContext, 0);
  if (ret != 0) {
    LOG_ERROR("Init Vamana bulk-build context failed");
    return ret;
  }
  ctx->update_dist_caculator_distance(add_distance_, add_batch_distance_);

  const uint32_t total_docs = entity_->doc_cnt();
  LOG_INFO(
      "Vamana bulk graph pass: docs=%u alpha=%.2f "
      "reverse_prune_batch_size=%u greedy_search=fast po=%u pl=%u",
      total_docs, entity_->alpha(), ctx->reverse_prune_batch_size(), ctx->po(),
      ctx->pl());
  for (node_id_t id = 0; id < total_docs; ++id) {
    if (entity_->get_key(id) == kInvalidKey) continue;
    ctx->clear();
    ctx->check_need_adjuct_ctx(id);
    ret = alg_->add_node(id, ctx.get());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Vamana bulk graph pass failed at node %u", id);
      return ret;
    }
    if (ailego_unlikely(ctx->error())) {
      LOG_ERROR("Vamana bulk graph distance error at node %u", id);
      return IndexError_Runtime;
    }
  }
  ret = alg_->flush_reverse_overflow(ctx.get());
  if (ret != 0) {
    LOG_ERROR("Vamana bulk Pass1 overflow flush failed, ret=%d", ret);
    return ret;
  }
  return 0;
}

int VamanaStreamer::finalize_build_locked(bool update_medoid) {
  if (build_finalized_.load()) {
    return 0;
  }
  if (!two_pass_build_enabled_ && !bulk_build_active_) {
    return 0;
  }

  // Keep bulk construction state alive through both passes. In particular,
  // pass 2 must use the same reverse-prune batch policy and temporary
  // adjacency overflow storage as pass 1. Always leave bulk mode when this
  // finalization attempt returns, so later Add/Merge calls cannot accidentally
  // defer ordinary streaming insertion.
  const bool bulk_build = bulk_build_active_;
  AILEGO_DEFER([&]() {
    if (bulk_build) {
      bulk_build_active_ = false;
    }
  });

  int ret = build_bulk_graph_locked();
  if (ret != 0) {
    return ret;
  }

  if (update_medoid) {
    update_entry_point_to_medoid();
  }

  // The final persisted graph always records the configured target alpha,
  // including the degenerate zero/one-document case.
  entity_->set_alpha(alpha_);

  uint32_t pass_count = 0;
  if (entity_->doc_cnt() <= 1) {
    stats_.mutable_attributes()->set("vamana_refine_pass_count", pass_count);
    build_finalized_.store(true);
    magic_ = IndexContext::GenerateMagic();
    return 0;
  }

  if (!two_pass_build_enabled_) {
    stats_.mutable_attributes()->set("vamana_refine_pass_count", pass_count);
    stats_.mutable_attributes()->set("vamana_build_pass_count", uint32_t{1});
    build_finalized_.store(true);
    magic_ = IndexContext::GenerateMagic();
    return 0;
  }

  VamanaEntity::Pointer entity_ref(entity_.get(), [](VamanaEntity *) {});
  auto ctx =
      std::make_unique<VamanaContext>(meta_.dimension(), metric_, entity_ref);
  ctx->set_ef(ef_);
  ctx->set_po(build_prefetch_offset_);
  ctx->set_pl(build_prefetch_lines_);
  ctx->set_build_fast_search(bulk_build);
  ctx->set_build_optimization_enabled(use_bulk_build_);
  // Construction-only overflow storage exists only in bulk contiguous mode.
  // Keep streaming and mmap construction on the immediate-prune path.
  ctx->set_reverse_prune_batch_size(bulk_build ? reverse_prune_batch_size_ : 1);
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);

  ret = ctx->init(VamanaContext::kStreamerContext);
  if (ret != 0) {
    LOG_ERROR("Init Vamana refine context failed");
    return ret;
  }

  ctx->check_need_adjuct_ctx(entity_->doc_cnt());
  ctx->update_dist_caculator_distance(
      add_distance_,
      use_bulk_build_ ? add_batch_distance_ : add_stored_batch_distance_);

  // The incremental add path above is the first pass (alpha=1.0). Reuse that
  // graph for exactly one full-graph second pass at the configured target
  // alpha. Keep the entity alpha in sync because reverse-link pruning reads it.
  LOG_INFO(
      "Vamana two-pass second graph pass: alpha=%.2f "
      "reverse_prune_batch_size=%u greedy_search=%s po=%u pl=%u",
      alpha_, ctx->reverse_prune_batch_size(),
      ctx->build_fast_search() ? "fast" : "dual", ctx->po(), ctx->pl());
  ret = alg_->refine_graph(ctx.get(), alpha_);
  if (ret != 0) return ret;
  ++pass_count;

  if (ctx->error()) {
    return IndexError_Runtime;
  }
  stats_.mutable_attributes()->set("vamana_refine_pass_count", pass_count);
  stats_.mutable_attributes()->set("vamana_build_pass_count", uint32_t{2});
  build_finalized_.store(true);
  magic_ = IndexContext::GenerateMagic();
  return 0;
}

IndexStreamer::Context::Pointer VamanaStreamer::create_context(void) const {
  if (ailego_unlikely(state_ != STATE_OPENED)) {
    LOG_ERROR("Create context failed, open storage first!");
    return Context::Pointer();
  }

  VamanaEntity::Pointer entity = entity_->clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("CreateContext clone failed");
    return Context::Pointer();
  }

  VamanaContext *ctx =
      new (std::nothrow) VamanaContext(meta_.dimension(), metric_, entity);
  if (ailego_unlikely(ctx == nullptr)) {
    LOG_ERROR("Failed to new VamanaContext");
    return Context::Pointer();
  }

  ctx->set_ef(ef_);
  ctx->set_po(build_prefetch_offset_);
  ctx->set_pl(build_prefetch_lines_);
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_magic(magic_);
  ctx->set_force_padding_topk(force_padding_topk_enabled_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);
  ctx->set_build_optimization_enabled(use_bulk_build_);
  ctx->set_build_fast_search(false);
  ctx->set_reverse_prune_batch_size(1);

  if (ailego_unlikely(ctx->init(VamanaContext::kStreamerContext) != 0)) {
    LOG_ERROR("Init VamanaContext failed");
    delete ctx;
    return Context::Pointer();
  }

  ctx->check_need_adjuct_ctx(entity_->doc_cnt());

  return Context::Pointer(ctx);
}

IndexProvider::Pointer VamanaStreamer::create_provider(void) const {
  LOG_DEBUG("VamanaStreamer create provider");

  auto entity = entity_->clone();
  if (ailego_unlikely(!entity)) {
    LOG_ERROR("Clone VamanaEntity failed");
    return nullptr;
  }
  return IndexProvider::Pointer(
      new VamanaIndexProvider(meta_, entity, "VamanaStreamer"));
}

int VamanaStreamer::update_context(VamanaContext *ctx) const {
  const VamanaEntity::Pointer entity = entity_->clone();
  if (!entity) {
    LOG_ERROR("Failed to clone search context entity");
    return IndexError_Runtime;
  }
  ctx->set_max_scan_limit(max_scan_limit_);
  ctx->set_min_scan_limit(min_scan_limit_);
  ctx->set_max_scan_ratio(max_scan_ratio_);
  ctx->set_po(build_prefetch_offset_);
  ctx->set_pl(build_prefetch_lines_);
  ctx->set_bruteforce_threshold(bruteforce_threshold_);
  ctx->set_build_optimization_enabled(use_bulk_build_);
  ctx->set_build_fast_search(false);
  ctx->set_reverse_prune_batch_size(1);
  return ctx->update_context(VamanaContext::kStreamerContext, meta_, metric_,
                             entity, magic_);
}

int VamanaStreamer::add_impl(uint64_t pkey, const void *query,
                             const IndexQueryMeta &qmeta,
                             Context::Pointer &context) {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) return ret;

  VamanaContext *ctx = dynamic_cast<VamanaContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to VamanaContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    ret = update_context(ctx);
    if (ret != 0) return ret;
  }

  if (ailego_unlikely(entity_->doc_cnt() >= docs_soft_limit_)) {
    if (entity_->doc_cnt() >= docs_hard_limit_) {
      LOG_ERROR("Current docs %u exceed hard limit", entity_->doc_cnt());
      const std::lock_guard<std::mutex> lk(mutex_);
      (*stats_.mutable_discarded_count())++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed soft limit", entity_->doc_cnt());
    }
  }

  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  if (!bulk_build_active_) {
    ctx->clear();
    ctx->update_dist_caculator_distance(
        add_distance_,
        use_bulk_build_ ? add_batch_distance_ : add_stored_batch_distance_);
    ctx->check_need_adjuct_ctx(entity_->doc_cnt());
  }

  if (metric_->support_train()) {
    const std::lock_guard<std::mutex> lk(mutex_);
    ret = metric_->train(query, meta_.dimension());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Vamana streamer metric train failed");
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
  }

  node_id_t id;
  ret = entity_->add_vector(pkey, query, &id);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Vamana streamer add vector failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (bulk_build_active_) {
    (*stats_.mutable_added_count())++;
    build_finalized_.store(false);
    return 0;
  }

  ret = alg_->add_node(id, ctx);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Vamana streamer add node failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (ailego_unlikely(ctx->error())) {
    (*stats_.mutable_discarded_count())++;
    return IndexError_Runtime;
  }
  (*stats_.mutable_added_count())++;
  build_finalized_.store(false);

  return 0;
}

int VamanaStreamer::add_with_id_impl(uint32_t id, const void *query,
                                     const IndexQueryMeta &qmeta,
                                     Context::Pointer &context) {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) return ret;

  VamanaContext *ctx = dynamic_cast<VamanaContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to VamanaContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    ret = update_context(ctx);
    if (ret != 0) return ret;
  }

  if (ailego_unlikely(entity_->doc_cnt() >= docs_soft_limit_)) {
    if (entity_->doc_cnt() >= docs_hard_limit_) {
      LOG_ERROR("Current docs %u exceed hard limit", entity_->doc_cnt());
      const std::lock_guard<std::mutex> lk(mutex_);
      (*stats_.mutable_discarded_count())++;
      return IndexError_IndexFull;
    } else {
      LOG_WARN("Current docs %u exceed soft limit", entity_->doc_cnt());
    }
  }

  if (ailego_unlikely(!shared_mutex_.try_lock_shared())) {
    LOG_ERROR("Cannot add vector while dumping index");
    (*stats_.mutable_discarded_count())++;
    return IndexError_Unsupported;
  }
  AILEGO_DEFER([&]() { shared_mutex_.unlock_shared(); });

  if (!bulk_build_active_) {
    ctx->clear();
    ctx->update_dist_caculator_distance(
        add_distance_,
        use_bulk_build_ ? add_batch_distance_ : add_stored_batch_distance_);
    ctx->check_need_adjuct_ctx(entity_->doc_cnt());
  }

  if (metric_->support_train()) {
    const std::lock_guard<std::mutex> lk(mutex_);
    ret = metric_->train(query, meta_.dimension());
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Vamana streamer metric train failed");
      (*stats_.mutable_discarded_count())++;
      return ret;
    }
  }

  ret = entity_->add_vector_with_id(id, query);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Vamana streamer add vector failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (bulk_build_active_) {
    (*stats_.mutable_added_count())++;
    build_finalized_.store(false);
    return 0;
  }

  ret = alg_->add_node(id, ctx);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Vamana streamer add node failed");
    (*stats_.mutable_discarded_count())++;
    return ret;
  }

  if (ailego_unlikely(ctx->error())) {
    (*stats_.mutable_discarded_count())++;
    return IndexError_Runtime;
  }
  (*stats_.mutable_added_count())++;
  build_finalized_.store(false);

  return 0;
}

int VamanaStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                                Context::Pointer &context) const {
  return search_impl(query, qmeta, 1, context);
}

int VamanaStreamer::search_keys_direct(const void *query,
                                       const IndexQueryMeta &qmeta,
                                       std::vector<uint64_t> *keys,
                                       Context::Pointer &context) const {
  if (keys == nullptr) return IndexError_InvalidArgument;

  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) return ret;

  VamanaContext *ctx = dynamic_cast<VamanaContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to VamanaContext failed");
    return IndexError_Cast;
  }

  // Filtered and brute-force searches have different result semantics.  The
  // buffer-pool algorithm and non-AVX2 machines do not use the direct
  // BlockHeap mmap/contiguous pool path.
  if (ctx->filter().is_valid() ||
      entity_->storage_mode() == VamanaStorageMode::kBufferPool ||
      !zvec::ailego::internal::CpuFeatures::static_flags_.AVX2 ||
      entity_->doc_cnt() <= ctx->get_bruteforce_threshold()) {
    return IndexError_NotImplemented;
  }

  if (ctx->magic() != magic_) {
    ret = update_context(ctx);
    if (ret != 0) return ret;
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(search_distance_, search_batch_distance_);
  ctx->check_need_adjuct_ctx(entity_->doc_cnt());
  ctx->reset_query(query);
  ctx->set_direct_pool_result(true);
  ret = alg_->search(ctx);
  ctx->set_direct_pool_result(false);
  if (ailego_unlikely(ret != 0)) {
    LOG_ERROR("Vamana direct candidate search failed");
    return ret;
  }
  if (ailego_unlikely(ctx->error())) return IndexError_Runtime;

  const BlockHeap &pool = ctx->block_pool();
  const size_t count = std::min(static_cast<size_t>(ctx->topk()),
                                static_cast<size_t>(pool.size()));
  keys->resize(count);
  for (size_t i = 0; i < count; ++i) {
    (*keys)[i] = entity_->get_key(pool.id(static_cast<int32_t>(i)));
  }
  return 0;
}

int VamanaStreamer::search_impl(const void *query, const IndexQueryMeta &qmeta,
                                uint32_t count,
                                Context::Pointer &context) const {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) return ret;

  VamanaContext *ctx = dynamic_cast<VamanaContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to VamanaContext failed");
    return IndexError_Cast;
  }

  if (entity_->doc_cnt() <= ctx->get_bruteforce_threshold()) {
    return search_bf_impl(query, qmeta, count, context);
  }

  if (ctx->magic() != magic_) {
    ret = update_context(ctx);
    if (ret != 0) return ret;
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(search_distance_, search_batch_distance_);
  ctx->resize_results(count);
  ctx->check_need_adjuct_ctx(entity_->doc_cnt());

  for (size_t q = 0; q < count; ++q) {
    ctx->reset_query(query);
    ret = alg_->search(ctx);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Vamana search failed");
      return ret;
    }
    ctx->topk_to_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) return IndexError_Runtime;
  return 0;
}

void VamanaStreamer::print_debug_info() {
  for (node_id_t id = 0; id < entity_->doc_cnt(); ++id) {
    if (entity_->get_key(id) == kInvalidKey) continue;
    Neighbors neighbours = entity_->get_neighbors(id);
    std::cout << "node: " << id << "; ";
    if (neighbours.size() == 0) {
      std::cout << std::endl;
      continue;
    }
    for (uint32_t i = 0; i < neighbours.size(); ++i) {
      std::cout << neighbours[i];
      if (i == neighbours.size() - 1) {
        std::cout << std::endl;
      } else {
        std::cout << ", ";
      }
    }
  }
}

int VamanaStreamer::search_bf_impl(const void *query,
                                   const IndexQueryMeta &qmeta,
                                   Context::Pointer &context) const {
  return search_bf_impl(query, qmeta, 1, context);
}

int VamanaStreamer::search_bf_impl(const void *query,
                                   const IndexQueryMeta &qmeta, uint32_t count,
                                   Context::Pointer &context) const {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) return ret;

  VamanaContext *ctx = dynamic_cast<VamanaContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to VamanaContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    ret = update_context(ctx);
    if (ret != 0) return ret;
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(search_distance_, search_batch_distance_);
  ctx->resize_results(count);

  const auto &filter = static_cast<IndexContext *>(ctx)->filter();
  auto &topk = ctx->topk_heap();

  for (size_t q = 0; q < count; ++q) {
    ctx->reset_query(query);
    topk.clear();
    for (node_id_t id = 0; id < entity_->doc_cnt(); ++id) {
      if (entity_->get_key(id) == kInvalidKey) continue;
      if (!filter.is_valid() || !filter(entity_->get_key(id))) {
        dist_t dist = ctx->dist_calculator().batch_dist(id);
        topk.emplace(id, dist);
      }
    }
    ctx->topk_to_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) return IndexError_Runtime;
  return 0;
}

int VamanaStreamer::search_bf_by_p_keys_impl(
    const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
    const IndexQueryMeta &qmeta, uint32_t count,
    Context::Pointer &context) const {
  int ret = check_params(query, qmeta);
  if (ailego_unlikely(ret != 0)) return ret;

  VamanaContext *ctx = dynamic_cast<VamanaContext *>(context.get());
  ailego_do_if_false(ctx) {
    LOG_ERROR("Cast context to VamanaContext failed");
    return IndexError_Cast;
  }
  if (ctx->magic() != magic_) {
    ret = update_context(ctx);
    if (ret != 0) return ret;
  }

  ctx->clear();
  ctx->update_dist_caculator_distance(search_distance_, search_batch_distance_);
  ctx->resize_results(count);

  auto &topk = ctx->topk_heap();

  for (size_t q = 0; q < count; ++q) {
    ctx->reset_query(query);
    topk.clear();
    for (const auto &keys : p_keys) {
      for (auto key : keys) {
        node_id_t id = entity_->get_id(key);
        if (id == kInvalidNodeId) continue;
        dist_t dist = ctx->dist_calculator().batch_dist(id);
        topk.emplace(id, dist);
      }
    }
    ctx->topk_to_result(q);
    query = static_cast<const char *>(query) + qmeta.element_size();
  }

  if (ailego_unlikely(ctx->error())) return IndexError_Runtime;
  return 0;
}

INDEX_FACTORY_REGISTER_STREAMER(VamanaStreamer);

}  // namespace core
}  // namespace zvec
