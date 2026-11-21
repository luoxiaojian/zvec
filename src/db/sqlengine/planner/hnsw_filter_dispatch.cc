// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "hnsw_filter_dispatch.h"
#include <type_traits>
#include "core/algorithm/hnsw/hnsw_node_filter.h"
#include "db/index/segment/segment.h"

namespace zvec::sqlengine {
namespace {

using Block = vector_column_params::HnswFilterBlock;
enum class Deletions { None, Sealed, LiveSegment };

template <typename Predicate>
bool reject_row(const Block &block, uint64_t row) {
  return static_cast<const Predicate *>(block.predicate)->reject(row);
}

bool reject_inverted_and_chunk(const Block &block, uint64_t row) {
  const auto &both = *static_cast<const InvertedAndForwardFilter *>(
      block.predicate);
  return both.inverted.reject(row) ||
         static_cast<const ForwardBitmapChunkFilter *>(block.chunk)->reject(row);
}

template <auto RejectRow, Deletions Mode, bool SingleChunk>
bool reject_node(const Block &block, uint32_t node) {
  if constexpr (SingleChunk) {
    // A writing block can grow after preparation. The raw span covers only
    // the captured rows; its original key filter handles later nodes.
    if (node >= block.size) return block.is_filtered(node);
  }
  const uint64_t row = block.offset + node;
  if constexpr (Mode == Deletions::Sealed) {
    if (block.deletions->contains(node)) return true;
  } else if constexpr (Mode == Deletions::LiveSegment) {
    if (block.segment_deletions->is_filtered(row)) return true;
  }
  return RejectRow(block, row);
}

template <auto RejectRow, bool SingleChunk = false>
void install(Block &block, core::IndexFilter &engine_filter) {
  constexpr auto sealed = &reject_node<RejectRow, Deletions::Sealed, SingleChunk>;
  constexpr auto live =
      &reject_node<RejectRow, Deletions::LiveSegment, SingleChunk>;
  constexpr auto business = &reject_node<RejectRow, Deletions::None, SingleChunk>;
  if (block.deletions) {
    engine_filter.set(core::make_hnsw_node_filter<sealed>(block));
  } else if (block.segment_deletions) {
    engine_filter.set(core::make_hnsw_node_filter<live>(block));
  } else {
    engine_filter.set(core::make_hnsw_node_filter<business>(block));
  }
}

template <typename Predicate>
void bind_block(Block &block, core::IndexFilter &engine_filter) {
  const auto &plan = *static_cast<const HnswFilterPlan *>(block.predicate);
  const auto &predicate = std::get<Predicate>(plan.business);
  block.segment_deletions = plan.segment_deletions;
  block.predicate = &predicate;
  if constexpr (std::is_same_v<Predicate, ForwardBitmapFilter>) {
    if (auto *chunk = predicate.single_chunk(block.offset, block.size)) {
      block.predicate = chunk;
      install<&reject_row<ForwardBitmapChunkFilter>, true>(block, engine_filter);
      return;
    }
  } else if constexpr (std::is_same_v<Predicate, InvertedAndForwardFilter>) {
    if (auto *chunk = predicate.forward.single_chunk(block.offset, block.size)) {
      block.chunk = chunk;
      install<&reject_inverted_and_chunk, true>(block, engine_filter);
      return;
    }
  }
  install<&reject_row<Predicate>>(block, engine_filter);
}

}  // namespace

void HnswFilterPlan::bind(vector_column_params::QueryParams &params) const & {
  params.deletion_set = deletions;
  params.hnsw_business_filter = {};
  std::visit(
      [&](const auto &predicate) {
        using Predicate = std::decay_t<decltype(predicate)>;
        if constexpr (!std::is_same_v<Predicate, std::monostate>) {
          params.hnsw_business_filter = {this, &bind_block<Predicate>};
        }
      },
      business);
}

}  // namespace zvec::sqlengine
