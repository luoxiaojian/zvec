// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "db/index/column/vector_column/vector_column_params.h"
#include "db/sqlengine/planner/hnsw_business_predicates.h"

namespace zvec::sqlengine {

// The plan owns the chunk metadata and borrows its inputs from DocFilter.
// Keep both alive and do not reprepare the filter during the synchronous search.
struct HnswFilterPlan {
  using Business = std::variant<std::monostate, InvertedBitmapFilter,
                                ForwardBitmapFilter, InvertedAndForwardFilter,
                                StileFilter, ArrowViewFilter, ArrowFetchFilter>;
  Business business;
  const NodeDeletionSet *deletions{};
  const SegmentIndexFilter *segment_deletions{};

  void bind(vector_column_params::QueryParams &params) const &;
  void bind(vector_column_params::QueryParams &) const && = delete;
};

}  // namespace zvec::sqlengine
