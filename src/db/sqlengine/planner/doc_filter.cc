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

#include "db/sqlengine/planner/doc_filter.h"
#include <optional>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <zvec/ailego/logger/logger.h>
#include "db/sqlengine/planner/forward_bitmap.h"
#include "db/sqlengine/planner/invert_search.h"

namespace zvec::sqlengine {

Status DocFilter::compute_filter() {
  forward_prepared_ = false;
  forward_bitmap_.reset();
  forward_view_.reset();
  forward_rows_ = 0;
  sealed_deletion_.reset();
  if (invert_cond_) {
    InvertSearch search(segment_.get());
    auto invert_res = search.exec_invert_cond_tree(invert_cond_.get());
    if (!invert_res) {
      return Status::InternalError("Execute invert search failed: ",
                                   invert_res.error().message());
    }
    invert_result_ = invert_res.value();
    invert_filter_ = invert_result_->make_filter();
  }

  // Preserve PR905's live deletion check for every prepared filter path.
  const bool with_deletion = delete_filter_ != nullptr;
  if (bitmap_filter_expr_) {
    const auto &columns = query_info_->get_forward_filter_field_names();
    auto prepared = segment_->prepare_forward_read(columns, with_deletion);
    if (!prepared.ok()) {
      return Status::InternalError("Prepare forward bitmap failed: ",
                                   prepared.status().ToString());
    }
    auto state = prepared.MoveValueUnsafe();
    auto compute =
        [&]() -> arrow::Result<std::shared_ptr<arrow::BooleanArray>> {
      if (state.view) {
        // PR905 shares each immutable column across projections. Assemble only
        // a Table wrapper: preserve its chunks and owned buffers without
        // traversing rows, copying values or doing per-candidate view reads.
        std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
        arrays.reserve(state.view->num_columns());
        bool physical = true;
        for (int i = 0; i < state.view->num_columns(); ++i) {
          const auto &data = state.view->column(i)->data();
          if (!data) {
            physical = false;
            break;
          }
          arrays.push_back(data);
        }
        if (physical) {
          auto table = arrow::Table::Make(state.schema, std::move(arrays),
                                          static_cast<int64_t>(state.num_rows));
          return evaluate_forward_bitmap(*table, *bitmap_filter_expr_);
        }
      }
      if (state.num_rows == 0) {
        ARROW_ASSIGN_OR_RAISE(auto empty,
                              arrow::Table::MakeEmpty(state.schema));
        return evaluate_forward_bitmap(*empty, *bitmap_filter_expr_);
      }
      if (columns.empty()) {
        auto empty = arrow::Table::Make(state.schema, arrow::ArrayVector{},
                                        static_cast<int64_t>(state.num_rows));
        return evaluate_forward_bitmap(*empty, *bitmap_filter_expr_);
      }
      // Keep bounded input memory for mutable/Parquet stores; the native reader
      // also emits virtual LOCAL_ROW_ID columns that have no physical buffers.
      // Retain only boolean output and evaluate the captured row prefix.
      auto reader = segment_->scan(columns);
      if (!reader) return arrow::Status::Invalid("Scan forward bitmap failed");
      return evaluate_forward_bitmap(*reader, *bitmap_filter_expr_,
                                     static_cast<int64_t>(state.num_rows));
    };
    auto result = compute();
    if (!result.ok()) {
      return Status::InternalError("Execute filter bitmap failed: ",
                                   result.status().ToString());
    }
    forward_bitmap_ = result.MoveValueUnsafe();
    sealed_deletion_ = std::move(state.deletion_filter);
  }

  if (forward_filter_expr_) {
    // get schema to bind to Expression
    const auto columns = query_info_->get_forward_filter_field_names();
    auto prepared = segment_->prepare_forward_read(columns, with_deletion);
    if (!prepared.ok()) {
      return Status::InternalError("Prepare forward read failed: ",
                                   prepared.status().ToString());
    }
    auto state = prepared.MoveValueUnsafe();
    forward_rows_ = state.num_rows;
    forward_view_ = std::move(state.view);
    sealed_deletion_ = std::move(state.deletion_filter);
    const auto &schema = state.schema;
    // Log outside segment synchronization.
    if (forward_view_) {
      LOG_DEBUG("Forward filter view: segment[%d] rows[%zu] columns[%zu]",
                segment_->id(), static_cast<size_t>(forward_rows_),
                columns.size());
    } else {
      LOG_DEBUG("Forward filter locked fetch: segment[%d] rows[%zu]",
                segment_->id(), static_cast<size_t>(forward_rows_));
    }
    auto bind_res = forward_filter_expr_->Bind(*schema);
    if (!bind_res.ok()) {
      return Status::InternalError("Bind forward filter expression failed",
                                   bind_res.status().ToString());
    }
    *forward_filter_expr_ = bind_res.MoveValueUnsafe();
    forward_prepared_ = true;
  } else if (with_deletion && !bitmap_filter_expr_) {
    // Without a forward predicate the capture is only an optimization: on
    // any failure keep the segment's filter, which gives the same results.
    auto prepared = segment_->prepare_forward_read({}, true);
    if (prepared.ok()) {
      sealed_deletion_ = std::move(prepared->deletion_filter);
    }
  }
  return Status::OK();
}

bool DocFilter::empty() const {
  return !(delete_filter_ || invert_filter_ || bitmap_filter_expr_ ||
           forward_filter_expr_);
}

bool DocFilter::is_filtered(uint64_t id) const {
  // An unprepared (or failed) forward filter never admits candidates.
  if (forward_filter_expr_ && !forward_prepared_) return true;
  if (bitmap_filter_expr_ && !forward_bitmap_) return true;
  if (forward_bitmap_ &&
      id >= static_cast<uint64_t>(forward_bitmap_->length())) {
    return true;
  }
  // An owning view holds exactly the rows captured at preparation; nothing
  // beyond them can be read. Locked fetch keeps reading the live segment.
  if (forward_view_ && id >= forward_rows_) return true;
  if (delete_filter_ &&
      (sealed_deletion_ ? sealed_deletion_ : delete_filter_)->is_filtered(id)) {
    return true;
  }
  if (invert_filter_ && invert_filter_->is_filtered(id)) {
    return true;
  }
  auto forward_bit = get_forward_bit(id);
  if (!forward_bit) {
    return false;
  }
  // revert to return false if forward filter is matched
  return !forward_bit.value();
}

std::optional<bool> DocFilter::get_forward_bit(uint64_t id) const {
  if (forward_filter_expr_) {
    return is_matched_by_forward_filter(id);
  }
  if (!forward_bitmap_) {
    return std::nullopt;
  }
  return forward_bitmap_->IsValid(id) && forward_bitmap_->Value(id);
}

std::optional<std::vector<uint64_t>> DocFilter::get_bf_by_keys_and_update(
    float ratio) {
  // TODO: support forward
  if (!invert_result_) {
    return std::nullopt;
  }
  // Insert/flush() rewrite the block behind doc_count(), so read it locked.
  size_t doc_count = segment_->doc_count_snapshot();
  uint64_t bf_by_keys_threshold = static_cast<uint64_t>(doc_count * ratio);

  // decide to use brute force by keys or not
  if (size_t match_count = invert_result_->count();
      match_count <= bf_by_keys_threshold) {
    std::vector<uint32_t> ids;
    invert_result_->extract_ids(&ids);
    invert_filter_.reset();
    invert_result_.reset();
    LOG_INFO(
        "Use brute force by keys, doc_count[%zu] invert_result_count[%zu] "
        "ratio[%.4f]",
        doc_count, match_count, ratio);
    return std::vector<uint64_t>(ids.begin(), ids.end());
  } else {
    LOG_DEBUG(
        "Not use brute force by keys, doc_count[%zu] invert_result_count[%zu] "
        "ratio[%.4f]",
        doc_count, match_count, ratio);
  }
  return std::nullopt;
}

std::optional<bool> DocFilter::is_matched_by_forward_filter(uint64_t id) const {
  ExecBatchPtr exec_batch;
  if (forward_view_) {
    exec_batch = forward_view_->read_row(id);
  } else {
    const auto &columns = query_info_->get_forward_filter_field_names();
    // Literal predicates need a one-row batch but no scalar storage access.
    exec_batch = columns.empty() ? std::make_shared<arrow::compute::ExecBatch>(
                                       std::vector<arrow::Datum>{}, 1)
                                 : segment_->fetch(columns, id);
  }
  if (!exec_batch) {
    LOG_ERROR("Fetch forward failed, id[%zu]", (size_t)id);
    return std::nullopt;
  }
  auto maybe_result = arrow::compute::ExecuteScalarExpression(
      *forward_filter_expr_, *exec_batch);
  if (!maybe_result.ok()) {
    LOG_ERROR("Execute scalar expression failed, id[%zu] err[%s]", (size_t)id,
              maybe_result.status().ToString().c_str());
    return std::nullopt;
  }
  arrow::Datum datum = maybe_result.MoveValueUnsafe();
  if (datum.is_scalar()) {
    const auto &scalar = datum.scalar_as<arrow::BooleanScalar>();
    return scalar.is_valid && scalar.value;
  }
  LOG_ERROR("Datum is not scalar, id[%zu] type[%s]", (size_t)id,
            datum.type()->ToString().c_str());
  return std::nullopt;
}


}  // namespace zvec::sqlengine
