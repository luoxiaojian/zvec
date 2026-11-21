// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "hnsw_business_predicates.h"
#include <cassert>
#include <arrow/array.h>
#include <arrow/chunked_array.h>
#include <arrow/compute/expression.h>
#include <arrow/scalar.h>
#include <zvec/ailego/logger/logger.h>
#include "db/index/column/inverted_column/inverted_search_result.h"
#include "db/index/segment/segment.h"

namespace zvec::sqlengine {

InvertedBitmapFilter::InvertedBitmapFilter(const InvertedSearchResult *result)
    : bitmap_(result->bitmap()) {
  assert(bitmap_);
}

ForwardBitmapFilter::ForwardBitmapFilter(const arrow::ChunkedArray &bitmap) {
  assert(bitmap.type()->id() == arrow::Type::BOOL);
  chunks_.reserve(bitmap.num_chunks());
  for (const auto &chunk : bitmap.chunks()) {
    if (chunk->length() == 0) continue;
    chunks_.push_back({num_rows_, static_cast<uint64_t>(chunk->offset()),
                       chunk->data()->GetValues<uint8_t>(1, 0),
                       chunk->null_bitmap_data()});
    num_rows_ += chunk->length();
  }

  if (chunks_.size() < 2) {
    stride_ = num_rows_;
    return;
  }
  stride_ = chunks_[1].first_row;
  for (size_t i = 1; i < chunks_.size(); ++i) {
    const bool last = i + 1 == chunks_.size();
    const auto end = last ? num_rows_ : chunks_[i + 1].first_row;
    const auto length = end - chunks_[i].first_row;
    if (last ? length > stride_ : length != stride_) {
      stride_ = 0;
      break;
    }
  }
}

const ForwardBitmapChunkFilter *ForwardBitmapFilter::single_chunk(
    uint64_t block_offset, uint64_t block_rows) const {
  if (block_rows == 0 || block_offset >= num_rows_ ||
      block_rows > num_rows_ - block_offset) {
    return nullptr;
  }
  const size_t i = chunk_index(block_offset);
  const auto end =
      i + 1 == chunks_.size() ? num_rows_ : chunks_[i + 1].first_row;
  return block_rows <= end - block_offset ? &chunks_[i] : nullptr;
}

void ForwardBitmapFilter::log_out_of_range(uint64_t row) const {
  LOG_ERROR("ID is out or range: id[%zu] count[%zu]", static_cast<size_t>(row),
            static_cast<size_t>(num_rows_));
}

namespace {

// Preserve DocFilter's distinction between a SQL NULL (no match) and a
// failure to read or evaluate the row (no forward decision).
std::optional<bool> MatchArrow(const arrow::compute::Expression &expression,
                               const arrow::compute::ExecBatch *batch,
                               uint64_t row) {
  if (!batch) {
    LOG_ERROR("Fetch forward failed, id[%zu]", static_cast<size_t>(row));
    return std::nullopt;
  }
  auto result = arrow::compute::ExecuteScalarExpression(expression, *batch);
  if (!result.ok()) {
    LOG_ERROR("Execute scalar expression failed, id[%zu] err[%s]",
              static_cast<size_t>(row), result.status().ToString().c_str());
    return std::nullopt;
  }
  const auto &datum = *result;
  if (datum.is_scalar()) {
    const auto &scalar = datum.scalar_as<arrow::BooleanScalar>();
    return scalar.is_valid && scalar.value;
  }
  LOG_ERROR("Datum is not scalar, id[%zu] type[%s]", static_cast<size_t>(row),
            datum.type()->ToString().c_str());
  return std::nullopt;
}

}  // namespace

std::optional<bool> ArrowViewFilter::match(uint64_t row) const {
  if (row >= static_cast<uint64_t>(view->num_rows())) return false;
  return MatchArrow(*expression, view->read_row(row).get(), row);
}

std::optional<bool> ArrowFetchFilter::match(uint64_t row) const {
  if (columns->empty()) {
    // Literal expressions still run for each row with a one-row input.
    static const arrow::compute::ExecBatch literal_batch(
        std::vector<arrow::Datum>{}, 1);
    return MatchArrow(*expression, &literal_batch, row);
  }
  return MatchArrow(*expression, segment->fetch(*columns, row).get(), row);
}

}  // namespace zvec::sqlengine
