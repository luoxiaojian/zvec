// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <arrow/type_fwd.h>
#include <arrow/util/bit_util.h>
#include <roaring/roaring.h>
#include "db/sqlengine/stile/predicate.h"

namespace zvec {
class ForwardReadView;
class InvertedSearchResult;
class Segment;

namespace sqlengine {

// Prepared predicates borrow the segment-local inputs owned by DocFilter.
struct InvertedBitmapFilter {
  explicit InvertedBitmapFilter(const InvertedSearchResult *result);

  bool reject(uint64_t row) const {
    return !roaring_bitmap_contains(bitmap_, static_cast<uint32_t>(row));
  }

 private:
  const roaring_bitmap_t *bitmap_;
};

struct ForwardBitmapChunkFilter {
  uint64_t first_row;
  uint64_t bit_offset;
  const uint8_t *values;
  const uint8_t *validity;

  bool reject(uint64_t row) const {
    const uint64_t bit = bit_offset + (row - first_row);
    return (validity && !arrow::bit_util::GetBit(validity, bit)) ||
           !arrow::bit_util::GetBit(values, bit);
  }
};

class ForwardBitmapFilter {
 public:
  // Borrow the buffers of a prepared Arrow boolean column, including slices.
  explicit ForwardBitmapFilter(const arrow::ChunkedArray &bitmap);

  bool reject(uint64_t row) const {
    if (row >= num_rows_) {
      log_out_of_range(row);
      return false;
    }
    return chunks_[chunk_index(row)].reject(row);
  }

  // A block wholly within one chunk can skip chunk lookup on every candidate.
  const ForwardBitmapChunkFilter *single_chunk(uint64_t block_offset,
                                               uint64_t block_rows) const;

 private:
  size_t chunk_index(uint64_t row) const {
    if (stride_) return row / stride_;
    auto it = std::upper_bound(
        chunks_.begin(), chunks_.end(), row,
        [](uint64_t r, const auto &chunk) { return r < chunk.first_row; });
    return static_cast<size_t>(it - chunks_.begin() - 1);
  }

  void log_out_of_range(uint64_t row) const;

  std::vector<ForwardBitmapChunkFilter> chunks_;
  uint64_t num_rows_{0};
  uint64_t stride_{0};
};

struct InvertedAndForwardFilter {
  InvertedBitmapFilter inverted;
  ForwardBitmapFilter forward;

  bool reject(uint64_t row) const {
    return inverted.reject(row) || forward.reject(row);
  }
};

struct StileFilter {
  const stile::Predicate *predicate;

  bool reject(uint64_t row) const {
    return !predicate->matches(static_cast<int64_t>(row));
  }
};

struct ArrowViewFilter {
  const ForwardReadView *view;
  const arrow::compute::Expression *expression;

  std::optional<bool> match(uint64_t row) const;
  bool reject(uint64_t row) const {
    return !match(row).value_or(true);
  }
};

struct ArrowFetchFilter {
  const Segment *segment;
  const std::vector<std::string> *columns;
  const arrow::compute::Expression *expression;

  std::optional<bool> match(uint64_t row) const;
  bool reject(uint64_t row) const {
    return !match(row).value_or(true);
  }
};

}  // namespace sqlengine
}  // namespace zvec
