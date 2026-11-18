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

#include <cstdint>
#include <memory>
#include <arrow/compute/expression.h>
#include <arrow/result.h>
#include <stile/predicate.h>
#include "db/index/segment/forward_read_view.h"

namespace zvec {
namespace test {
class ForwardPredicateTest;
}  // namespace test

namespace sqlengine {

// Per-candidate evaluation of a bound forward filter over an owning view,
// by stile (https://github.com/luoxiaojian/stile). Every result equals
// Arrow's ExecuteScalarExpression on that row, with NULL treated as no
// match; anything stile does not cover returns NotImplemented so the caller
// keeps Arrow evaluation.
//
// Immutable after Make(); concurrent matches() calls are safe. The view and
// stile's table index keep every buffer alive.
class ForwardPredicate {
 public:
  static arrow::Result<std::unique_ptr<ForwardPredicate>> Make(
      const arrow::compute::Expression &bound, ForwardReadView::Ptr view);

  // Rows outside the view never match.
  bool matches(uint64_t row) const {
    return predicate_->Matches(static_cast<int64_t>(row));
  }

 private:
  friend class zvec::test::ForwardPredicateTest;

  ForwardPredicate(ForwardReadView::Ptr view,
                   std::unique_ptr<stile::Predicate> predicate)
      : view_(std::move(view)),
        predicate_(std::move(predicate)),
        layout_(predicate_->shares_chunk_layout() ? this : nullptr) {}

  ForwardReadView::Ptr view_;
  std::unique_ptr<stile::Predicate> predicate_;
  // Non-null when all referenced columns share one chunk layout (tests).
  const void *layout_;
};

}  // namespace sqlengine
}  // namespace zvec
