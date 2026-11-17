// Copyright 2026-present the zvec project
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

#include <arrow/array.h>
#include <arrow/compute/expression.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>

namespace zvec::sqlengine {

// Bind with Arrow's normal type/cast rules, then specialize supported exact
// comparisons. The returned expression owns its kernels and is query-local.
arrow::Result<arrow::compute::Expression> bind_forward_bitmap_expression(
    const arrow::compute::Expression &expression, const arrow::Schema &schema,
    arrow::compute::ExecContext *context);

// Preserve input chunks and row order. The result is always one BooleanArray,
// including scalar predicates and kernels that produce chunked output.
// NULL remains NULL here; consumers treat it as not matching at the boundary.
arrow::Result<std::shared_ptr<arrow::BooleanArray>> evaluate_forward_bitmap(
    const arrow::Table &table, const arrow::compute::Expression &expression);

// Streaming fallback for stores that do not own an immutable in-memory Table.
// Retain only boolean output, and evaluate the captured row prefix even if a
// writing segment grows after preparation. A short read is an error.
arrow::Result<std::shared_ptr<arrow::BooleanArray>> evaluate_forward_bitmap(
    arrow::RecordBatchReader &reader,
    const arrow::compute::Expression &expression, int64_t num_rows);

}  // namespace zvec::sqlengine
