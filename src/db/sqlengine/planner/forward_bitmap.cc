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

#include "db/sqlengine/planner/forward_bitmap.h"
#include <arrow/array/concatenate.h>
#include <arrow/array/util.h>
#include <arrow/compute/api.h>
#include <zvec/ailego/logger/logger.h>
#include "db/sqlengine/planner/ops/bitmap_compare.h"

namespace zvec::sqlengine {
namespace cp = arrow::compute;
namespace {

cp::Expression specialize(const cp::Expression &expression,
                          const std::shared_ptr<cp::ScalarFunction> &function,
                          const cp::Kernel *kernel, size_t *replacements) {
  const auto *original = expression.call();
  if (!original) return expression;
  auto call = *original;
  for (auto &argument : call.arguments) {
    argument = specialize(argument, function, kernel, replacements);
  }
  // Bind first: a wider literal may require widening the input. Never change
  // those casts or truncate constants merely to enable an INT32 kernel.
  if (call.function_name == "greater" && call.arguments.size() == 2 &&
      call.arguments[0].type()->id() == arrow::Type::INT32 &&
      call.arguments[1].type()->id() == arrow::Type::INT32 &&
      call.arguments[1].literal() && call.arguments[1].literal()->is_scalar()) {
    // Keep the bound types, options, field indices and semantic function name.
    // Call::function owns the replacement kernel; no global registry mutation
    // or sharing of mutable per-query KernelState is needed.
    call.function = function;
    call.kernel = kernel;
    call.kernel_state.reset();
    ++*replacements;
  }
  return cp::Expression(std::move(call));
}

arrow::Result<std::shared_ptr<arrow::BooleanArray>> as_boolean_array(
    const arrow::Datum &result, int64_t num_rows) {
  if (!result.type() || result.type()->id() != arrow::Type::BOOL) {
    return arrow::Status::TypeError("Filter bitmap is not boolean type");
  }
  std::shared_ptr<arrow::Array> array;
  if (result.is_scalar()) {
    ARROW_ASSIGN_OR_RAISE(
        array, arrow::MakeArrayFromScalar(*result.scalar(), num_rows));
  } else if (result.is_array()) {
    array = result.make_array();
  } else if (result.is_chunked_array()) {
    const auto &chunks = result.chunked_array()->chunks();
    if (chunks.empty()) {
      ARROW_ASSIGN_OR_RAISE(array, arrow::MakeArrayOfNull(arrow::boolean(), 0));
    } else if (chunks.size() == 1) {
      array = chunks[0];
    } else {
      // Compact only boolean output, never input columns.
      ARROW_ASSIGN_OR_RAISE(array, arrow::Concatenate(chunks));
    }
  } else {
    return arrow::Status::TypeError("Filter did not produce a boolean array");
  }
  if (array->length() != num_rows) {
    return arrow::Status::Invalid("Filter bitmap row count mismatch");
  }
  return std::static_pointer_cast<arrow::BooleanArray>(std::move(array));
}

}  // namespace

arrow::Result<cp::Expression> bind_forward_bitmap_expression(
    const cp::Expression &expression, const arrow::Schema &schema,
    cp::ExecContext *context) {
  ARROW_ASSIGN_OR_RAISE(auto bound, expression.Bind(schema, context));
  if (!bound.IsScalarExpression() || bound.type()->id() != arrow::Type::BOOL) {
    return arrow::Status::TypeError(
        "Filter must be a scalar boolean expression");
  }
  const auto &selected = bitmap_internal::int32_bitmap_kernel();
  if (selected.isa == bitmap_internal::Int32BitmapIsa::kScalar) {
    LOG_DEBUG("Forward bitmap kernel[scalar] int32_comparisons[0]");
    return bound;
  }

  static const auto function_result =
      bitmap_internal::make_int32_bitmap_function(selected);
  ARROW_ASSIGN_OR_RAISE(auto function, function_result);
  ARROW_ASSIGN_OR_RAISE(
      const auto *kernel,
      function->DispatchExact({arrow::int32(), arrow::int32()}));
  size_t replacements = 0;
  auto result = specialize(bound, function, kernel, &replacements);
  LOG_DEBUG("Forward bitmap kernel[%s] int32_comparisons[%zu]",
            bitmap_internal::int32_bitmap_isa_name(selected.isa), replacements);
  return result;
}

arrow::Result<std::shared_ptr<arrow::BooleanArray>> evaluate_forward_bitmap(
    const arrow::Table &table, const cp::Expression &expression) {
  cp::ExecContext context;
  context.set_preallocate_contiguous(true);
  ARROW_ASSIGN_OR_RAISE(auto bound, bind_forward_bitmap_expression(
                                        expression, *table.schema(), &context));
  std::vector<arrow::Datum> columns;
  columns.reserve(table.num_columns());
  for (const auto &column : table.columns()) columns.emplace_back(column);
  cp::ExecBatch batch(std::move(columns), table.num_rows());
  ARROW_ASSIGN_OR_RAISE(auto result,
                        cp::ExecuteScalarExpression(bound, batch, &context));
  return as_boolean_array(result, table.num_rows());
}

arrow::Result<std::shared_ptr<arrow::BooleanArray>> evaluate_forward_bitmap(
    arrow::RecordBatchReader &reader, const cp::Expression &expression,
    int64_t num_rows) {
  if (!reader.schema() || num_rows < 0) {
    return arrow::Status::Invalid("Invalid forward bitmap reader");
  }
  cp::ExecContext context;
  context.set_preallocate_contiguous(true);
  ARROW_ASSIGN_OR_RAISE(
      auto bound,
      bind_forward_bitmap_expression(expression, *reader.schema(), &context));
  arrow::ArrayVector output;
  int64_t rows = 0;
  while (rows < num_rows) {
    ARROW_ASSIGN_OR_RAISE(auto batch, reader.Next());
    if (!batch)
      return arrow::Status::Invalid("Forward bitmap scan is incomplete");
    if (batch->num_rows() == 0) continue;
    if (batch->num_rows() > num_rows - rows)
      batch = batch->Slice(0, num_rows - rows);
    ARROW_ASSIGN_OR_RAISE(
        auto result,
        cp::ExecuteScalarExpression(bound, cp::ExecBatch(*batch), &context));
    ARROW_ASSIGN_OR_RAISE(auto bits,
                          as_boolean_array(result, batch->num_rows()));
    output.emplace_back(std::move(bits));
    rows += batch->num_rows();
  }
  return as_boolean_array(arrow::Datum(std::make_shared<arrow::ChunkedArray>(
                              std::move(output), arrow::boolean())),
                          num_rows);
}

}  // namespace zvec::sqlengine
