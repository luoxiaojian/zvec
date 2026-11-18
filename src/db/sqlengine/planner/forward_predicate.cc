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

#include "db/sqlengine/planner/forward_predicate.h"
#include <utility>
#include <vector>
#include <stile/functions.h>
#include <stile/table_index.h>
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/planner/ops/contain_op.h"

namespace zvec::sqlengine {
namespace {

// The element type each ARRAY_* data type stores.
bool ElementMatches(DataType type, arrow::Type::type element) {
  switch (type) {
    case DataType::ARRAY_INT32:
      return element == arrow::Type::INT32;
    case DataType::ARRAY_UINT32:
      return element == arrow::Type::UINT32;
    case DataType::ARRAY_INT64:
      return element == arrow::Type::INT64;
    case DataType::ARRAY_UINT64:
      return element == arrow::Type::UINT64;
    case DataType::ARRAY_FLOAT:
      return element == arrow::Type::FLOAT;
    case DataType::ARRAY_DOUBLE:
      return element == arrow::Type::DOUBLE;
    case DataType::ARRAY_STRING:
      return element == arrow::Type::STRING;
    case DataType::ARRAY_BOOL:
      return element == arrow::Type::BOOL;
    default:
      return false;
  }
}

// zvec's contain_all / contain_any (ops/contain_op.cc): a NULL list is
// NULL; elements and set values compare by their stored values with
// operator==, regardless of element validity.
const stile::FunctionRegistry &Functions() {
  static const stile::FunctionRegistry registry = [] {
    stile::FunctionRegistry functions;
    for (const bool all : {true, false}) {
      auto describe = [all](const arrow::compute::Expression::Call &call)
          -> arrow::Result<stile::ListMembership> {
        const auto *options =
            dynamic_cast<const ContainOp::Options *>(call.options.get());
        const auto &type = call.arguments[0].type();
        if (!options || !options->args.value_set || !type ||
            type->id() != arrow::Type::LIST ||
            !ElementMatches(options->args.data_type,
                            type->field(0)->type()->id())) {
          return arrow::Status::NotImplemented("CONTAIN operands");
        }
        return stile::ListMembership{
            all ? stile::ListMembership::Mode::kAll
                : stile::ListMembership::Mode::kAny,
            stile::ListMembership::Nulls::kStoredValue,
            options->args.value_set};
      };
      auto status = functions.AddListMembership(
          all ? kContainAll : kContainAny, std::move(describe));
      (void)status;
    }
    return functions;
  }();
  return registry;
}

}  // namespace

arrow::Result<std::unique_ptr<ForwardPredicate>> ForwardPredicate::Make(
    const arrow::compute::Expression &bound, ForwardReadView::Ptr view) {
  if (!view) return arrow::Status::Invalid("Missing forward view");
  std::vector<stile::ColumnIndex::Ptr> columns;
  columns.reserve(static_cast<size_t>(view->num_columns()));
  for (int i = 0; i < view->num_columns(); ++i) {
    const auto &index = view->column(i)->stile_index();
    if (!index) return arrow::Status::NotImplemented("Column not indexed");
    columns.push_back(index);
  }
  ARROW_ASSIGN_OR_RAISE(auto table,
                        stile::TableIndex::Make(view->schema(), view->num_rows(),
                                                std::move(columns)));
  stile::CompileOptions options;
  options.functions = &Functions();
  ARROW_ASSIGN_OR_RAISE(auto predicate,
                        stile::Predicate::Compile(bound, std::move(table),
                                                  options));
  return std::unique_ptr<ForwardPredicate>(
      new ForwardPredicate(std::move(view), std::move(predicate)));
}

}  // namespace zvec::sqlengine
