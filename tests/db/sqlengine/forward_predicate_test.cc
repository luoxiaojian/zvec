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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gtest/gtest.h>
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/planner/op_register.h"
#include "db/sqlengine/planner/ops/contain_op.h"

namespace zvec::test {

namespace cp = arrow::compute;
using sqlengine::ContainOp;
using sqlengine::ForwardPredicate;

template <typename T>
T Unwrap(arrow::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(result.status().ToString());
  return std::move(result).ValueOrDie();
}

void Check(const arrow::Status &status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}

class ForwardPredicateTest : public testing::Test {
 public:
  static ForwardReadView::Ptr MakeView(std::shared_ptr<arrow::Table> table) {
    return ForwardReadView::Make(std::move(table));
  }
  static ForwardReadView::Ptr MakeRowIdView(int64_t rows) {
    auto field = arrow::field("row_id", arrow::uint64(), false);
    return ForwardReadView::Compose(arrow::schema({field}), rows,
                                    {ForwardColumn::MakeRowId(field, rows)});
  }
  static bool SharesLayout(const ForwardPredicate &predicate) {
    return predicate.layout_ != nullptr;
  }
  static ForwardColumn::Ptr MakeColumn(
      const std::shared_ptr<arrow::ChunkedArray> &data) {
    return Unwrap(ForwardColumn::Make(arrow::field("c", data->type()), data,
                                      data->length()));
  }

 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(cp::Initialize().ok());
    ASSERT_TRUE(sqlengine::OpRegister::register_ops().ok());
  }

  // The per-candidate Arrow path the predicate replaces; throws on errors.
  static bool PerRow(const cp::Expression &bound, const ForwardReadView &view,
                     uint64_t row) {
    auto batch = view.read_row(row);
    auto datum = Unwrap(cp::ExecuteScalarExpression(bound, *batch));
    const auto &scalar = datum.scalar_as<arrow::BooleanScalar>();
    return scalar.is_valid && scalar.value;
  }
};

namespace {

// Chunk layouts of sealed segments: arbitrary, regular, and blocks that each
// end with a partial chunk.
enum class Layout { kRandom, kUniform, kBlocks };

// Random data with nulls, NaN payloads, signed zeros, extremes, multibyte and
// invalid UTF-8, LIKE metacharacters, lists, slices and per-column chunking.
class Generator {
 public:
  explicit Generator(uint64_t seed) : rng_(seed) {}

  size_t below(size_t n) {
    return std::uniform_int_distribution<size_t>(0, n - 1)(rng_);
  }
  bool coin(double p) {
    return std::bernoulli_distribution(p)(rng_);
  }

  std::shared_ptr<arrow::Scalar> special(const arrow::DataType &type) {
    switch (type.id()) {
      case arrow::Type::FLOAT: {
        const float values[] = {std::numeric_limits<float>::quiet_NaN(),
                                FromBits<float>(0x7fc00123u),
                                -0.0f,
                                0.0f,
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::denorm_min(),
                                std::numeric_limits<float>::lowest(),
                                1.5f};
        return std::make_shared<arrow::FloatScalar>(values[below(9)]);
      }
      case arrow::Type::DOUBLE: {
        const double values[] = {std::numeric_limits<double>::quiet_NaN(),
                                 FromBits<double>(0x7ff8000000000123ull),
                                 -0.0,
                                 0.0,
                                 std::numeric_limits<double>::infinity(),
                                 -std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::max(),
                                 -2.25,
                                 1.5};
        return std::make_shared<arrow::DoubleScalar>(values[below(9)]);
      }
      case arrow::Type::BOOL:
        return std::make_shared<arrow::BooleanScalar>(coin(0.5));
      case arrow::Type::STRING:
        return std::make_shared<arrow::StringScalar>(word());
      case arrow::Type::LARGE_STRING:
        return std::make_shared<arrow::LargeStringScalar>(word());
      case arrow::Type::BINARY:
        return std::make_shared<arrow::BinaryScalar>(word());
      case arrow::Type::LARGE_BINARY:
        return std::make_shared<arrow::LargeBinaryScalar>(word());
      default:
        break;
    }
    const size_t pick = below(6);
    if (pick < 2) return Limit(type, pick == 0);
    // Negative values wrap for unsigned types, adding more extremes.
    const int64_t small = static_cast<int64_t>(below(21)) - 10;
    return Unwrap(arrow::MakeScalar(type.GetSharedPtr(), small));
  }

  std::string word() {
    static const char *tokens[] = {"a",  "b",  "ab", "%",   "_",   "\\", "é",
                                   "中", ".",  "*",  "(",   "[",   "\n", "A",
                                   "Ab", "%a", "a_", "\\%", "\xff"};
    std::string out;
    const size_t n = below(5);
    for (size_t i = 0; i < n; ++i) out += tokens[below(19)];
    return out;
  }

  std::string pattern() {
    static const char *tokens[] = {"a",   "b",  "%",    "_",   "\\", "\\%",
                                   "\\_", "é",  "中",   ".",   "*",  "(",
                                   "A",   "%%", "\xff", "\xc3"};
    std::string out;
    const size_t n = below(6);
    for (size_t i = 0; i < n; ++i) out += tokens[below(16)];
    return out;
  }

  // Chunk sizes for `rows` rows in the given layout.
  std::vector<int64_t> sizes(int64_t rows, Layout layout) {
    const int64_t stride = 7 + static_cast<int64_t>(below(58));
    const int64_t block = stride * (1 + static_cast<int64_t>(below(3))) + 1 +
                          static_cast<int64_t>(below(stride - 1));
    std::vector<int64_t> out;
    for (int64_t begin = 0; begin < rows;) {
      int64_t size = stride;
      if (layout == Layout::kRandom) {
        size = coin(0.1) ? 0 : 1 + static_cast<int64_t>(below(rows / 3 + 1));
      } else if (layout == Layout::kBlocks) {
        size = std::min(stride, block - begin % block);
      }
      size = std::min(size, rows - begin);
      out.push_back(size);
      begin += size;
    }
    return out;
  }

  // A sliced column split into `chunk_sizes`, or into its own random sizes.
  std::shared_ptr<arrow::ChunkedArray> column(
      const std::shared_ptr<arrow::DataType> &type, int64_t rows, Layout layout,
      const std::vector<int64_t> *chunk_sizes = nullptr) {
    std::unique_ptr<arrow::ArrayBuilder> builder;
    Check(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));
    const int64_t skip = static_cast<int64_t>(below(9));
    for (int64_t i = 0; i < rows + skip; ++i) append(type, builder.get());
    const auto array = Unwrap(builder->Finish())->Slice(skip, rows);
    arrow::ArrayVector chunks;
    int64_t begin = 0;
    for (const auto size : chunk_sizes ? *chunk_sizes : sizes(rows, layout)) {
      chunks.push_back(array->Slice(begin, size));
      begin += size;
    }
    return std::make_shared<arrow::ChunkedArray>(std::move(chunks), type);
  }

 private:
  void append(const std::shared_ptr<arrow::DataType> &type,
              arrow::ArrayBuilder *builder) {
    if (coin(0.1)) {
      Check(builder->AppendNull());
    } else if (type->id() == arrow::Type::LIST) {
      auto *lists = static_cast<arrow::ListBuilder *>(builder);
      Check(lists->Append());
      const auto &element = type->field(0)->type();
      for (size_t i = below(6); i > 0; --i) {
        Check(coin(0.1)
                  ? lists->value_builder()->AppendNull()
                  : lists->value_builder()->AppendScalar(*special(*element)));
      }
    } else {
      Check(builder->AppendScalar(*special(*type)));
    }
  }

  template <typename T, typename Bits>
  static T FromBits(Bits bits) {
    T value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  static std::shared_ptr<arrow::Scalar> Limit(const arrow::DataType &type,
                                              bool min) {
    switch (type.id()) {
#define ZVEC_LIMIT(ID, CTYPE)                                               \
  case arrow::Type::ID:                                                     \
    return Unwrap(arrow::MakeScalar(type.GetSharedPtr(),                    \
                                    min ? std::numeric_limits<CTYPE>::min() \
                                        : std::numeric_limits<CTYPE>::max()));
      ZVEC_LIMIT(INT8, int8_t)
      ZVEC_LIMIT(INT16, int16_t)
      ZVEC_LIMIT(INT32, int32_t)
      ZVEC_LIMIT(INT64, int64_t)
      ZVEC_LIMIT(UINT8, uint8_t)
      ZVEC_LIMIT(UINT16, uint16_t)
      ZVEC_LIMIT(UINT32, uint32_t)
      ZVEC_LIMIT(UINT64, uint64_t)
#undef ZVEC_LIMIT
      default:
        return nullptr;
    }
  }

  std::mt19937_64 rng_;
};

DataType ArrayType(arrow::Type::type element) {
  switch (element) {
    case arrow::Type::INT32:
      return DataType::ARRAY_INT32;
    case arrow::Type::UINT32:
      return DataType::ARRAY_UINT32;
    case arrow::Type::INT64:
      return DataType::ARRAY_INT64;
    case arrow::Type::UINT64:
      return DataType::ARRAY_UINT64;
    case arrow::Type::FLOAT:
      return DataType::ARRAY_FLOAT;
    case arrow::Type::DOUBLE:
      return DataType::ARRAY_DOUBLE;
    case arrow::Type::STRING:
      return DataType::ARRAY_STRING;
    default:
      return DataType::ARRAY_BOOL;
  }
}

// Random predicates over the planner's vocabulary (comparisons with typed
// literals, IS [NOT] NULL, IN / NOT IN, LIKE, CONTAIN, array_length, AND,
// OR, NOT) plus Arrow siblings: other null-matching behaviors, plain and/or,
// literals on the left, and literals wider than the column.
class ExpressionGenerator {
 public:
  ExpressionGenerator(Generator *gen, std::shared_ptr<arrow::Schema> schema)
      : gen_(gen), schema_(std::move(schema)) {}

  cp::Expression predicate(int depth) {
    if (depth == 0 || gen_->coin(0.35)) return leaf();
    static const char *logic[] = {"and_kleene", "or_kleene", "and", "or"};
    if (gen_->coin(0.2)) return cp::call("invert", {predicate(depth - 1)});
    return cp::call(logic[gen_->below(4)],
                    {predicate(depth - 1), predicate(depth - 1)});
  }

 private:
  cp::Expression literal(const arrow::DataType &type) {
    if (gen_->coin(0.05)) {
      return cp::literal(arrow::MakeNullScalar(type.GetSharedPtr()));
    }
    return cp::literal(gen_->special(type));
  }

  // value OP literal, with the literal on either side.
  cp::Expression compare(const cp::Expression &value,
                         const cp::Expression &literal) {
    static const char *ops[] = {"equal",      "not_equal", "less",
                                "less_equal", "greater",   "greater_equal"};
    const char *op = ops[gen_->below(6)];
    return gen_->coin(0.3) ? cp::call(op, {literal, value})
                           : cp::call(op, {value, literal});
  }

  std::shared_ptr<arrow::Array> values(const arrow::DataType &type,
                                       double nulls) {
    std::unique_ptr<arrow::ArrayBuilder> builder;
    Check(arrow::MakeBuilder(arrow::default_memory_pool(), type.GetSharedPtr(),
                             &builder));
    for (size_t i = gen_->below(6); i > 0; --i) {
      Check(gen_->coin(nulls) ? builder->AppendNull()
                              : builder->AppendScalar(*gen_->special(type)));
    }
    return Unwrap(builder->Finish());
  }

  cp::Expression list_leaf(const cp::Expression &ref,
                           const arrow::DataType &type) {
    const auto &element = *type.field(0)->type();
    const auto length = cp::call("list_value_length", {ref});
    switch (gen_->below(4)) {
      case 0: {
        const auto &name =
            gen_->coin(0.5) ? sqlengine::kContainAll : sqlengine::kContainAny;
        auto contain =
            cp::call(name, {ref},
                     std::make_shared<ContainOp::Options>(ContainOp::Args{
                         values(element, 0.15), ArrayType(element.id())}));
        return gen_->coin(0.25) ? cp::not_(contain) : contain;
      }
      case 1: {
        // The planner types array_length literals as uint32; Bind narrows
        // small ones and widens the length for the rest.
        const uint32_t value = gen_->coin(0.15)
                                   ? 4000000000u
                                   : static_cast<uint32_t>(gen_->below(7));
        return compare(length, cp::literal(value));
      }
      case 2:
        return gen_->coin(0.5) ? cp::is_null(ref) : cp::is_valid(ref);
      default:
        return gen_->coin(0.5) ? cp::is_null(length) : cp::is_valid(length);
    }
  }

  cp::Expression leaf() {
    const auto &field = schema_->field(static_cast<int>(
        gen_->below(static_cast<size_t>(schema_->num_fields()))));
    const auto &type = *field->type();
    const auto ref = cp::field_ref(field->name());
    if (type.id() == arrow::Type::LIST) return list_leaf(ref, type);
    const bool binary = arrow::is_base_binary_like(type.id());
    const size_t kind = gen_->below(11);
    if (kind < 5) return compare(ref, literal(type));
    if (kind == 5 && arrow::is_numeric(type.id())) {
      // Wider literal: Bind narrows it or inserts a cast of the column.
      return compare(ref, arrow::is_integer(type.id())
                              ? cp::literal(gen_->coin(0.5) ? int64_t{1} << 40
                                                            : int64_t{-1})
                              : cp::literal(0.1));
    }
    if (kind == 6) {
      return gen_->coin(0.5) ? cp::is_valid(ref)
                             : cp::is_null(ref, gen_->coin(0.5));
    }
    if (kind <= 8) {
      static const cp::SetLookupOptions::NullMatchingBehavior behaviors[] = {
          cp::SetLookupOptions::MATCH, cp::SetLookupOptions::SKIP,
          cp::SetLookupOptions::EMIT_NULL, cp::SetLookupOptions::INCONCLUSIVE};
      auto in = cp::call(
          "is_in", {ref},
          cp::SetLookupOptions(values(type, 0.15), behaviors[gen_->below(4)]));
      return gen_->coin(0.25) ? cp::not_(in) : in;
    }
    if (kind == 9 && binary) {
      return cp::call(
          "match_like", {ref},
          cp::MatchSubstringOptions(gen_->pattern(), gen_->coin(0.2)));
    }
    if (type.id() == arrow::Type::BOOL && gen_->coin(0.5)) return ref;
    return literal(*arrow::boolean());
  }

  Generator *gen_;
  std::shared_ptr<arrow::Schema> schema_;
};

}  // namespace

TEST_F(ForwardPredicateTest, RandomPredicatesMatchArrowPerRow) {
  const std::vector<std::shared_ptr<arrow::DataType>> types = {
      arrow::int8(),
      arrow::int16(),
      arrow::int32(),
      arrow::int64(),
      arrow::uint8(),
      arrow::uint16(),
      arrow::uint32(),
      arrow::uint64(),
      arrow::float32(),
      arrow::float64(),
      arrow::boolean(),
      arrow::utf8(),
      arrow::large_utf8(),
      arrow::binary(),
      arrow::large_binary(),
      arrow::list(arrow::int32()),
      arrow::list(arrow::uint32()),
      arrow::list(arrow::int64()),
      arrow::list(arrow::uint64()),
      arrow::list(arrow::float32()),
      arrow::list(arrow::float64()),
      arrow::list(arrow::utf8()),
      arrow::list(arrow::boolean())};
  size_t compiled = 0, arrow_errors = 0, lossy_casts = 0, unbindable = 0;
  size_t contains = 0, lengths = 0, casts = 0, shared = 0;
  for (uint64_t seed = 1; seed <= 12; ++seed) {
    Generator gen(seed);
    const auto layout = static_cast<Layout>(seed % 3);
    const int64_t rows = 300 + static_cast<int64_t>(gen.below(400));
    // Seeds above 9 chunk every column alike, as one sealed block set does.
    const auto common = gen.sizes(rows, layout);
    arrow::FieldVector fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    for (size_t i = 0; i < types.size(); ++i) {
      fields.push_back(arrow::field("c" + std::to_string(i), types[i]));
      columns.push_back(
          gen.column(types[i], rows, layout, seed > 9 ? &common : nullptr));
    }
    auto table = arrow::Table::Make(arrow::schema(fields), columns, rows);
    auto view = MakeView(table);
    ASSERT_NE(view, nullptr);
    if (layout == Layout::kUniform) EXPECT_GT(view->column(2)->stride(), 0);
    if (layout == Layout::kBlocks) EXPECT_EQ(view->column(2)->stride(), 0);
    ExpressionGenerator expressions(&gen, table->schema());
    for (int round = 0; round < 400; ++round) {
      auto bound = expressions.predicate(4).Bind(*table->schema());
      if (!bound.ok()) {
        ++unbindable;  // Arrow has no kernel for this combination.
        continue;
      }
      const auto text = bound->ToString();
      SCOPED_TRACE(text);
      auto predicate = ForwardPredicate::Make(*bound, view);
      std::vector<bool> expected(static_cast<size_t>(rows));
      try {
        for (int64_t row = 0; row < rows; ++row) {
          expected[row] = PerRow(*bound, *view, row);
        }
      } catch (const std::runtime_error &) {
        // Arrow fails the call (for example an invalid LIKE pattern); the
        // predicate must defer to it so behavior stays identical.
        EXPECT_FALSE(predicate.ok());
        ++arrow_errors;
        continue;
      }
      if (!predicate.ok()) {
        // The only accepted refusal: Bind cast a column with possible value
        // changes (for example uint64 against a negative literal).
        ASSERT_NE(predicate.status().message().find("may change values"),
                  std::string::npos)
            << predicate.status().ToString();
        ++lossy_casts;
        continue;
      }
      ++compiled;
      shared += SharesLayout(**predicate);
      contains += text.find("contain_") != std::string::npos;
      lengths += text.find("list_value_length") != std::string::npos;
      casts += text.find("cast(") != std::string::npos;
      for (int64_t row = 0; row < rows; ++row) {
        ASSERT_EQ((*predicate)->matches(row), expected[row]) << "row " << row;
      }
      EXPECT_FALSE((*predicate)->matches(rows));
    }
  }
  std::printf(
      "compiled=%zu (contain=%zu array_length=%zu cast=%zu shared_layout=%zu) "
      "arrow_errors=%zu lossy_casts=%zu unbindable=%zu\n",
      compiled, contains, lengths, casts, shared, arrow_errors, lossy_casts,
      unbindable);
  EXPECT_GT(compiled, 3000u);
  EXPECT_GT(contains, 200u);
  EXPECT_GT(lengths, 200u);
  EXPECT_GT(casts, 20u);
  EXPECT_GT(shared, 800u);
  EXPECT_GT(arrow_errors, 0u);
}

// The stile index cached with each forward column resolves every row of
// every layout, including tiny and empty chunks, to the chunk a binary
// search over the chunk ends finds.
TEST_F(ForwardPredicateTest, ChunkIndexMatchesBinarySearch) {
  Generator gen(17);
  for (int trial = 0; trial < 300; ++trial) {
    const int64_t rows = 1 + static_cast<int64_t>(gen.below(3000));
    arrow::Int32Builder builder;
    for (int64_t i = 0; i < rows; ++i) {
      ASSERT_TRUE(builder.Append(static_cast<int32_t>(i)).ok());
    }
    const auto array = Unwrap(builder.Finish());
    arrow::ArrayVector chunks;
    const int64_t typical = 1 + static_cast<int64_t>(gen.below(200));
    for (int64_t begin = 0; begin < rows;) {
      const int64_t size = std::min<int64_t>(
          rows - begin,
          gen.coin(0.2) ? static_cast<int64_t>(gen.below(4)) : typical);
      chunks.push_back(array->Slice(begin, size));
      begin += size;
    }
    auto forward = MakeColumn(
        std::make_shared<arrow::ChunkedArray>(chunks, arrow::int32()));
    ASSERT_NE(forward->stile_index(), nullptr);
    const auto &column = forward->stile_index();
    const auto &ends = column->chunk_ends();
    ASSERT_EQ(ends, forward->chunk_ends());
    for (int64_t row = 0; row < rows; ++row) {
      const auto chunk = column->chunk_index(row);
      ASSERT_EQ(chunk, static_cast<size_t>(
                           std::upper_bound(ends.begin(), ends.end(), row) -
                           ends.begin()))
          << "rows=" << rows << " row=" << row;
      const auto &values =
          static_cast<const arrow::Int32Array &>(*column->arrays()[chunk]);
      ASSERT_EQ(values.Value(row - column->chunk_begin(chunk)), row);
    }
  }
}

TEST_F(ForwardPredicateTest, VirtualRowIds) {
  auto view = MakeRowIdView(100);
  const auto row = cp::field_ref("row_id");
  arrow::UInt64Builder set;
  ASSERT_TRUE(set.AppendValues({3, 50, 99}).ok());
  const std::vector<std::pair<cp::Expression, std::function<bool(uint64_t)>>>
      cases = {
          {cp::greater(row, cp::literal(uint64_t{10})),
           [](uint64_t r) { return r > 10; }},
          {cp::less_equal(cp::literal(uint64_t{90}), row),
           [](uint64_t r) { return r >= 90; }},
          {cp::call("is_in", {row}, cp::SetLookupOptions(Unwrap(set.Finish()))),
           [](uint64_t r) { return r == 3 || r == 50 || r == 99; }},
          {cp::is_null(row), [](uint64_t) { return false; }},
      };
  for (const auto &[expression, expected] : cases) {
    auto bound = Unwrap(expression.Bind(*view->schema()));
    SCOPED_TRACE(bound.ToString());
    auto predicate = Unwrap(ForwardPredicate::Make(bound, view));
    for (uint64_t r = 0; r < 100; ++r) {
      EXPECT_EQ(predicate->matches(r), expected(r)) << r;
    }
  }
}

TEST_F(ForwardPredicateTest, UnsupportedExpressionsKeepArrow) {
  arrow::Int32Builder ints;
  ASSERT_TRUE(ints.AppendValues({1, 2, 3}).ok());
  arrow::UInt64Builder wide;
  ASSERT_TRUE(wide.AppendValues({1, 2, 3}).ok());
  arrow::StructBuilder pairs(
      arrow::struct_({arrow::field("x", arrow::int32())}),
      arrow::default_memory_pool(), {std::make_shared<arrow::Int32Builder>()});
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(pairs.Append().ok());
    ASSERT_TRUE(static_cast<arrow::Int32Builder *>(pairs.field_builder(0))
                    ->Append(i)
                    .ok());
  }
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("i", arrow::int32()),
                     arrow::field("u", arrow::uint64()),
                     arrow::field("s", pairs.type())}),
      {Unwrap(ints.Finish()), Unwrap(wide.Finish()), Unwrap(pairs.Finish())});
  auto view = MakeView(table);
  ASSERT_NE(view, nullptr);
  arrow::Int64Builder set;
  ASSERT_TRUE(set.Append(2).ok());
  const auto i = cp::field_ref("i");
  for (const auto &expression : std::vector<cp::Expression>{
           // uint64 against a negative literal: the cast may change values.
           cp::greater(cp::field_ref("u"), cp::literal(int64_t{-1})),
           cp::greater(cp::call("add", {i, cp::literal(int32_t{1})}),
                       cp::literal(int32_t{2})),
           cp::is_valid(cp::field_ref("s")),
           // Arrow casts a differently typed set.
           cp::call("is_in", {i}, cp::SetLookupOptions(Unwrap(set.Finish()))),
           cp::equal(i, i)}) {
    auto bound = Unwrap(expression.Bind(*table->schema()));
    SCOPED_TRACE(bound.ToString());
    auto predicate = ForwardPredicate::Make(bound, view);
    ASSERT_FALSE(predicate.ok());
    EXPECT_TRUE(predicate.status().IsNotImplemented());
  }
}

TEST_F(ForwardPredicateTest, ConcurrentCallsAndOwnedBuffers) {
  Generator gen(42);
  const int64_t rows = 5000;
  auto table = arrow::Table::Make(
      arrow::schema({arrow::field("s", arrow::utf8()),
                     arrow::field("x", arrow::float64()),
                     arrow::field("l", arrow::list(arrow::int64()))}),
      {gen.column(arrow::utf8(), rows, Layout::kBlocks),
       gen.column(arrow::float64(), rows, Layout::kUniform),
       gen.column(arrow::list(arrow::int64()), rows, Layout::kRandom)},
      rows);
  arrow::Int64Builder members;
  ASSERT_TRUE(members.AppendValues({-1, 0, 1}).ok());
  auto bound = Unwrap(
      cp::or_(cp::and_(cp::call("match_like", {cp::field_ref("s")},
                                cp::MatchSubstringOptions("%a_b%")),
                       cp::greater_equal(cp::field_ref("x"), cp::literal(0.0))),
              cp::call(sqlengine::kContainAny, {cp::field_ref("l")},
                       std::make_shared<ContainOp::Options>(ContainOp::Args{
                           Unwrap(members.Finish()), DataType::ARRAY_INT64})))
          .Bind(*table->schema()));
  auto view = MakeView(table);
  std::vector<bool> expected(rows);
  for (int64_t row = 0; row < rows; ++row) {
    expected[row] = PerRow(bound, *view, row);
  }
  auto predicate = Unwrap(ForwardPredicate::Make(bound, std::move(view)));
  table.reset();  // The predicate's view owns every buffer it reads.
  std::vector<std::thread> threads;
  std::vector<size_t> mismatches(8, 0);
  for (size_t t = 0; t < mismatches.size(); ++t) {
    threads.emplace_back([&, t] {
      for (int repeat = 0; repeat < 4; ++repeat) {
        for (int64_t row = 0; row < rows; ++row) {
          mismatches[t] += predicate->matches(row) != expected[row];
        }
      }
    });
  }
  for (auto &thread : threads) thread.join();
  for (const auto count : mismatches) EXPECT_EQ(count, 0u);
}

}  // namespace zvec::test
