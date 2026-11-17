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
#include <future>
#include <limits>
#include <stdexcept>
#include <arrow/acero/api.h>
#include <arrow/api.h>
#include <arrow/array/concatenate.h>
#include <arrow/compute/api.h>
#include <arrow/compute/registry.h>
#include <arrow/util/bit_util.h>
#include <gtest/gtest.h>
#include "ailego/internal/cpu_features.h"
#include "db/sqlengine/common/util.h"
#include "db/sqlengine/planner/op_register.h"
#include "db/sqlengine/planner/ops/bitmap_compare.h"
#include "db/sqlengine/planner/ops/contain_op.h"

namespace zvec::sqlengine {
namespace {
namespace bi = bitmap_internal;

template <typename T>
T value(arrow::Result<T> result) {
  if (!result.ok()) throw std::runtime_error(result.status().ToString());
  return result.MoveValueUnsafe();
}

void check(const arrow::Status &status) {
  if (!status.ok()) throw std::runtime_error(status.ToString());
}

std::vector<bi::Int32BitmapKernel> available_kernels() {
  std::vector<bi::Int32BitmapKernel> kernels{
      bi::select_int32_bitmap_kernel(false, false)};
  const auto selected = bi::int32_bitmap_kernel().isa;
  if (selected != bi::Int32BitmapIsa::kScalar)
    kernels.push_back(bi::select_int32_bitmap_kernel(true, false));
  if (selected == bi::Int32BitmapIsa::kAvx512)
    kernels.push_back(bi::select_int32_bitmap_kernel(true, true));
  return kernels;
}

std::shared_ptr<arrow::ChunkedArray> chunks(
    const std::shared_ptr<arrow::Array> &array,
    const std::vector<int64_t> &lengths) {
  arrow::ArrayVector parts;
  int64_t offset = 0;
  for (const auto length : lengths) {
    parts.push_back(array->Slice(offset, length));
    offset += length;
  }
  if (offset != array->length()) throw std::runtime_error("bad test chunks");
  return std::make_shared<arrow::ChunkedArray>(std::move(parts), array->type());
}

class ForwardBitmapTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(cp::Initialize().ok());
    ASSERT_TRUE(OpRegister::register_ops().ok());
  }

  std::shared_ptr<arrow::Table> table() {
    arrow::Int32Builder i32;
    arrow::Int64Builder i64;
    arrow::StringBuilder text;
    arrow::BooleanBuilder flag;
    for (int i = 0; i < 514; ++i) {
      check(i % 11 == 0 ? i32.AppendNull() : i32.Append(i - 200));
      check(i % 13 == 0 ? i64.AppendNull() : i64.Append(int64_t{i} << 32));
      check(i % 7 == 0 ? text.AppendNull()
                       : text.Append(i % 2 == 0 ? "alpha" : "beta"));
      check(i % 3 == 0 ? flag.AppendNull() : flag.Append(i % 2 == 0));
    }
    return arrow::Table::Make(
        arrow::schema({arrow::field("i32", arrow::int32()),
                       arrow::field("i64", arrow::int64()),
                       arrow::field("text", arrow::utf8()),
                       arrow::field("flag", arrow::boolean())}),
        {chunks(value(i32.Finish())->Slice(1), {3, 0, 17, 63, 0, 430}),
         chunks(value(i64.Finish())->Slice(1), {7, 250, 256}),
         chunks(value(text.Finish())->Slice(1), {4, 100, 409}),
         chunks(value(flag.Finish())->Slice(1), {300, 213})});
  }

  void expect_acero_parity(const std::shared_ptr<arrow::Table> &input,
                           const cp::Expression &expression) {
    auto actual = value(evaluate_forward_bitmap(*input, expression));
    auto plan = arrow::acero::Declaration::Sequence(
        {{"table_source", arrow::acero::TableSourceNodeOptions{input}},
         {"project",
          arrow::acero::ProjectNodeOptions{{expression}, {"matches"}}}});
    auto expected =
        value(arrow::acero::DeclarationToTable(std::move(plan), false));
    EXPECT_TRUE(std::make_shared<arrow::ChunkedArray>(actual)->Equals(
        *expected->column(0)));
    EXPECT_EQ(actual->length(), input->num_rows());
    EXPECT_TRUE(actual->ValidateFull().ok());
  }
};

TEST_F(ForwardBitmapTest, DispatchUsesAilegoAndPrefersAvailableWideKernel) {
  EXPECT_EQ(bi::select_int32_bitmap_kernel(false, false).isa,
            bi::Int32BitmapIsa::kScalar);
  EXPECT_EQ(bi::select_int32_bitmap_kernel(false, true).isa,
            bi::Int32BitmapIsa::kScalar);
  const auto maximum = bi::select_int32_bitmap_kernel(true, true).isa;
  if (maximum != bi::Int32BitmapIsa::kScalar) {
    EXPECT_EQ(maximum, bi::Int32BitmapIsa::kAvx512);
    EXPECT_EQ(bi::select_int32_bitmap_kernel(true, false).isa,
              bi::Int32BitmapIsa::kAvx2);
  }
  const auto &selected = bi::int32_bitmap_kernel();
  EXPECT_EQ(&selected, &bi::int32_bitmap_kernel());
  const auto &features = ailego::internal::CpuFeatures::static_flags_;
  if (selected.isa != bi::Int32BitmapIsa::kScalar) {
    EXPECT_TRUE(features.AVX2);
  }
  if (selected.isa == bi::Int32BitmapIsa::kAvx512) {
    EXPECT_TRUE(features.AVX512F);
  }
}

TEST_F(ForwardBitmapTest, PackedSlicesTailsAndNeighbourBitsAcrossIsas) {
  std::vector<int64_t> lengths;
  for (int64_t n = 0; n <= 65; ++n) lengths.push_back(n);
  for (int64_t n : {127, 128, 129, 255, 257}) lengths.push_back(n);
  for (const auto &kernel : available_kernels()) {
    SCOPED_TRACE(bi::int32_bitmap_isa_name(kernel.isa));
    for (const auto n : lengths) {
      for (int input_offset = 0; input_offset < 8; ++input_offset) {
        std::vector<int32_t> input(std::max<int64_t>(1, n + input_offset));
        for (size_t i = 0; i < input.size(); ++i)
          input[i] = static_cast<int32_t>(i * UINT32_C(2654435761));
        if (n >= 2) {
          input[input_offset] = INT32_MIN;
          input[input_offset + 1] = INT32_MAX;
        }
        for (const int32_t threshold :
             {INT32_MIN, -1, 0, 536870911, INT32_MAX}) {
          for (int bit_offset = 0; bit_offset < 16; ++bit_offset) {
            std::vector<uint8_t> bits((bit_offset + n + 7) / 8 + 1, 0xa5);
            const auto before = bits;
            kernel.pack(input.data() + input_offset, threshold, n, bits.data(),
                        bit_offset);
            for (int64_t b = 0; b < static_cast<int64_t>(bits.size() * 8);
                 ++b) {
              const bool expected =
                  b >= bit_offset && b < bit_offset + n
                      ? input[input_offset + b - bit_offset] > threshold
                      : arrow::bit_util::GetBit(before.data(), b);
              ASSERT_EQ(arrow::bit_util::GetBit(bits.data(), b), expected)
                  << "length=" << n << " input_offset=" << input_offset
                  << " bit_offset=" << bit_offset << " bit=" << b;
            }
          }
        }
      }
    }
  }
}

TEST_F(ForwardBitmapTest, ArrowKernelPreservesNullsAndArgumentShapes) {
  for (const auto &kernel : available_kernels()) {
    auto function = value(bi::make_int32_bitmap_function(kernel));
    for (int null_pattern = 0; null_pattern < 3; ++null_pattern) {
      arrow::Int32Builder builder;
      for (int i = 0; i < 260; ++i)
        check(null_pattern == 2 || (null_pattern == 1 && i % 3 == 0)
                  ? builder.AppendNull()
                  : builder.Append(i % 2 ? INT32_MIN : i));
      auto array = value(builder.Finish())->Slice(3, 257);
      auto column = chunks(array, {3, 0, 17, 63, 174});
      const arrow::Datum scalar(int32_t{48});
      const arrow::Datum null_scalar(arrow::MakeNullScalar(arrow::int32()));
      for (const auto &args :
           std::vector<std::vector<arrow::Datum>>{{array, scalar},
                                                  {scalar, array},
                                                  {array, array},
                                                  {array, null_scalar},
                                                  {column, scalar},
                                                  {scalar, scalar}}) {
        auto expected = value(cp::CallFunction("greater", args));
        auto actual = value(function->Execute(args, nullptr, nullptr));
        EXPECT_TRUE(actual.Equals(expected));
      }
    }
  }
}

TEST_F(ForwardBitmapTest,
       WholeExpressionMatchesAceroWithUnalignedColumnChunks) {
  const auto input = table();
  const auto greater =
      cp::greater(cp::field_ref("i32"), cp::literal(int32_t{48}));
  const auto less =
      cp::less(cp::field_ref("i64"), cp::literal(INT64_C(300) << 32));
  for (const auto &expression : std::vector<cp::Expression>{
           greater, cp::not_(greater), cp::and_(greater, less),
           cp::or_(greater, less),
           cp::call("and_kleene", {greater, cp::field_ref("flag")}),
           cp::call("or_kleene", {greater, cp::field_ref("flag")}),
           cp::call("is_null", {cp::field_ref("i32")}),
           cp::greater(cp::field_ref("i32"), cp::literal(INT64_C(2147483648))),
           cp::equal(cp::field_ref("text"), cp::literal("alpha")),
           cp::call("match_substring", {cp::field_ref("text")},
                    cp::MatchSubstringOptions("alp")),
           cp::field_ref("flag"), cp::literal(true), cp::literal(false),
           cp::literal(arrow::MakeNullScalar(arrow::boolean()))}) {
    SCOPED_TRACE(expression.ToString());
    expect_acero_parity(input, expression);
  }
  EXPECT_EQ(input->column(0)->num_chunks(), 6);
  EXPECT_EQ(input->column(0)->chunk(0)->offset(), 1);
  EXPECT_EQ(input->column(0)->chunk(2)->offset(), 4);
}

TEST_F(ForwardBitmapTest, EmptyTablesAndConstantPredicates) {
  const auto input = table();
  for (const int64_t rows : {0, 17}) {
    const auto no_fields =
        arrow::Table::Make(arrow::schema({}), arrow::ArrayVector{}, rows);
    for (const auto &expr :
         {cp::literal(true), cp::literal(false),
          cp::literal(arrow::MakeNullScalar(arrow::boolean()))}) {
      auto actual = value(evaluate_forward_bitmap(*no_fields, expr));
      EXPECT_EQ(actual->length(), rows);
      for (int64_t i = 0; i < rows; ++i) {
        EXPECT_EQ(actual->IsValid(i), expr.literal()->scalar()->is_valid);
        if (actual->IsValid(i)) {
          EXPECT_EQ(actual->Value(i),
                    expr.literal()->scalar_as<arrow::BooleanScalar>().value);
        }
      }
    }
  }
  auto empty = value(evaluate_forward_bitmap(
      *input->Slice(0, 0),
      cp::greater(cp::field_ref("i32"), cp::literal(int32_t{0}))));
  EXPECT_EQ(empty->length(), 0);
}

TEST_F(ForwardBitmapTest, NormalTypeResolutionAndGlobalRegistryArePreserved) {
  const auto input = table();
  auto original = value(cp::GetFunctionRegistry()->GetFunction("greater"));
  cp::ExecContext context;
  auto bound = value(bind_forward_bitmap_expression(
      cp::greater(cp::field_ref("i32"), cp::literal(int32_t{48})),
      *input->schema(), &context));
  ASSERT_NE(bound.call(), nullptr);
  if (bi::int32_bitmap_kernel().isa != bi::Int32BitmapIsa::kScalar) {
    EXPECT_NE(bound.call()->function, original);
  }
  auto widened = value(bind_forward_bitmap_expression(
      cp::greater(cp::field_ref("i32"), cp::literal(INT64_C(2147483648))),
      *input->schema(), &context));
  ASSERT_NE(widened.call(), nullptr);
  EXPECT_EQ(widened.call()->function, original);
  EXPECT_EQ(value(cp::GetFunctionRegistry()->GetFunction("greater")), original);
  EXPECT_FALSE(evaluate_forward_bitmap(*input, cp::field_ref("i32")).ok());
  EXPECT_FALSE(
      evaluate_forward_bitmap(
          *input, cp::greater(cp::field_ref("missing"), cp::literal(0)))
          .ok());
}

TEST_F(ForwardBitmapTest, CustomContainKernelOutputIsCompacted) {
  auto values = std::make_shared<arrow::Int32Builder>();
  arrow::ListBuilder lists(arrow::default_memory_pool(), values);
  for (int i = 0; i < 57; ++i) {
    check(lists.Append());
    check(values->Append(i % 3));
    check(values->Append(i % 5));
  }
  auto array = value(lists.Finish());
  auto input =
      arrow::Table::Make(arrow::schema({arrow::field("items", array->type())}),
                         {chunks(array, {3, 0, 17, 37})});
  arrow::Int32Builder wanted;
  check(wanted.Append(1));
  auto options = std::make_shared<ContainOp::Options>(
      ContainOp::Args{value(wanted.Finish()), DataType::ARRAY_INT32});
  for (const auto &name : {kContainAny, kContainAll})
    expect_acero_parity(input,
                        cp::call(name, {cp::field_ref("items")}, options));
}

TEST_F(ForwardBitmapTest,
       StreamingReaderRetainsCapturedPrefixAndRejectsShortRead) {
  auto input = table();
  const auto expression = cp::call(
      "or_kleene", {cp::greater(cp::field_ref("i32"), cp::literal(int32_t{48})),
                    cp::field_ref("flag")});
  for (int64_t rows : {0, 1, 17, 257, 513}) {
    arrow::TableBatchReader reader(input);
    reader.set_chunksize(19);
    const auto actual =
        value(evaluate_forward_bitmap(reader, expression, rows));
    const auto expected =
        value(evaluate_forward_bitmap(*input->Slice(0, rows), expression));
    EXPECT_TRUE(actual->Equals(*expected));
  }
  arrow::TableBatchReader short_reader(input);
  EXPECT_FALSE(evaluate_forward_bitmap(short_reader, expression, 514).ok());
  arrow::TableBatchReader invalid_reader(input);
  EXPECT_FALSE(evaluate_forward_bitmap(invalid_reader, expression, -1).ok());
}

TEST_F(ForwardBitmapTest, FloatingAndUnsignedComparisonsUseArrowSemantics) {
  arrow::DoubleBuilder floating;
  check(floating.AppendValues({-0.0, 0.0, -1.0, 1.0,
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}));
  check(floating.AppendNull());
  arrow::UInt32Builder integer;
  check(integer.AppendValues(
      {0, 1, UINT32_MAX, UINT32_C(2147483648), 42, 100, UINT32_MAX - 1}));
  check(integer.AppendNull());
  auto input =
      arrow::Table::Make(arrow::schema({arrow::field("f", arrow::float64()),
                                        arrow::field("u", arrow::uint32())}),
                         {chunks(value(floating.Finish()), {1, 0, 3, 4}),
                          chunks(value(integer.Finish()), {3, 5})});
  expect_acero_parity(input, cp::greater(cp::field_ref("f"), cp::literal(0.0)));
  expect_acero_parity(
      input, cp::not_(cp::greater(cp::field_ref("f"), cp::literal(0.0))));
  expect_acero_parity(input, cp::greater(cp::field_ref("u"),
                                         cp::literal(UINT32_C(2147483647))));
}

TEST_F(ForwardBitmapTest, ConcurrentEvaluationsHaveIndependentExpressionState) {
  auto input = table();
  auto expression =
      cp::not_(cp::greater(cp::field_ref("i32"), cp::literal(int32_t{48})));
  auto expected = value(evaluate_forward_bitmap(*input, expression));
  std::vector<std::future<bool>> tasks;
  for (int t = 0; t < 8; ++t)
    tasks.push_back(std::async(std::launch::async, [&] {
      for (int i = 0; i < 32; ++i) {
        auto actual = value(evaluate_forward_bitmap(*input, expression));
        if (!actual->Equals(*expected)) return false;
      }
      return true;
    }));
  for (auto &task : tasks) EXPECT_TRUE(task.get());
}

}  // namespace
}  // namespace zvec::sqlengine
