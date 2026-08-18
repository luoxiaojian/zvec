// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/turbo/turbo.h>
#include "metric/metric_params.h"

namespace zvec::core {
namespace {

float ScalarDistance(const uint8_t *lhs, const uint8_t *rhs, size_t bytes) {
  int64_t sum = 0;
  for (size_t i = 0; i < bytes; ++i) {
    const int low = static_cast<int>(lhs[i] & 15U) -
                    static_cast<int>(rhs[i] & 15U);
    const int high = static_cast<int>(lhs[i] >> 4U) -
                     static_cast<int>(rhs[i] >> 4U);
    sum += low * low + high * high;
  }
  return static_cast<float>(sum);
}

IndexMetric::Pointer CreateMetric(size_t encoded_dimension) {
  auto metric = IndexFactory::CreateMetric("UniformUint4");
  if (!metric) return nullptr;
  IndexMeta meta(IndexMeta::DataType::DT_INT8, encoded_dimension);
  ailego::Params params;
  params.set(UNIFORM_UINT4_METRIC_ORIGIN_METRIC_NAME,
             std::string("SquaredEuclidean"));
  return metric->init(meta, params) == 0 ? metric : nullptr;
}

TEST(UniformUint4Metric, PairAndBatchMatchScalarExactly) {
  std::mt19937 generator(20260807);
  std::uniform_int_distribution<int> bytes(0, 255);
  for (const size_t logical_dimension : {128UL, 256UL, 1024UL, 65536UL}) {
    const size_t encoded_dimension = logical_dimension / 2U;
    auto metric = CreateMetric(encoded_dimension);
    ASSERT_NE(nullptr, metric);
    auto distance = metric->distance();
    auto batch_distance = metric->batch_distance();
    ASSERT_TRUE(static_cast<bool>(distance));
    ASSERT_TRUE(static_cast<bool>(batch_distance));
    EXPECT_FALSE(metric->use_stored_query_for_build());

    constexpr size_t count = 7;
    std::vector<uint8_t> query(encoded_dimension);
    std::vector<std::vector<uint8_t>> rows(
        count, std::vector<uint8_t>(encoded_dimension));
    std::vector<const void *> pointers(count);
    std::vector<float> expected(count);
    std::vector<float> actual(count);
    for (auto &value : query) value = static_cast<uint8_t>(bytes(generator));
    for (size_t i = 0; i < count; ++i) {
      for (auto &value : rows[i]) value = static_cast<uint8_t>(bytes(generator));
      pointers[i] = rows[i].data();
      expected[i] = ScalarDistance(rows[i].data(), query.data(),
                                   encoded_dimension);
      float pair = 0.0f;
      distance(rows[i].data(), query.data(), encoded_dimension, &pair);
      EXPECT_EQ(expected[i], pair);
    }
    batch_distance(pointers.data(), query.data(), count, encoded_dimension,
                   actual.data(), nullptr);
    EXPECT_EQ(expected, actual) << "logical_dimension=" << logical_dimension;
  }
}

TEST(UniformUint4Metric, QuantizeMatchesReimplPackingAndPadding) {
  auto quantize = turbo::get_uniform_uint4_quantize_func(
      turbo::DataType::kInt4);
  if (!quantize) GTEST_SKIP() << "AVX-512 VNNI is unavailable";

  constexpr size_t dimension = 131;
  constexpr size_t encoded_dimension = 128;
  std::vector<float> input(dimension);
  for (size_t i = 0; i < dimension; ++i) {
    input[i] = -5.0f + static_cast<float>(i % 31U) * 0.5f;
  }
  std::vector<uint8_t> actual(encoded_dimension, 0xff);
  std::vector<uint8_t> expected(encoded_dimension, 0);
  constexpr float minimum = -3.25f;
  constexpr float range = 11.5f;
  constexpr float almost_half = 0.4999999701976776123046875f;
  for (size_t d = 0; d < dimension; ++d) {
    float normalized = (input[d] - minimum) / range;
    normalized = std::min(1.0f, std::max(0.0f, normalized));
    const auto code = static_cast<uint8_t>(
        static_cast<int>(normalized * 15.0f + almost_half));
    expected[d >> 1U] |= static_cast<uint8_t>(code << (4U * (d & 1U)));
  }
  quantize(input.data(), dimension, minimum, range, actual.data());
  EXPECT_EQ(expected, actual);
}

TEST(Fp16RefineTurbo, HomogeneousBatchMatchesScalarReference) {
  auto batch = turbo::get_batch_distance_func(
      turbo::MetricType::kSquaredEuclidean, turbo::DataType::kFp16,
      turbo::QuantizeType::kDefault);
  if (!batch) GTEST_SKIP() << "AVX-512F/F16C is unavailable";

  std::mt19937 generator(20260807);
  std::uniform_real_distribution<float> values(-20.0f, 20.0f);
  for (const size_t dimension : {1UL, 17UL, 128UL, 131UL}) {
    constexpr size_t count = 7;
    std::vector<float> query_fp32(dimension);
    std::vector<ailego::Float16> query(dimension);
    std::vector<std::vector<ailego::Float16>> rows(
        count, std::vector<ailego::Float16>(dimension));
    std::vector<const void *> pointers(count);
    std::vector<float> expected(count, 0.0f);
    std::vector<float> actual(count, 0.0f);
    for (float &value : query_fp32) value = values(generator);
    ailego::FloatHelper::ToFP16(
        query_fp32.data(), dimension,
        reinterpret_cast<uint16_t *>(query.data()));
    for (size_t i = 0; i < count; ++i) {
      pointers[i] = rows[i].data();
      for (size_t d = 0; d < dimension; ++d) {
        rows[i][d] = values(generator);
        const float delta =
            static_cast<float>(query[d]) - static_cast<float>(rows[i][d]);
        expected[i] += delta * delta;
      }
    }
    batch(pointers.data(), query.data(), count, dimension, actual.data(),
          nullptr);
    for (size_t i = 0; i < count; ++i) {
      EXPECT_NEAR(expected[i], actual[i],
                  std::max(1e-3f, expected[i] * 2e-6f))
          << "dimension=" << dimension << " row=" << i;
    }
  }
}

TEST(Fp16RefineTurbo, QueryConversionMatchesFloatHelper) {
  auto convert = turbo::get_fp32_to_fp16_convert_func();
  if (!convert) GTEST_SKIP() << "AVX-512F/F16C is unavailable";

  std::mt19937 generator(20260809);
  std::uniform_real_distribution<float> values(-100.0f, 100.0f);
  for (const size_t dimension :
       {1UL, 7UL, 8UL, 15UL, 16UL, 17UL, 31UL, 32UL, 128UL, 131UL}) {
    std::vector<float> input(dimension);
    std::vector<uint16_t> expected(dimension);
    std::vector<uint16_t> actual(dimension);
    for (float &value : input) value = values(generator);
    ailego::FloatHelper::ToFP16(input.data(), dimension, expected.data());
    convert(input.data(), dimension, actual.data());
    EXPECT_EQ(expected, actual) << "dimension=" << dimension;
  }
}

}  // namespace
}  // namespace zvec::core
