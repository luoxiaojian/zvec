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

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include "metric/metric_params.h"

namespace zvec::core {
namespace {

float ScalarReference(const int8_t *lhs, const int8_t *rhs, size_t dimension) {
  int32_t result = 0;
  for (size_t i = 0; i < dimension; ++i) {
    const int difference = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    result += difference * difference;
  }
  return static_cast<float>(result);
}

IndexMetric::Pointer CreateUniformInt8Metric(size_t dimension) {
  auto metric = IndexFactory::CreateMetric("UniformInt8");
  if (!metric) return nullptr;

  IndexMeta meta(IndexMeta::DataType::DT_INT8, dimension);
  ailego::Params params;
  params.set(UNIFORM_INT8_METRIC_ORIGIN_METRIC_NAME,
             std::string("SquaredEuclidean"));
  if (metric->init(meta, params) != 0) return nullptr;
  return metric;
}

TEST(UniformInt8Metric, PairAndEveryBatchRemainderMatchScalarExactly) {
  std::mt19937 generator(20260825);
  std::uniform_int_distribution<int> value_distribution(0, 127);

  for (size_t dimension :
       {1UL, 31UL, 63UL, 64UL, 65UL, 127UL, 128UL, 129UL, 960UL, 1024UL}) {
    auto metric = CreateUniformInt8Metric(dimension);
    ASSERT_NE(nullptr, metric);
    auto distance = metric->distance();
    auto batch_distance = metric->batch_distance();
    ASSERT_TRUE(static_cast<bool>(distance));
    ASSERT_TRUE(static_cast<bool>(batch_distance));

    constexpr size_t vector_count = 11;
    std::vector<int8_t> query(dimension);
    std::vector<std::vector<int8_t>> records(vector_count,
                                             std::vector<int8_t>(dimension));
    std::vector<const void *> vectors(vector_count);
    std::vector<float> expected(vector_count);
    for (size_t d = 0; d < dimension; ++d) {
      query[d] = static_cast<int8_t>(value_distribution(generator));
    }
    for (size_t i = 0; i < vector_count; ++i) {
      for (size_t d = 0; d < dimension; ++d) {
        records[i][d] = static_cast<int8_t>(value_distribution(generator));
      }
      vectors[i] = records[i].data();
      expected[i] = ScalarReference(records[i].data(), query.data(), dimension);

      float actual = 0.0f;
      distance(records[i].data(), query.data(), dimension, &actual);
      EXPECT_EQ(expected[i], actual)
          << "dimension=" << dimension << ", vector=" << i;
    }

    for (size_t count = 1; count <= vector_count; ++count) {
      std::vector<float> actual(count, 0.0f);
      batch_distance(vectors.data(), query.data(), count, dimension,
                     actual.data(), nullptr);
      EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin()))
          << "dimension=" << dimension << ", count=" << count;
    }
  }
}

}  // namespace
}  // namespace zvec::core
