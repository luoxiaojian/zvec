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

constexpr size_t kTailBytes = sizeof(int32_t);

float ScalarReference(const std::vector<int8_t> &lhs,
                      const std::vector<int8_t> &rhs, size_t dimension) {
  int64_t sum = 0;
  for (size_t i = 0; i < dimension; ++i) {
    const int difference = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    sum += difference * difference;
  }
  return static_cast<float>(sum);
}

IndexMetric::Pointer CreateUniformUint8Metric(size_t original_dimension) {
  auto metric = IndexFactory::CreateMetric("UniformUint8");
  if (!metric) return nullptr;

  IndexMeta meta(IndexMeta::DataType::DT_INT8, original_dimension + kTailBytes);
  ailego::Params params;
  params.set(UNIFORM_UINT8_METRIC_ORIGIN_METRIC_NAME,
             std::string("SquaredEuclidean"));
  if (metric->init(meta, params) != 0) return nullptr;
  return metric;
}

TEST(UniformUint8Metric, TurboBuildDistanceMatchesScalarExactly) {
  std::mt19937 generator(20260716);
  std::uniform_int_distribution<int> byte_distribution(-128, 127);

  for (size_t original_dimension :
       {1UL, 15UL, 16UL, 31UL, 63UL, 64UL, 65UL, 127UL, 128UL, 129UL, 1024UL,
        65536UL, 1048577UL}) {
    auto metric = CreateUniformUint8Metric(original_dimension);
    ASSERT_NE(nullptr, metric);
    auto distance = metric->distance();
    auto batch_distance = metric->batch_distance();
    ASSERT_TRUE(static_cast<bool>(distance));
    ASSERT_TRUE(static_cast<bool>(batch_distance));

    const size_t encoded_dimension = original_dimension + kTailBytes;
    std::vector<int8_t> lhs(encoded_dimension, 0);
    std::vector<int8_t> rhs(encoded_dimension, 0);
    for (size_t i = 0; i < original_dimension; ++i) {
      lhs[i] = static_cast<int8_t>(byte_distribution(generator));
      rhs[i] = static_cast<int8_t>(byte_distribution(generator));
    }

    float actual = 0.0f;
    distance(lhs.data(), rhs.data(), encoded_dimension, &actual);
    EXPECT_EQ(ScalarReference(lhs, rhs, original_dimension), actual)
        << "dimension=" << original_dimension;

    std::fill(lhs.begin(), lhs.begin() + original_dimension,
              static_cast<int8_t>(-128));
    std::fill(rhs.begin(), rhs.begin() + original_dimension,
              static_cast<int8_t>(127));
    distance(lhs.data(), rhs.data(), encoded_dimension, &actual);
    EXPECT_EQ(ScalarReference(lhs, rhs, original_dimension), actual)
        << "extreme dimension=" << original_dimension;

    const void *vectors[] = {lhs.data(), rhs.data()};
    float batch_results[2] = {};
    batch_distance(vectors, rhs.data(), 2, encoded_dimension, batch_results);
    EXPECT_EQ(ScalarReference(lhs, rhs, original_dimension), batch_results[0]);
    EXPECT_EQ(0.0f, batch_results[1]);
  }
}

}  // namespace
}  // namespace zvec::core
