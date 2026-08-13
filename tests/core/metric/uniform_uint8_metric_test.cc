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
#include <cstring>
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

float ScalarRawQueryReference(const std::vector<int8_t> &stored,
                              const std::vector<uint8_t> &raw_query,
                              size_t dimension) {
  int64_t sum = 0;
  for (size_t i = 0; i < dimension; ++i) {
    const int difference =
        static_cast<int>(stored[i]) - (static_cast<int>(raw_query[i]) - 128);
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

void SetStoredNorm(std::vector<int8_t> *record, size_t dimension) {
  int64_t sum_sq = 0;
  for (size_t d = 0; d < dimension; ++d) {
    const int raw = static_cast<int>((*record)[d]) + 128;
    sum_sq += raw * raw;
  }
  const int32_t encoded_sum_sq = static_cast<int32_t>(sum_sq);
  std::memcpy(record->data() + dimension, &encoded_sum_sq,
              sizeof(encoded_sum_sq));
}

std::vector<uint8_t> PrepareStoredQuery(const IndexMetric::Pointer &metric,
                                        const std::vector<int8_t> &stored) {
  auto preprocess = metric->get_query_preprocess_func();
  EXPECT_TRUE(static_cast<bool>(preprocess));
  std::vector<uint8_t> query(stored.size());
  std::memcpy(query.data(), stored.data(), stored.size());
  if (preprocess) {
    preprocess(query.data(), query.size());
  }
  return query;
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
    auto stored_batch_distance = metric->stored_batch_distance();
    ASSERT_TRUE(static_cast<bool>(distance));
    ASSERT_TRUE(static_cast<bool>(batch_distance));
    ASSERT_TRUE(static_cast<bool>(stored_batch_distance));

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
    SetStoredNorm(&lhs, original_dimension);
    SetStoredNorm(&rhs, original_dimension);
    distance(lhs.data(), rhs.data(), encoded_dimension, &actual);
    EXPECT_EQ(ScalarReference(lhs, rhs, original_dimension), actual)
        << "extreme dimension=" << original_dimension;

    const void *vectors[] = {lhs.data(), rhs.data()};
    float batch_results[2] = {};
    const auto query = PrepareStoredQuery(metric, rhs);
    batch_distance(vectors, query.data(), 2, encoded_dimension, batch_results,
                   nullptr);
    const float expected_lhs = ScalarReference(lhs, rhs, original_dimension);
    const float expected_rhs = 0.0f;
    EXPECT_EQ(expected_lhs, batch_results[0]);
    EXPECT_EQ(expected_rhs, batch_results[1]);

    float stored_batch_results[2] = {};
    stored_batch_distance(vectors, rhs.data(), 2, encoded_dimension,
                          stored_batch_results, nullptr);
    EXPECT_EQ(expected_lhs, stored_batch_results[0]);
    EXPECT_EQ(expected_rhs, stored_batch_results[1]);
  }
}

TEST(UniformUint8Metric,
     PreprocessedStoredBatchMatchesPairwiseDistanceExactly) {
  std::mt19937 generator(20260730);
  std::uniform_int_distribution<int> byte_distribution(-128, 127);

  for (size_t original_dimension :
       {1UL, 31UL, 32UL, 63UL, 64UL, 127UL, 128UL, 129UL, 1024UL, 300001UL}) {
    auto metric = CreateUniformUint8Metric(original_dimension);
    ASSERT_NE(nullptr, metric);
    auto distance = metric->distance();
    auto batch_distance = metric->batch_distance();
    ASSERT_TRUE(static_cast<bool>(distance));
    ASSERT_TRUE(static_cast<bool>(batch_distance));

    const size_t encoded_dimension = original_dimension + kTailBytes;
    constexpr size_t vector_count = 7;
    std::vector<int8_t> query(encoded_dimension, 0);
    std::vector<std::vector<int8_t>> records(
        vector_count, std::vector<int8_t>(encoded_dimension, 0));
    std::vector<const void *> pointers(vector_count);
    std::vector<float> expected(vector_count, 0.0f);
    std::vector<float> actual(vector_count, 0.0f);

    for (size_t d = 0; d < original_dimension; ++d) {
      query[d] = static_cast<int8_t>(byte_distribution(generator));
    }
    SetStoredNorm(&query, original_dimension);
    for (size_t i = 0; i < vector_count; ++i) {
      for (size_t d = 0; d < original_dimension; ++d) {
        records[i][d] = static_cast<int8_t>(byte_distribution(generator));
      }
      SetStoredNorm(&records[i], original_dimension);
      pointers[i] = records[i].data();
      distance(records[i].data(), query.data(), encoded_dimension,
               &expected[i]);
    }

    const auto prepared_query = PrepareStoredQuery(metric, query);
    batch_distance(pointers.data(), prepared_query.data(), vector_count,
                   encoded_dimension, actual.data(), nullptr);
    EXPECT_EQ(expected, actual) << "dimension=" << original_dimension;
  }
}

TEST(UniformUint8Metric, ExactRawQueryBatchMatchesScalarDistance) {
  std::mt19937 generator(20260731);
  std::uniform_int_distribution<int> raw_byte_distribution(0, 255);

  for (size_t original_dimension :
       {1UL, 31UL, 32UL, 63UL, 64UL, 127UL, 128UL, 129UL, 1024UL, 300001UL}) {
    auto metric = CreateUniformUint8Metric(original_dimension);
    ASSERT_NE(nullptr, metric);
    auto query_metric = metric->query_metric();
    ASSERT_NE(nullptr, query_metric);
    auto distance = query_metric->distance();
    auto batch_distance = query_metric->batch_distance();
    ASSERT_TRUE(static_cast<bool>(distance));
    ASSERT_TRUE(static_cast<bool>(batch_distance));

    const size_t encoded_dimension = original_dimension + kTailBytes;
    constexpr size_t vector_count = 7;
    std::vector<int8_t> stored_query(encoded_dimension, 0);
    std::vector<uint8_t> expected_raw_query(original_dimension, 0);
    std::vector<std::vector<int8_t>> records(
        vector_count, std::vector<int8_t>(encoded_dimension, 0));
    std::vector<const void *> pointers(vector_count);
    std::vector<float> expected(vector_count, 0.0f);
    std::vector<float> actual(vector_count, 0.0f);

    int64_t query_sum_sq = 0;
    int64_t query_sum = 0;
    for (size_t d = 0; d < original_dimension; ++d) {
      const int raw = raw_byte_distribution(generator);
      expected_raw_query[d] = static_cast<uint8_t>(raw);
      stored_query[d] = static_cast<int8_t>(raw - 128);
      query_sum_sq += raw * raw;
      query_sum += raw;
    }
    const int32_t stored_query_tail = static_cast<int32_t>(query_sum_sq);
    std::memcpy(stored_query.data() + original_dimension, &stored_query_tail,
                sizeof(stored_query_tail));
    const auto query = PrepareStoredQuery(metric, stored_query);
    EXPECT_TRUE(std::equal(expected_raw_query.begin(), expected_raw_query.end(),
                           query.begin()))
        << "dimension=" << original_dimension;
    int32_t actual_query_tail = 0;
    std::memcpy(&actual_query_tail, query.data() + original_dimension,
                sizeof(actual_query_tail));
    EXPECT_EQ(static_cast<int32_t>(query_sum_sq - 256 * query_sum),
              actual_query_tail)
        << "dimension=" << original_dimension;
    for (size_t i = 0; i < vector_count; ++i) {
      int64_t record_sum_sq = 0;
      for (size_t d = 0; d < original_dimension; ++d) {
        const int raw_value = raw_byte_distribution(generator);
        records[i][d] = static_cast<int8_t>(raw_value - 128);
        record_sum_sq += raw_value * raw_value;
      }
      if (original_dimension <= 8192) {
        const int32_t encoded_record_sum_sq =
            static_cast<int32_t>(record_sum_sq);
        std::memcpy(records[i].data() + original_dimension,
                    &encoded_record_sum_sq, sizeof(encoded_record_sum_sq));
      }
      pointers[i] = records[i].data();
      expected[i] =
          ScalarRawQueryReference(records[i], query, original_dimension);
      float scalar_actual = 0.0f;
      distance(records[i].data(), query.data(), encoded_dimension,
               &scalar_actual);
      EXPECT_EQ(expected[i], scalar_actual)
          << "dimension=" << original_dimension << ", vector=" << i;
    }

    batch_distance(pointers.data(), query.data(), vector_count,
                   encoded_dimension, actual.data(), nullptr);
    EXPECT_EQ(expected, actual) << "dimension=" << original_dimension;
  }
}

TEST(UniformUint8Metric, ExtraValuesQueryDistanceMatchesInlineRecordExactly) {
  constexpr size_t original_dimension = 128;
  constexpr size_t encoded_dimension = original_dimension + kTailBytes;
  constexpr size_t vector_count = 11;
  constexpr size_t extra_values_count = 37;

  auto metric = CreateUniformUint8Metric(original_dimension);
  ASSERT_NE(nullptr, metric);
  auto query_metric = metric->query_metric();
  ASSERT_NE(nullptr, query_metric);

  auto batch_distance = query_metric->batch_distance();
  ASSERT_TRUE(static_cast<bool>(batch_distance));
  EXPECT_EQ(kTailBytes, query_metric->extra_values_size_per_vector());

  std::mt19937 generator(20260721);
  std::uniform_int_distribution<int> raw_byte_distribution(0, 255);
  std::vector<std::vector<int8_t>> records(
      vector_count, std::vector<int8_t>(encoded_dimension, 0));
  std::vector<int32_t> extra_values(extra_values_count, 0);
  std::vector<const void *> extra_value_ptrs(vector_count);
  std::vector<uint32_t> extra_value_ids = {31, 2,  19, 7,  13, 29,
                                           3,  23, 5,  17, 11};
  std::vector<const void *> inline_vectors(vector_count);
  std::vector<const void *> body_vectors(vector_count);

  for (size_t i = 0; i < vector_count; ++i) {
    int32_t sum_sq = 0;
    for (size_t d = 0; d < original_dimension; ++d) {
      const int raw_value = raw_byte_distribution(generator);
      records[i][d] = static_cast<int8_t>(raw_value - 128);
      sum_sq += raw_value * raw_value;
    }
    std::memcpy(records[i].data() + original_dimension, &sum_sq,
                sizeof(sum_sq));
    extra_values[extra_value_ids[i]] = sum_sq;
    extra_value_ptrs[i] = &extra_values[extra_value_ids[i]];
    inline_vectors[i] = records[i].data();
    body_vectors[i] = records[i].data();
  }

  std::vector<uint8_t> query(encoded_dimension, 0);
  int32_t query_sum_sq = 0;
  int32_t query_sum = 0;
  for (size_t d = 0; d < original_dimension; ++d) {
    const int raw = raw_byte_distribution(generator);
    query[d] = static_cast<uint8_t>(static_cast<int8_t>(raw - 128));
    query_sum_sq += raw * raw;
    query_sum += raw;
  }
  std::memcpy(query.data() + original_dimension, &query_sum_sq,
              sizeof(query_sum_sq));
  auto preprocess = metric->get_query_preprocess_func();
  ASSERT_TRUE(static_cast<bool>(preprocess));
  preprocess(query.data(), encoded_dimension);

  std::vector<float> inline_scores(vector_count, 0.0f);
  std::vector<float> extra_values_scores(vector_count, 0.0f);
  std::vector<float> expected_distances(vector_count, 0.0f);
  for (size_t i = 0; i < vector_count; ++i) {
    expected_distances[i] =
        ScalarRawQueryReference(records[i], query, original_dimension);
  }
  batch_distance(inline_vectors.data(), query.data(), vector_count,
                 encoded_dimension, inline_scores.data(), nullptr);
  batch_distance(body_vectors.data(), query.data(), vector_count,
                 encoded_dimension, extra_values_scores.data(),
                 extra_value_ptrs.data());

  EXPECT_EQ(expected_distances, inline_scores);
  EXPECT_EQ(inline_scores, extra_values_scores);
}

}  // namespace
}  // namespace zvec::core
