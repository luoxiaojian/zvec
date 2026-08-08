// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstring>

#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_holder.h>

using namespace zvec::core;

TEST(RawUint8Reformer, DirectPhysicalConversion) {
  constexpr size_t kDimension = 128;
  IndexMeta input_meta(IndexMeta::DataType::DT_FP32, kDimension);

  auto converter = IndexFactory::CreateConverter("RawUint8Converter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(input_meta, zvec::ailego::Params()));
  EXPECT_EQ(converter->meta().data_type(), IndexMeta::DataType::DT_UINT8);
  EXPECT_EQ(converter->meta().element_size(), kDimension);

  auto source =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          kDimension);
  zvec::ailego::NumericalVector<float> vector(kDimension);
  for (size_t i = 0; i < kDimension; ++i) {
    vector[i] = static_cast<float>((i * 37) & 0xff);
  }
  source->emplace(7, vector);
  ASSERT_EQ(0, IndexConverter::TrainAndTransform(converter, source));

  auto encoded = converter->result();
  ASSERT_TRUE(encoded);
  EXPECT_EQ(encoded->element_size(), kDimension);
  auto iterator = encoded->create_iterator();
  ASSERT_TRUE(iterator->is_valid());
  const auto *bytes = static_cast<const uint8_t *>(iterator->data());
  for (size_t i = 0; i < kDimension; ++i) {
    EXPECT_EQ(bytes[i], static_cast<uint8_t>(vector[i]));
  }

  auto reformer = IndexFactory::CreateReformer("RawUint8Reformer");
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(zvec::ailego::Params()));
  std::string query;
  IndexQueryMeta query_meta;
  ASSERT_EQ(0, reformer->transform(
                   vector.data(),
                   IndexQueryMeta(IndexMeta::DataType::DT_FP32, kDimension),
                   &query, &query_meta));
  EXPECT_EQ(query.size(), kDimension);
  EXPECT_EQ(query_meta.data_type(), IndexMeta::DataType::DT_UINT8);
  EXPECT_EQ(0, std::memcmp(query.data(), bytes, kDimension));

  std::string reverted;
  ASSERT_EQ(0, reformer->revert(query.data(), query_meta, &reverted));
  ASSERT_EQ(reverted.size(), kDimension * sizeof(float));
  const auto *round_trip = reinterpret_cast<const float *>(reverted.data());
  for (size_t i = 0; i < kDimension; ++i) {
    EXPECT_FLOAT_EQ(round_trip[i], vector[i]);
  }
}
