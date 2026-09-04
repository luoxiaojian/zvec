// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <gtest/gtest.h>
#include "utility/block_heap.h"

using zvec::core::BlockHeap;

TEST(BlockHeap, PopExposesNextUnexpandedCandidate) {
  BlockHeap pool;
  pool.reset(4, 4);

  const float distances[] = {30.0f, 10.0f, 40.0f, 20.0f};
  const uint32_t ids[] = {30, 10, 40, 20};
  pool.push_block(distances, ids, 4);

  uint32_t next = UINT32_MAX;
  EXPECT_EQ(10u, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);
  EXPECT_EQ(20u, pool.pop_with_next(&next));
  EXPECT_EQ(30u, next);
  EXPECT_EQ(30u, pool.pop_with_next(&next));
  EXPECT_EQ(40u, next);
  EXPECT_EQ(40u, pool.pop_with_next(&next));
  EXPECT_EQ(UINT32_MAX, next);
  EXPECT_FALSE(pool.has_next());
}

TEST(BlockHeap, RewindIsVisibleThroughNextCandidate) {
  BlockHeap pool;
  pool.reset(4, 2);

  const float initial_distances[] = {10.0f, 20.0f};
  const uint32_t initial_ids[] = {10, 20};
  pool.push_block(initial_distances, initial_ids, 2);

  uint32_t next = UINT32_MAX;
  EXPECT_EQ(10u, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);

  const float new_distances[] = {5.0f, 30.0f};
  const uint32_t new_ids[] = {5, 30};
  pool.push_block(new_distances, new_ids, 2);

  EXPECT_EQ(5u, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);
}

TEST(BlockHeap, ExistingPopInterfaceIsPreserved) {
  BlockHeap pool;
  pool.reset(2, 2);

  const float distances[] = {2.0f, 1.0f};
  const uint32_t ids[] = {2, 1};
  pool.push_block(distances, ids, 2);

  EXPECT_EQ(1u, pool.pop());
  EXPECT_EQ(2u, pool.pop());
  EXPECT_FALSE(pool.has_next());
}
