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
#include "utility/linear_pool.h"

using zvec::core::LinearPool;

TEST(LinearPool, InsertImmediatelyExposesAndRewindsNextCandidate) {
  LinearPool<float> pool;
  pool.reset(4, 0);

  EXPECT_TRUE(pool.insert(10, 10.0f));
  EXPECT_TRUE(pool.insert(20, 20.0f));
  ASSERT_TRUE(pool.has_next());

  uint32_t next = UINT32_MAX;
  EXPECT_EQ(10, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);

  bool rewound = false;
  EXPECT_TRUE(pool.insert_with_rewind(5, 5.0f, &rewound));
  EXPECT_TRUE(rewound);
  ASSERT_TRUE(pool.has_next());
  EXPECT_EQ(5, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);

  rewound = true;
  EXPECT_TRUE(pool.insert_with_rewind(30, 30.0f, &rewound));
  EXPECT_FALSE(rewound);
  EXPECT_EQ(20, pool.pop_with_next(&next));
  EXPECT_EQ(30u, next);
  EXPECT_EQ(30, pool.pop_with_next(&next));
  EXPECT_EQ(UINT32_MAX, next);
}

TEST(LinearPool, RejectedCandidateDoesNotRewind) {
  LinearPool<float> pool;
  pool.reset(2, 0);
  EXPECT_TRUE(pool.insert(1, 1.0f));
  EXPECT_TRUE(pool.insert(2, 2.0f));

  bool rewound = true;
  EXPECT_FALSE(pool.insert_with_rewind(3, 3.0f, &rewound));
  EXPECT_FALSE(rewound);
  EXPECT_EQ(2, pool.size());
  EXPECT_EQ(1, pool.pop());
}
