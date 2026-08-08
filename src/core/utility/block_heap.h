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
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
#include <zvec/ailego/internal/platform.h>

#if (defined(__GNUC__) || defined(__clang__)) &&                         \
    (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define ZVEC_BLOCK_HEAP_AVX2_TARGET __attribute__((target("avx2")))
#define ZVEC_BLOCK_HEAP_AVX2_INTRINSICS 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define ZVEC_BLOCK_HEAP_AVX2_TARGET
#define ZVEC_BLOCK_HEAP_AVX2_INTRINSICS 1
#else
#define ZVEC_BLOCK_HEAP_AVX2_TARGET
#define ZVEC_BLOCK_HEAP_AVX2_INTRINSICS 0
#endif

namespace zvec {
namespace core {

// BlockHeap is a block-insert optimized alternative to LinearPool for graph
// search. It receives candidates in batches (push_block) and maintains a
// distance-sorted prefix of size ef with amortized O(k) bookkeeping per
// batch, replacing LinearPool's one-by-one sorted insert.
//
// Derived from pyglass' BlockHeap (https://github.com/zilliztech/pyglass,
// MIT License; see the NOTICE file and linear_pool.h for the full attribution).
// The graph prefetch is intentionally omitted: the call-site is expected to
// issue the neighbor-array prefetch itself (Vamana's greedy_search already
// does so).
//
// AVX2 requirement
// ----------------
// The implementation uses an AVX2 function target for the common full-pool
// filter. Keeping the small pool implementation header-local lets the query
// translation unit inline reset/pop and specialize push_block, while the
// rest of zvec stays at the portable x86-64 baseline. Callers MUST continue
// to runtime-gate BlockHeap on CpuFeatures::AVX2.
struct BlockHeap {
  BlockHeap() = default;
  ~BlockHeap() = default;

  BlockHeap(const BlockHeap &) = delete;
  BlockHeap &operator=(const BlockHeap &) = delete;

  BlockHeap(BlockHeap &&) = default;
  BlockHeap &operator=(BlockHeap &&) = default;

  // Reset the pool state for a new search round.  `capacity` is the retained
  // top-k size, `block_size` is an upper bound on the per-call push_block size
  // (used only for capacity hints).  Visited-node tracking is no longer owned
  // by the pool — the caller passes a VisitFilter reference instead.
  void reset(int32_t capacity, int32_t block_size) {
    ef_ = capacity;
    block_size_ = block_size;
    data_.clear();
    const size_t reserve_cnt =
        static_cast<size_t>(std::max(capacity, block_size)) +
        static_cast<size_t>(block_size);
    data_.reserve(reserve_cnt);
    tmp_.clear();
    tmp_.reserve(static_cast<size_t>(block_size));
    cur_ = 0;
  }

  // Insert a block of candidates. The distance array must have at least
  // `block_size` entries and the id array must have the same length.
  // `block_size` may differ from the value passed to reset(); reset()'s
  // block_size is only a capacity hint.
  ZVEC_BLOCK_HEAP_AVX2_TARGET void push_block(const float *distances,
                                               const uint32_t *nodes,
                                               int32_t block_size) {
    if (static_cast<int32_t>(data_.size()) == ef_) {
      const float max_dist = data_.back().second;
#if ZVEC_BLOCK_HEAP_AVX2_INTRINSICS
      const __m256 threshold_vec = _mm256_set1_ps(max_dist);
      int32_t i = 0;
      for (; i + 8 <= block_size; i += 8) {
        const __m256 d = _mm256_loadu_ps(distances + i);
        const __m256 mask =
            _mm256_cmp_ps(d, threshold_vec, _CMP_LT_OS);
        int bitmask = _mm256_movemask_ps(mask);
        while (bitmask != 0) {
          const int lane = ailego_ctz32(bitmask);
          tmp_.push_back({nodes[i + lane], distances[i + lane]});
          bitmask &= bitmask - 1;
        }
      }
      for (; i < block_size; ++i) {
        if (distances[i] < max_dist) {
          tmp_.push_back({nodes[i], distances[i]});
        }
      }
#else
      for (int32_t i = 0; i < block_size; ++i) {
        if (distances[i] < max_dist) {
          tmp_.push_back({nodes[i], distances[i]});
        }
      }
#endif
    } else {
      for (int32_t i = 0; i < block_size; ++i) {
        tmp_.push_back({nodes[i], distances[i]});
      }
    }
    if (tmp_.empty()) return;

    const auto less = [](const std::pair<uint32_t, float> &left,
                         const std::pair<uint32_t, float> &right) {
      return left.second < right.second;
    };
    if (static_cast<int32_t>(tmp_.size()) > ef_) {
      std::nth_element(tmp_.begin(), tmp_.begin() + ef_, tmp_.end(), less);
      tmp_.resize(static_cast<size_t>(ef_));
    }
    if (tmp_.size() <= 32) {
      for (size_t i = 1; i < tmp_.size(); ++i) {
        const auto value = tmp_[i];
        size_t position = i;
        while (position > 0 &&
               tmp_[position - 1].second > value.second) {
          tmp_[position] = tmp_[position - 1];
          --position;
        }
        tmp_[position] = value;
      }
    } else {
      std::sort(tmp_.begin(), tmp_.end(), less);
    }

    const int old_size = static_cast<int>(data_.size());
    const int temporary_size = static_cast<int>(tmp_.size());
    int old_index = old_size - 1;
    int temporary_index = temporary_size - 1;
    int output_index = old_size + temporary_size - 1;
    data_.resize(std::min(static_cast<size_t>(old_size + temporary_size),
                          static_cast<size_t>(ef_)));
    while (output_index >= ef_) {
      if (data_[old_index].second > tmp_[temporary_index].second) {
        --old_index;
      } else {
        --temporary_index;
      }
      --output_index;
    }
    while (old_index >= 0 && temporary_index >= 0) {
      if (data_[old_index].second > tmp_[temporary_index].second) {
        data_[output_index--] = data_[old_index--];
      } else {
        data_[output_index--] = tmp_[temporary_index--];
      }
    }
    if (temporary_index >= 0) {
      while (temporary_index >= 0) {
        data_[output_index--] = tmp_[temporary_index--];
      }
      cur_ = 0;
    } else if (static_cast<size_t>(output_index + 1) <= cur_) {
      cur_ = static_cast<size_t>(output_index + 1);
    }
    tmp_.clear();
  }

  // Is there an unpopped candidate?
  bool has_next() const {
    return cur_ < data_.size();
  }

  // Pop the closest unpopped candidate id (without the check bit).
  // Caller must ensure has_next() is true.
  uint32_t pop() {
    const size_t result = cur_;
    set_checked(data_[cur_].first);
    while (cur_ < data_.size() && is_checked(data_[cur_].first)) {
      ++cur_;
    }
    return get_id(data_[result].first);
  }

  // Retained candidate count.
  int32_t size() const {
    return static_cast<int32_t>(data_.size());
  }

  // Export sorted top-`length` ids (and optionally scores) — data_ is already
  // distance-sorted ascending.
  void to_sorted(uint32_t *ids, float *scores, int32_t length) const {
    const int32_t n = std::min(length, static_cast<int32_t>(data_.size()));
    for (int32_t i = 0; i < n; ++i) {
      ids[i] = get_id(data_[i].first);
      if (scores != nullptr) scores[i] = data_[i].second;
    }
  }

  // Direct sorted accessors (used by search result copy-out).
  uint32_t id(int32_t i) const {
    return get_id(data_[i].first);
  }
  float dist(int32_t i) const {
    return data_[i].second;
  }

  // Internal check-bit helpers (high bit marks a popped entry).
  static constexpr uint32_t kCheckedBit = 0x80000000u;
  static constexpr uint32_t kIdMask = 0x7FFFFFFFu;

  static void set_checked(uint32_t &id) {
    id |= kCheckedBit;
  }
  static bool is_checked(uint32_t id) {
    return (id & kCheckedBit) != 0u;
  }
  static uint32_t get_id(uint32_t id) {
    return id & kIdMask;
  }

 private:
  std::vector<std::pair<uint32_t, float>> data_;
  std::vector<std::pair<uint32_t, float>> tmp_;
  int32_t ef_{0};
  int32_t block_size_{0};
  size_t cur_{0};
};

}  // namespace core
}  // namespace zvec

#undef ZVEC_BLOCK_HEAP_AVX2_INTRINSICS
#undef ZVEC_BLOCK_HEAP_AVX2_TARGET
