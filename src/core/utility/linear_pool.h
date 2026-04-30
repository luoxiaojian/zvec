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

#include <sys/mman.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace zvec {
namespace core {

namespace linear_pool_impl {

constexpr size_t size_64B = 64;
constexpr size_t size_2M = 2 * 1024 * 1024;
constexpr size_t size_1G = 1 * 1024 * 1024 * 1024;

template <size_t alignment>
inline void *align_alloc_memory(size_t nbytes, bool set = true, uint8_t x = 0) {
  size_t len = (nbytes + alignment - 1) / alignment * alignment;
  if (alignment == size_1G) {
    printf("Allocating %.2fG memory for %.2fG data\n", double(len) / size_1G,
           double(nbytes) / size_1G);
  }
  auto p = std::aligned_alloc(alignment, len);
  if constexpr (alignment >= size_2M) {
    madvise(p, len, MADV_HUGEPAGE);
  }
  if (set) {
    std::memset(p, x, len);
  }
  return p;
}

inline void *align_alloc(size_t nbytes, bool set = true, uint8_t x = 0) {
  if (nbytes >= size_1G) {
    return align_alloc_memory<size_1G>(nbytes, set, x);
  } else if (nbytes >= size_2M) {
    return align_alloc_memory<size_2M>(nbytes, set, x);
  } else {
    return align_alloc_memory<size_64B>(nbytes, set, x);
  }
}

template <typename dist_t = float>
struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }

  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};

template <typename Block = uint64_t>
struct Bitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int32_t nb = 0;
  int nbytes = 0;
  Block *data = nullptr;

  Bitset() = default;

  explicit Bitset(int n)
      : nb(n),
        nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((Block *)align_alloc(nbytes)) {}

  friend void swap(Bitset &lhs, Bitset &rhs) {
    using std::swap;
    swap(lhs.nb, rhs.nb);
    swap(lhs.nbytes, rhs.nbytes);
    swap(lhs.data, rhs.data);
  }

  Bitset(const Bitset &) = delete;

  Bitset(Bitset &&rhs) {
    swap(*this, rhs);
  }

  Bitset &operator=(const Bitset &) = delete;

  Bitset &operator=(Bitset &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  ~Bitset() {
    if (data) {
      free(data);
    }
  }

  void reset(int32_t n) {
    if (n != nb) {
      // printf("Reset bitset size to %d\n", n);
      *this = Bitset(n);
    } else {
      memset(data, 0, nbytes);
    }
  }

  void set(int i) {
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }

  bool get(int i) const {
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }
};

}  // namespace linear_pool_impl

template <typename dist_t, typename BitsetType = linear_pool_impl::Bitset<>>
struct LinearPool {
  using dist_type = dist_t;

  LinearPool() = default;

  LinearPool(int n, int ef, int capacity)
      : nb(n), ef_(ef), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

  friend void swap(LinearPool &lhs, LinearPool &rhs) {
    using std::swap;
    swap(lhs.nb, rhs.nb);
    swap(lhs.size_, rhs.size_);
    swap(lhs.cur_, rhs.cur_);
    swap(lhs.ef_, rhs.ef_);
    swap(lhs.capacity_, rhs.capacity_);
    swap(lhs.data_, rhs.data_);
    swap(lhs.vis, rhs.vis);
  }

  LinearPool(const LinearPool &) = delete;

  LinearPool(LinearPool &&rhs) {
    swap(*this, rhs);
  }

  LinearPool &operator=(const LinearPool &) = delete;

  LinearPool &operator=(LinearPool &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  void reset(int32_t n, int ef, int32_t cap) {
    nb = n;
    size_ = cur_ = 0;
    ef_ = ef;
    capacity_ = cap;
    if (data_.size() < cap + 1) {
      data_.resize(cap + 1);
    }
    vis.reset(n);
  }

  __attribute__((always_inline)) int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
    // int len = size_;
    // int loc = 0;
    // while (len > 1) {
    //   int half = len / 2;
    //   loc += (dist > data_[loc + half - 1].distance) * half;
    //   len -= half;
    // }
    // return loc;
  }

  __attribute__((always_inline)) bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(linear_pool_impl::Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const {
    return cur_ < size_ && cur_ < ef_;
  }
  int id(int i) const {
    return get_id(data_[i].id);
  }
  dist_type dist(int i) const {
    return data_[i].distance;
  }
  int size() const {
    return size_;
  }
  int capacity() const {
    return capacity_;
  }

  void set_visited(int32_t u) {
    vis.set(u);
  }
  bool check_visited(int32_t u) const {
    return vis.get(u);
  }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const {
    return id & kMask;
  }
  void set_checked(int &id) {
    id |= 1 << 31;
  }
  bool is_checked(int id) const {
    return id >> 31 & 1;
  }

  void to_sorted(int32_t *ids, float *scores, int32_t length) const {
    for (int32_t i = 0; i < length; ++i) {
      ids[i] = id(i);
      if (scores) {
        scores[i] = dist(i);
      }
    }
  }

  int nb, size_ = 0, cur_ = 0, ef_, capacity_;
  std::vector<linear_pool_impl::Neighbor<dist_t>> data_;
  BitsetType vis;
};

}  // namespace core
}  // namespace zvec