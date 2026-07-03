#pragma once
#include <chrono>
#include <cstdint>

namespace zvec {

// 5 slots: 0=Python(unused in C++), 1=Binding, 2=Collection, 3=HNSW, 4=Vamana
struct AnnBenchTimer {
  static constexpr int kSlots = 5;
  static inline int64_t accum_ns[kSlots] = {};
  static inline int64_t count[kSlots] = {};

  static void reset() {
    for (int i = 0; i < kSlots; ++i) {
      accum_ns[i] = 0;
      count[i] = 0;
    }
  }

  static void add(int slot, int64_t ns) {
    accum_ns[slot] += ns;
    ++count[slot];
  }

  static int64_t get_ns(int slot) { return accum_ns[slot]; }
  static int64_t get_count(int slot) { return count[slot]; }
};

struct ScopedTimer {
  int slot_;
  std::chrono::steady_clock::time_point t0_;
  ScopedTimer(int slot) : slot_(slot), t0_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - t0_)
                  .count();
    AnnBenchTimer::add(slot_, ns);
  }
};

}  // namespace zvec
