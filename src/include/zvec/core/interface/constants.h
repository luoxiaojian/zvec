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

#include <cstdint>
#include <limits>

namespace zvec::core_interface {

constexpr static uint32_t kDefaultHnswEfConstruction = 500;
constexpr static uint32_t kDefaultHnswNeighborCnt = 50;

constexpr static uint32_t kDefaultHnswEfSearch = 300;
constexpr static uint32_t kDefaultPrefetchOffset = 8;
constexpr static uint32_t kDefaultPrefetchLines = 0;

constexpr static uint32_t kDefaultVamanaMaxDegree = 64;
constexpr static uint32_t kDefaultVamanaSearchListSize = 100;
constexpr static float kDefaultVamanaAlpha = 1.2f;
constexpr static uint32_t kDefaultVamanaEfSearch = 200;
constexpr static uint32_t kDefaultVamanaMaxOcclusionSize = 750;
constexpr static bool kDefaultVamanaSaturateGraph = false;

// Vamana's beam-search traversal uses a bounded vector-prefetch budget.
// Values left at kVamanaQueryPrefetchAuto are resolved after loading the
// index, when the stored (possibly quantized) vector size and graph degree are
// known. The automatic policy seeds at most two cache lines per neighbor and
// limits one expansion's vector-prefetch footprint to roughly 6 KiB.
// Explicit query values, including zero, override the automatic value.
constexpr static uint32_t kVamanaQueryPrefetchCacheLineBytes = 64;
constexpr static uint32_t kVamanaQueryPrefetchBudgetBytes = 6 * 1024;
constexpr static uint32_t kVamanaQueryPrefetchTargetLines = 2;
constexpr static uint32_t kVamanaQueryPrefetchMaxValue = 256;
constexpr static uint32_t kVamanaQueryPrefetchAuto =
    std::numeric_limits<uint32_t>::max();
constexpr static uint32_t kDefaultVamanaPrefetchLines =
    kVamanaQueryPrefetchAuto;
constexpr static uint32_t kDefaultVamanaPrefetchOffset =
    kVamanaQueryPrefetchAuto;

constexpr const uint32_t kDefaultRabitqTotalBits = 7;
constexpr const uint32_t kDefaultRabitqNumClusters = 16;

constexpr const uint32_t kDefaultIvfRabitqNlist = 1024;
constexpr const uint32_t kDefaultIvfRabitqNprobe = 10;

constexpr const uint32_t kDefaultDiskAnnMaxDegree = 100;
constexpr const uint32_t kDefaultDiskAnnListSize = 200;
constexpr const uint32_t kDefaultDiskAnnPqChunkNum = 16;

}  // namespace zvec::core_interface
