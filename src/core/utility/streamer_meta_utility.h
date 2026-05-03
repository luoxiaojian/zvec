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

#include <zvec/core/framework/index_meta.h>

namespace zvec {
namespace core {

//! Merge metric / reformer info from a stored index meta into a streamer's
//! in-memory meta on open(). Common to every IndexStreamer implementation
//! that supports re-opening a persisted index (HNSW, Vamana, ...).
//!
//! - metric: stored metric_params take precedence; user-provided params from
//!   the in-memory meta are merged on top so caller-side overrides survive.
//! - reformer: copied verbatim when the stored meta carries one. Some
//!   quantizers (e.g. UniformInt8) compute their reformer params during
//!   train(); only the persisted meta has the real values, so the streamer
//!   must propagate them before downstream init.
inline void MergeStoredIndexMeta(const IndexMeta &stored, IndexMeta *target) {
  // Stored metric params win; in-memory params (e.g. MipsSquaredEuclidean
  // overrides) are merged in afterward.
  auto metric_params = stored.metric_params();
  metric_params.merge(target->metric_params());
  target->set_metric(stored.metric_name(), 0, metric_params);

  if (!stored.reformer_name().empty()) {
    target->set_reformer(stored.reformer_name(), 0, stored.reformer_params());
  }
}

}  // namespace core
}  // namespace zvec
