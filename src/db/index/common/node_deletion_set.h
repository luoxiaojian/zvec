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
#include <vector>
#include "db/index/common/delete_store.h"

namespace zvec {

// The collection's query guard keeps the sealed segment and its row storage
// alive throughout the synchronous search. The deletion bitmap stays live.
class NodeDeletionSet {
 public:
  NodeDeletionSet() = default;
  NodeDeletionSet(const std::vector<uint64_t> &doc_ids,
                  const DeleteStore &store)
      : NodeDeletionSet(doc_ids.data(), doc_ids.size(), store) {}

  explicit operator bool() const {
    return size_ != 0;
  }

  // Unknown nodes are kept, as in the segment filter. is_deleted() locks the
  // live bitmap for each lookup.
  bool contains(uint64_t node) const {
    return node < size_ && store_->is_deleted(doc_ids_[node]);
  }

  NodeDeletionSet subspan(uint64_t offset, uint64_t count) const {
    if (count == 0 || offset >= size_ || count > size_ - offset) return {};
    return NodeDeletionSet(doc_ids_ + offset, count, *store_);
  }

 private:
  NodeDeletionSet(const uint64_t *doc_ids, uint64_t size,
                  const DeleteStore &store)
      : doc_ids_(doc_ids), size_(size), store_(&store) {}

  const uint64_t *doc_ids_{nullptr};
  uint64_t size_{0};
  const DeleteStore *store_{nullptr};
};

}  // namespace zvec
