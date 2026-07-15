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

#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_helper.h>
#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_runner.h>
#include <zvec/core/framework/index_stats.h>

namespace zvec {
namespace core {

/*! Index Streamer
 */
class IndexStreamer : public IndexRunner {
 public:
  //! Index Streamer Pointer
  typedef std::shared_ptr<IndexStreamer> Pointer;

  //! Destructor
  ~IndexStreamer(void) override = default;

  //! Initialize the builder
  virtual int init(const IndexMeta & /*meta*/,
                   const ailego::Params & /*params*/) {
    return IndexError_NotImplemented;
  }

  //! Open a index from storage
  virtual int open(IndexStorage::Pointer stg) = 0;

  //! Flush index
  virtual int flush(uint64_t check_point) = 0;

  //! Finalize a completed build before persistence. Streaming indexes that
  //! need a whole-graph post-build phase override this hook. The default is a
  //! no-op so existing streamers keep their current merge/flush behavior.
  virtual int finalize_build(void) {
    return 0;
  }

  //! Close index
  virtual int close(void) = 0;

  //! Retrieve meta of index
  virtual const IndexMeta &meta(void) const = 0;

  //! Merge converter/reformer params produced after deferred training (e.g.
  //! UniformInt8) into the streamer meta so they are persisted on flush.
  //! Default is a no-op for streamers that persist their meta through other
  //! paths (e.g. FlatStreamer serializes meta at dump time); streamers that
  //! rely on flush to persist trained meta override this.
  virtual void merge_trained_meta(const IndexMeta & /*trained_meta*/) {}
};

}  // namespace core
}  // namespace zvec
