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

#include "python_collection.h"

#include "zvec/db/ann_bench_timer.h"
#include <cstring>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <zvec/db/collection.h>

namespace zvec {

inline void throw_if_error(const Status &status) {
  switch (status.code()) {
    case StatusCode::OK:
      return;
    case StatusCode::NOT_FOUND:
      throw py::key_error(status.message());
    case StatusCode::INVALID_ARGUMENT:
      throw py::value_error(status.message());
    case StatusCode::INTERNAL_ERROR:
    case StatusCode::ALREADY_EXISTS:
    case StatusCode::NOT_SUPPORTED:
    case StatusCode::PERMISSION_DENIED:
    case StatusCode::FAILED_PRECONDITION:
    case StatusCode::UNKNOWN:
    default:
      throw std::runtime_error(status.message());
  }
}


template <typename T>
T unwrap_expected(const tl::expected<T, Status> &exp) {
  if (exp.has_value()) {
    return exp.value();
  }
  throw_if_error(exp.error());
  return T{};
}

void ZVecPyCollection::Initialize(pybind11::module_ &m) {
  py::class_<Collection, Collection::Ptr> collection(m, "_Collection");
  bind_db_methods(collection);
  bind_ddl_methods(collection);
  bind_dml_methods(collection);
  bind_dql_methods(collection);
  collection.def(py::pickle(
      [](const Collection &c) {
        return py::make_tuple(c.Path(), c.Schema(), c.Options());
      },
      [](py::tuple t) {
        if (t.size() != 3) {
          throw std::runtime_error("Invalid tuple size for Collection pickle");
        }
        std::string path = t[0].cast<std::string>();
        auto schema = t[1].cast<CollectionSchema>();
        CollectionOptions options = t[2].cast<CollectionOptions>();
        auto result = Collection::Open(path, options);
        // auto result = Collection::CreateAndOpen(path, schema, options);
        return unwrap_expected(result);
      }));
}

void ZVecPyCollection::bind_db_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  col.def_static("CreateAndOpen",
                 [](const std::string &path, const CollectionSchema &schema,
                    const CollectionOptions &options) {
                   Result<Collection::Ptr> result;
                   {
                     py::gil_scoped_release release;
                     result = Collection::CreateAndOpen(path, schema, options);
                   }
                   return unwrap_expected(result);
                 })
      .def_static("Open", [](const std::string &path,
                             const CollectionOptions &options) {
        Result<Collection::Ptr> result;
        {
          py::gil_scoped_release release;
          result = Collection::Open(path, options);
        }
        return unwrap_expected(result);
      });
}


void ZVecPyCollection::bind_ddl_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // bind collection properties
  col.def("Path",
          [](const Collection &self) {
            auto ret = self.Path();
            return unwrap_expected(ret);
          })
      .def("Options",
           [](const Collection &self) {
             auto ret = self.Options();
             return unwrap_expected(ret);
           })
      .def("Schema",
           [](const Collection &self) {
             auto ret = self.Schema();
             return unwrap_expected(ret);
           })
      .def("Stats", [](const Collection &self) {
        auto ret = self.Stats();
        return unwrap_expected(ret);
      });

  // bind collection ddl methods
  col.def("Destroy",
          [](Collection &self) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.Destroy();
            }
            throw_if_error(status);
          })
      .def("Flush", [](Collection &self) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.Flush();
        }
        throw_if_error(status);
      });

  // binding index ddl methods
  col.def("CreateIndex",
          [](Collection &self, const std::string &column_name,
             const IndexParams::Ptr &index_options,
             const CreateIndexOptions &options) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.CreateIndex(column_name, index_options, options);
            }
            throw_if_error(status);
          })
      .def("DropIndex",
           [](Collection &self, const std::string &column_name) {
             Status status;
             {
               py::gil_scoped_release release;
               status = self.DropIndex(column_name);
             }
             throw_if_error(status);
           })
      .def("Optimize", [](Collection &self, const OptimizeOptions &options) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.Optimize(options);
        }
        throw_if_error(status);
      });

  // binding column ddl methods
  col.def("AddColumn",
          [](Collection &self, const FieldSchema::Ptr &column_schema,
             const std::string &expression, const AddColumnOptions &options) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.AddColumn(column_schema, expression, options);
            }
            throw_if_error(status);
          })
      .def("DropColumn",
           [](Collection &self, std::string &column_name) {
             Status status;
             {
               py::gil_scoped_release release;
               status = self.DropColumn(column_name);
             }
             throw_if_error(status);
           })
      .def("AlterColumn", [](Collection &self, std::string &column_name,
                             const std::string &rename,
                             const FieldSchema::Ptr &new_column_schema,
                             const AlterColumnOptions &options) {
        Status status;
        {
          py::gil_scoped_release release;
          status =
              self.AlterColumn(column_name, rename, new_column_schema, options);
        }
        throw_if_error(status);
      });
}

void ZVecPyCollection::bind_dml_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  // bind collection upsert/insert/update/delete methods
  col.def("Insert",
          [](Collection &self, std::vector<Doc> &docs) {
            Result<WriteResults> result;
            {
              py::gil_scoped_release release;
              result = self.Insert(docs);
            }
            return unwrap_expected(result);
          })
      .def("Update",
           [](Collection &self, std::vector<Doc> &docs) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.Update(docs);
             }
             return unwrap_expected(result);
           })
      .def("Upsert",
           [](Collection &self, std::vector<Doc> &docs) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.Upsert(docs);
             }
             return unwrap_expected(result);
           })
      .def("Delete",
           [](Collection &self, const std::vector<std::string> &pks) {
             Result<WriteResults> result;
             {
               py::gil_scoped_release release;
               result = self.Delete(pks);
             }
             return unwrap_expected(result);
           })
      .def("DeleteByFilter", [](Collection &self, const std::string &filter) {
        Status status;
        {
          py::gil_scoped_release release;
          status = self.DeleteByFilter(filter);
        }
        throw_if_error(status);
      });
}

void ZVecPyCollection::bind_dql_methods(
    py::class_<Collection, Collection::Ptr> &col) {
  col.def("Query",
          [](const Collection &self, const SearchQuery &query) {
            Result<DocPtrList> result;
            {
              py::gil_scoped_release release;
              result = self.Query(query);
            }
            // return DocPtrList
            return unwrap_expected(result);
          })
      // Lowest-overhead dense KNN bypass: a single call takes the raw query
      // vector buffer (numpy) + params, goes straight to the segment indexer
      // (no SQL planner / Arrow pipeline / Doc materialization), and returns
      // (ids, scores). Maximizes recovery of native C++ search throughput.
      .def(
          "fast_query",
          [](const Collection &self, const std::string &field_name,
             const py::array &vector, int topk,
             const QueryParams::Ptr &query_params) {
            // request() needs the GIL; do it before releasing below.
            py::buffer_info info = vector.request();
            const void *ptr = info.ptr;
            Result<RawSearchResult> result;
            {
              py::gil_scoped_release release;
              result = self.FastQuery(field_name, ptr, topk, query_params);
            }
            RawSearchResult raw = unwrap_expected(result);
            return std::pair<std::vector<std::string>, std::vector<float>>(
                std::move(raw.pks), std::move(raw.scores));
          },
          py::arg("field_name"), py::arg("vector"), py::arg("topk"),
          py::arg("query_params") = QueryParams::Ptr{})
      // Like fast_query, but returns stable internal global doc ids (insertion
      // order) as an int64 numpy array instead of string primary keys. Skips
      // the Arrow/USER_ID string materialization, so it is the cheapest
      // id-returning bypass. Runs in parallel to fast_query.
      .def(
          "fast_query_doc_ids",
          [](const Collection &self, const std::string &field_name,
             const py::array &vector, int topk,
             const QueryParams::Ptr &query_params) {
            py::buffer_info info = vector.request();
            const void *ptr = info.ptr;
            Result<RawSearchResultDocIds> result;
            {
              py::gil_scoped_release release;
              result =
                  self.FastQueryDocIds(field_name, ptr, topk, query_params);
            }
            RawSearchResultDocIds raw = unwrap_expected(result);
            const auto n = static_cast<py::ssize_t>(raw.ids.size());
            py::array_t<int64_t> ids(n);
            py::array_t<float> scores(n);
            if (n > 0) {
              std::memcpy(ids.mutable_data(), raw.ids.data(),
                          raw.ids.size() * sizeof(int64_t));
              std::memcpy(scores.mutable_data(), raw.scores.data(),
                          raw.scores.size() * sizeof(float));
            }
            return std::pair<py::array_t<int64_t>, py::array_t<float>>(
                std::move(ids), std::move(scores));
          },
          py::arg("field_name"), py::arg("vector"), py::arg("topk"),
          py::arg("query_params") = QueryParams::Ptr{})
      // Like fast_query_doc_ids, but returns ONLY the int64 id array (no
      // scores), handing the result buffer to numpy via a capsule (zero-copy,
      // no extra numpy allocation or memcpy). Matches the lowest-overhead
      // bindings used by the top ann-benchmarks entries (glass/scann/n2). Use
      // when scores are not needed (e.g. recall benchmarking).
      .def(
          "fast_query_doc_ids_only",
          [](const Collection &self, const std::string &field_name,
             const py::array &vector, int topk,
             const QueryParams::Ptr &query_params) {
            py::buffer_info info = vector.request();
            const void *ptr = info.ptr;
            Result<RawSearchResultDocIds> result;
            {
              py::gil_scoped_release release;
              result =
                  self.FastQueryDocIds(field_name, ptr, topk, query_params);
            }
            RawSearchResultDocIds raw = unwrap_expected(result);
            // Move the id vector to the heap and let a capsule own it; the
            // numpy array aliases its buffer directly (no copy).
            auto *ids_vec = new std::vector<int64_t>(std::move(raw.ids));
            const auto n = static_cast<py::ssize_t>(ids_vec->size());
            py::capsule free_when_done(ids_vec, [](void *p) {
              delete reinterpret_cast<std::vector<int64_t> *>(p);
            });
            return py::array_t<int64_t>(
                {n}, {static_cast<py::ssize_t>(sizeof(int64_t))},
                ids_vec->data(), free_when_done);
          },
          py::arg("field_name"), py::arg("vector"), py::arg("topk"),
          py::arg("query_params") = QueryParams::Ptr{})
      // Ann-benchmarks bypass: cache indexers at prepare time; set search
      // params once; per-query call passes only (vector, topk). Parallel to
      // fast_query_doc_ids_only — does not modify that path.
      .def(
          "ann_bench_prepare",
          [](Collection &self, const std::string &field_name) {
            Status status;
            {
              py::gil_scoped_release release;
              status = self.AnnBenchPrepare(field_name);
            }
            throw_if_error(status);
          },
          py::arg("field_name"))
      .def(
          "ann_bench_set_query_params",
          [](Collection &self, const QueryParams::Ptr &query_params) {
            py::gil_scoped_release release;
            self.AnnBenchSetQueryParams(query_params);
          },
          py::arg("query_params") = QueryParams::Ptr{})
      .def(
          "ann_bench_search_doc_ids_only",
          [](const Collection &self, const py::array &vector, int topk) {
            py::buffer_info info = vector.request();
            const void *ptr = info.ptr;
            Result<RawSearchResultDocIds> result;
            {
              py::gil_scoped_release release;
              result = self.AnnBenchSearchDocIds(ptr, topk);
            }
            RawSearchResultDocIds raw = unwrap_expected(result);
            auto *ids_vec = new std::vector<int64_t>(std::move(raw.ids));
            const auto n = static_cast<py::ssize_t>(ids_vec->size());
            py::capsule free_when_done(ids_vec, [](void *p) {
              delete reinterpret_cast<std::vector<int64_t> *>(p);
            });
            return py::array_t<int64_t>(
                {n}, {static_cast<py::ssize_t>(sizeof(int64_t))},
                ids_vec->data(), free_when_done);
          },
          py::arg("vector"), py::arg("topk"))
      .def(
          "ann_bench_search_fast",
          [](const Collection &self,
             const py::array_t<float, py::array::c_style> &vector,
             py::array_t<int64_t> &output) {
            const float *ptr = vector.data();
            int topk = static_cast<int>(output.size());
            int64_t *out_ptr = output.mutable_data();
            {
              py::gil_scoped_release release;
              zvec::ScopedTimer _t(1);
              self.AnnBenchSearchFast(ptr, topk, out_ptr);
            }
          },
          py::arg("vector"), py::arg("output"))
      .def(
          "ann_bench_timer_reset",
          [](const Collection &) { zvec::AnnBenchTimer::reset(); })
      .def(
          "ann_bench_timer_get_ns",
          [](const Collection &, int slot) {
            return zvec::AnnBenchTimer::get_ns(slot);
          },
          py::arg("slot"))
      .def(
          "ann_bench_timer_get_count",
          [](const Collection &, int slot) {
            return zvec::AnnBenchTimer::get_count(slot);
          },
          py::arg("slot"))
      // Batch version of fast_query_doc_ids_only: process all queries in a
      // single C++ loop with the GIL released, eliminating per-query Python
      // dispatch overhead. Input: (nq, dim) array. Output: (nq, topk) int64.
      .def(
          "batch_fast_query_doc_ids_only",
          [](const Collection &self, const std::string &field_name,
             const py::array &queries, int topk,
             const QueryParams::Ptr &query_params) {
            py::buffer_info info = queries.request();
            if (info.ndim != 2)
              throw std::runtime_error("queries must be 2D (nq x dim)");
            const auto nq = static_cast<py::ssize_t>(info.shape[0]);
            const auto dim = static_cast<py::ssize_t>(info.shape[1]);
            const auto item_size = static_cast<py::ssize_t>(info.itemsize);
            const char *base = static_cast<const char *>(info.ptr);

            // Pre-allocate output: (nq, topk) int64
            py::array_t<int64_t> out({nq, static_cast<py::ssize_t>(topk)});
            int64_t *out_ptr = out.mutable_data();

            {
              py::gil_scoped_release release;
              for (py::ssize_t i = 0; i < nq; ++i) {
                const void *qptr = base + i * dim * item_size;
                auto result =
                    self.FastQueryDocIds(field_name, qptr, topk, query_params);
                auto raw = unwrap_expected(result);
                const auto n =
                    std::min(static_cast<py::ssize_t>(raw.ids.size()),
                             static_cast<py::ssize_t>(topk));
                std::memcpy(out_ptr + i * topk, raw.ids.data(),
                            n * sizeof(int64_t));
                // Pad with -1 if fewer than topk results
                for (py::ssize_t j = n; j < topk; ++j)
                  out_ptr[i * topk + j] = -1;
              }
            }
            return out;
          },
          py::arg("field_name"), py::arg("queries"), py::arg("topk"),
          py::arg("query_params") = QueryParams::Ptr{})
      // Batch version of fast_query_doc_ids: returns both ids and scores.
      // Input: (nq, dim) array. Output: ((nq, topk) int64, (nq, topk) float32).
      .def(
          "batch_fast_query_doc_ids",
          [](const Collection &self, const std::string &field_name,
             const py::array &queries, int topk,
             const QueryParams::Ptr &query_params) {
            py::buffer_info info = queries.request();
            if (info.ndim != 2)
              throw std::runtime_error("queries must be 2D (nq x dim)");
            const auto nq = static_cast<py::ssize_t>(info.shape[0]);
            const auto dim = static_cast<py::ssize_t>(info.shape[1]);
            const auto item_size = static_cast<py::ssize_t>(info.itemsize);
            const char *base = static_cast<const char *>(info.ptr);

            py::array_t<int64_t> ids_out({nq, static_cast<py::ssize_t>(topk)});
            py::array_t<float> scores_out({nq, static_cast<py::ssize_t>(topk)});
            int64_t *ids_ptr = ids_out.mutable_data();
            float *scores_ptr = scores_out.mutable_data();

            {
              py::gil_scoped_release release;
              for (py::ssize_t i = 0; i < nq; ++i) {
                const void *qptr = base + i * dim * item_size;
                auto result =
                    self.FastQueryDocIds(field_name, qptr, topk, query_params);
                auto raw = unwrap_expected(result);
                const auto n =
                    std::min(static_cast<py::ssize_t>(raw.ids.size()),
                             static_cast<py::ssize_t>(topk));
                std::memcpy(ids_ptr + i * topk, raw.ids.data(),
                            n * sizeof(int64_t));
                std::memcpy(scores_ptr + i * topk, raw.scores.data(),
                            n * sizeof(float));
                for (py::ssize_t j = n; j < topk; ++j) {
                  ids_ptr[i * topk + j] = -1;
                  scores_ptr[i * topk + j] = std::numeric_limits<float>::max();
                }
              }
            }
            return std::pair<py::array_t<int64_t>, py::array_t<float>>(
                std::move(ids_out), std::move(scores_out));
          },
          py::arg("field_name"), py::arg("queries"), py::arg("topk"),
          py::arg("query_params") = QueryParams::Ptr{})
      // MultiQuery: multi query with reranker
      .def(
          "Query",
          [](const Collection &self, const MultiQuery &query) {
            Result<DocPtrList> result;
            {
              py::gil_scoped_release release;
              result = self.Query(query);
            }
            // return DocPtrList
            return unwrap_expected(result);
          },
          py::arg("query"), "Execute a multi query with re-ranking.")
      .def("GroupByQuery",
           [](const Collection &self, const GroupByVectorQuery &query) {
             Result<GroupResults> result;
             {
               py::gil_scoped_release release;
               result = self.GroupByQuery(query);
             }
             // return GroupResults
             return unwrap_expected(result);
           })
      .def(
          "Fetch",
          [](const Collection &self, const std::vector<std::string> &pks,
             const std::optional<std::vector<std::string>> &output_fields,
             bool include_vector) {
            Result<DocPtrMap> result;
            {
              py::gil_scoped_release release;
              result = self.Fetch(pks, output_fields, include_vector);
            }
            // return DocPtrMap
            return unwrap_expected(result);
          },
          py::arg("pks"), py::arg("output_fields") = py::none(),
          py::arg("include_vector") = true)
      .def(
          "_debug_hnsw_storage_mode",
          [](const Collection &self, const std::string &column_name) {
            const auto result = self.DebugGetHnswStorageMode(column_name);
            return unwrap_expected(result);
          },
          py::arg("column_name"),
          "Debug-only: returns the storage mode of the HNSW entity on the "
          "given vector column. One of 'mmap', 'buffer_pool', 'contiguous'. "
          "Raises KeyError if no HNSW index exists on the column, or "
          "ValueError if the column's index is not an HNSW index. Intended "
          "for introspection and testing only; not part of the stable API.");
}

}  // namespace zvec