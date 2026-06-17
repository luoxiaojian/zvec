# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for ``QuantizeType.UNIFORM_INT8`` on HNSW dense vector indexes.

Uniform int8 quantization uses a single global scale/bias trained from the
full dataset (``UniformInt8StreamingConverter`` in C++). These tests verify
that the Python binding exposes the enum and index param correctly, and that
a collection configured with ``quantize_type=QuantizeType.UNIFORM_INT8`` can
insert FP32 vectors, build a persisted index via ``optimize()``, and serve
search traffic.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pytest

import zvec
from zvec import (
    Collection,
    CollectionOption,
    CollectionSchema,
    Doc,
    FieldSchema,
    FlatIndexParam,
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    Query,
    VectorSchema,
)
from zvec.typing import DataType, IndexType, MetricType, QuantizeType

DIMENSION = 32
NUM_DOCS = 128
TOPK = 5


def _build_schema(
    name: str,
    *,
    index_param: HnswIndexParam | None = None,
) -> CollectionSchema:
    if index_param is None:
        index_param = HnswIndexParam(
            metric_type=MetricType.L2,
            m=16,
            ef_construction=100,
            quantize_type=QuantizeType.UNIFORM_INT8,
        )
    return CollectionSchema(
        name=name,
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
        ],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=DIMENSION,
                index_param=index_param,
            ),
        ],
    )


def _generate_docs(rng: np.random.Generator, num: int = NUM_DOCS) -> list[Doc]:
    docs: list[Doc] = []
    for i in range(num):
        vec = rng.standard_normal(DIMENSION).astype(np.float32)
        docs.append(
            Doc(
                id=str(i),
                fields={"id": i},
                vectors={"dense": vec.tolist()},
            )
        )
    return docs


def _query_topk(coll: Collection, query_vec: list[float], *, ef: int = 128) -> list[str]:
    hits = coll.query(
        Query(
            field_name="dense",
            vector=query_vec,
            param=HnswQueryParam(ef=ef),
        ),
        topk=TOPK,
    )
    assert hits is not None, "query returned None"
    assert len(hits) >= 1, f"expected at least one hit, got {hits!r}"
    return [doc.id for doc in hits]


def _build_flat_schema(name: str, *, metric_type: MetricType = MetricType.L2):
    return CollectionSchema(
        name=name,
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
        ],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=DIMENSION,
                index_param=FlatIndexParam(
                    metric_type=metric_type,
                    quantize_type=QuantizeType.UNIFORM_INT8,
                ),
            ),
        ],
    )


def _exact_topk_ids(docs: list[Doc], query_vec: list[float], k: int) -> list[str]:
    """Brute-force exact L2 nearest-neighbor ids, as a recall ground truth."""
    mat = np.asarray([d.vector("dense") for d in docs], dtype=np.float32)
    q = np.asarray(query_vec, dtype=np.float32)
    dists = np.sum((mat - q) ** 2, axis=1)
    order = np.argsort(dists, kind="stable")[:k]
    return [docs[i].id for i in order]


class TestUniformInt8QuantizeTypeSurface:
    """Verify ``QuantizeType.UNIFORM_INT8`` is exposed on the Python surface."""

    def test_enum_value(self):
        assert QuantizeType.UNIFORM_INT8.value == 5
        assert QuantizeType.UNIFORM_INT8.name == "UNIFORM_INT8"

    def test_top_level_namespace(self):
        assert zvec.QuantizeType.UNIFORM_INT8 is QuantizeType.UNIFORM_INT8
        assert "QuantizeType" in zvec.__all__


class TestHnswIndexParamUniformInt8Surface:
    """Verify ``HnswIndexParam`` round-trips ``UNIFORM_INT8`` correctly."""

    def test_custom_construction(self):
        param = HnswIndexParam(
            metric_type=MetricType.L2,
            m=48,
            ef_construction=200,
            quantize_type=QuantizeType.UNIFORM_INT8,
        )
        assert param.type == IndexType.HNSW
        assert param.metric_type == MetricType.L2
        assert param.m == 48
        assert param.ef_construction == 200
        assert param.quantize_type == QuantizeType.UNIFORM_INT8

    def test_to_dict(self):
        param = HnswIndexParam(
            metric_type=MetricType.L2,
            m=16,
            quantize_type=QuantizeType.UNIFORM_INT8,
        )
        data = param.to_dict()
        assert data["quantize_type"] == "UNIFORM_INT8"
        assert data["metric_type"] == "L2"
        assert data["m"] == 16

    def test_repr_contains_quantize_type(self):
        text = repr(
            HnswIndexParam(
                metric_type=MetricType.L2,
                quantize_type=QuantizeType.UNIFORM_INT8,
            )
        )
        assert "UNIFORM_INT8" in text or "quantize_type" in text

    def test_pickle_roundtrip(self):
        original = HnswIndexParam(
            metric_type=MetricType.L2,
            m=24,
            ef_construction=150,
            quantize_type=QuantizeType.UNIFORM_INT8,
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.quantize_type == QuantizeType.UNIFORM_INT8
        assert restored.to_dict() == original.to_dict()


class TestFlatIndexParamUniformInt8Surface:
    """``FlatIndexParam`` also accepts ``UNIFORM_INT8`` on FP32 columns."""

    def test_custom_construction(self):
        param = FlatIndexParam(
            metric_type=MetricType.L2,
            quantize_type=QuantizeType.UNIFORM_INT8,
        )
        assert param.type == IndexType.FLAT
        assert param.quantize_type == QuantizeType.UNIFORM_INT8
        assert param.to_dict()["quantize_type"] == "UNIFORM_INT8"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def collection_option() -> CollectionOption:
    return CollectionOption(read_only=False, enable_mmap=True)


class TestHnswUniformInt8EndToEnd:
    """Build and query a collection with uniform int8 HNSW quantization."""

    def test_schema_round_trip(self, tmp_path_factory, collection_option):
        schema = _build_schema("uniform_int8_schema")
        path = tmp_path_factory.mktemp("zvec") / "uniform_int8_schema"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            vec_schema = coll.schema.vectors[0]
            assert vec_schema.index_param.quantize_type == QuantizeType.UNIFORM_INT8
            assert vec_schema.index_param.metric_type == MetricType.L2
        finally:
            coll.destroy()

    def test_insert_query_before_optimize(
        self, tmp_path_factory, collection_option, rng
    ):
        """Writer segment should answer queries before ``optimize()``."""
        schema = _build_schema("uniform_int8_writer")
        path = tmp_path_factory.mktemp("zvec") / "uniform_int8_writer"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            docs = _generate_docs(rng)
            for r in coll.insert(docs=docs):
                assert r.ok(), f"insert failed: code={r.code()}"
            assert coll.stats.doc_count == NUM_DOCS

            query_vec = docs[0].vector("dense")
            ids = _query_topk(coll, query_vec)
            assert ids[0] == "0", f"expected self-recall on writer, got top-1 id={ids[0]}"
        finally:
            coll.destroy()

    def test_insert_optimize_query_and_reopen(
        self, tmp_path_factory, collection_option, rng
    ):
        """Persisted HNSW index with uniform int8 quantizer serves search."""
        schema = _build_schema("uniform_int8_persist")
        path = tmp_path_factory.mktemp("zvec") / "uniform_int8_persist"
        path_str = str(path)

        coll = zvec.create_and_open(
            path=path_str, schema=schema, option=collection_option
        )
        try:
            docs = _generate_docs(rng)
            for r in coll.insert(docs=docs):
                assert r.ok(), f"insert failed: code={r.code()}"

            coll.optimize()

            query_vec = docs[0].vector("dense")
            ids = _query_topk(coll, query_vec)
            assert ids[0] == "0", (
                f"expected self-recall after optimize, got top-1 id={ids[0]}"
            )
        finally:
            del coll

        reopened = zvec.open(path=path_str, option=collection_option)
        try:
            assert reopened.stats.doc_count == NUM_DOCS
            query_vec = docs[0].vector("dense")
            ids = _query_topk(reopened, query_vec)
            assert ids[0] == "0", (
                f"expected self-recall after reopen, got top-1 id={ids[0]}"
            )
        finally:
            reopened.destroy()

    def test_multi_segment_optimize_recall(
        self, tmp_path_factory, collection_option, rng
    ):
        """Optimize must train the global scale/bias across multiple input
        segments and keep recall high versus an exact FP32 ground truth."""
        schema = _build_schema("uniform_int8_multiseg")
        path = tmp_path_factory.mktemp("zvec") / "uniform_int8_multiseg"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            docs = _generate_docs(rng, num=256)
            # Insert in two batches with a flush in between so that optimize()
            # has to merge/train across more than one input segment.
            for batch in (docs[:128], docs[128:]):
                for r in coll.insert(docs=batch):
                    assert r.ok(), f"insert failed: code={r.code()}"
                coll.flush()
            assert coll.stats.doc_count == 256

            coll.optimize()

            # Measure recall@TOPK against exact L2 over a handful of queries.
            total = 0.0
            num_queries = 10
            for i in range(num_queries):
                query_vec = docs[i].vector("dense")
                got = set(_query_topk(coll, query_vec))
                want = set(_exact_topk_ids(docs, query_vec, TOPK))
                total += len(got & want) / float(TOPK)
            mean_recall = total / num_queries
            assert mean_recall >= 0.7, (
                f"uniform int8 recall too low after multi-segment optimize: "
                f"{mean_recall:.3f}"
            )
        finally:
            coll.destroy()


class TestFlatUniformInt8EndToEnd:
    """``FlatIndexParam`` + ``UNIFORM_INT8`` must build, persist and reopen.

    Regression guard: the deferred-training meta for the flat path must be
    persisted so that a reopened index can still create its reformer.
    """

    def test_insert_optimize_query_and_reopen(
        self, tmp_path_factory, collection_option, rng
    ):
        schema = _build_flat_schema("flat_uniform_int8_persist")
        path = tmp_path_factory.mktemp("zvec") / "flat_uniform_int8_persist"
        path_str = str(path)

        coll = zvec.create_and_open(
            path=path_str, schema=schema, option=collection_option
        )
        try:
            docs = _generate_docs(rng)
            for r in coll.insert(docs=docs):
                assert r.ok(), f"insert failed: code={r.code()}"

            coll.optimize()

            query_vec = docs[0].vector("dense")
            hits = coll.query(
                Query(field_name="dense", vector=query_vec), topk=TOPK
            )
            assert hits is not None and len(hits) >= 1
            assert hits[0].id == "0", (
                f"expected self-recall after optimize, got top-1 id={hits[0].id}"
            )
        finally:
            del coll

        reopened = zvec.open(path=path_str, option=collection_option)
        try:
            assert reopened.stats.doc_count == NUM_DOCS
            query_vec = docs[0].vector("dense")
            hits = reopened.query(
                Query(field_name="dense", vector=query_vec), topk=TOPK
            )
            assert hits is not None and len(hits) >= 1
            assert hits[0].id == "0", (
                f"expected self-recall after reopen, got top-1 id={hits[0].id}"
            )
        finally:
            reopened.destroy()


class TestUniformInt8MetricValidation:
    """UNIFORM_INT8 only supports the L2 metric; IP/COSINE must be rejected at
    schema-validation time (collection creation) rather than failing opaquely
    later during optimize()."""

    @pytest.mark.parametrize("metric", [MetricType.IP, MetricType.COSINE])
    def test_hnsw_non_l2_rejected(
        self, tmp_path_factory, collection_option, metric
    ):
        schema = _build_schema(
            "uniform_int8_bad_metric_hnsw",
            index_param=HnswIndexParam(
                metric_type=metric,
                m=16,
                ef_construction=100,
                quantize_type=QuantizeType.UNIFORM_INT8,
            ),
        )
        path = tmp_path_factory.mktemp("zvec") / "uniform_int8_bad_metric_hnsw"
        with pytest.raises(Exception, match="UNIFORM_INT8"):
            zvec.create_and_open(
                path=str(path), schema=schema, option=collection_option
            )

    @pytest.mark.parametrize("metric", [MetricType.IP, MetricType.COSINE])
    def test_flat_non_l2_rejected(
        self, tmp_path_factory, collection_option, metric
    ):
        schema = _build_flat_schema(
            "uniform_int8_bad_metric_flat", metric_type=metric
        )
        path = tmp_path_factory.mktemp("zvec") / "uniform_int8_bad_metric_flat"
        with pytest.raises(Exception, match="UNIFORM_INT8"):
            zvec.create_and_open(
                path=str(path), schema=schema, option=collection_option
            )
