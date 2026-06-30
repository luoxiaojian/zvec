"""Zvec bench client aligned with VectorDBBench's zvec adapter."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    HnswQueryParam,
    LogLevel,
    OptimizeOption,
    Query,
    VamanaQueryParam,
    create_and_open,
    open,
)

from .config import BuildConfig, SearchConfig
from .schema import VECTOR_FIELD, build_collection_schema, make_query_param

# Match VectorDBBench: init once at import time with low log noise.
zvec.init(log_level=LogLevel.WARN)


@dataclass
class ZvecBenchClient:
    """Thin wrapper around zvec Collection API for sift/gist core bench."""

    path: str
    query_param: HnswQueryParam | VamanaQueryParam | None = None
    schema: CollectionSchema | None = None
    read_only: bool = True
    enable_mmap: bool = True
    collection: zvec.Collection | None = None

    @property
    def option(self) -> CollectionOption:
        return CollectionOption(read_only=self.read_only, enable_mmap=self.enable_mmap)

    @classmethod
    def from_build_config(cls, cfg: BuildConfig) -> ZvecBenchClient:
        return cls(
            path=str(cfg.collection_path),
            schema=build_collection_schema(cfg),
            query_param=None,
            read_only=False,
        )

    @classmethod
    def from_search_config(cls, cfg: SearchConfig) -> ZvecBenchClient:
        # VectorDBBench caches query_param once at construction time.
        return cls(
            path=str(cfg.collection_path),
            query_param=make_query_param(cfg),
            read_only=True,
        )

    def drop_old(self) -> None:
        """Drop an existing collection before rebuild (VectorDBBench drop_old)."""
        try:
            collection = open(self.path, self.option)
            collection.destroy()
        except Exception:
            pass

    @contextmanager
    def init(self) -> Iterator[None]:
        """Open collection for the current process (VectorDBBench ``init()`` pattern)."""
        self.collection = open(self.path, self.option)
        try:
            yield
        finally:
            self.collection = None

    @contextmanager
    def build_session(self) -> Iterator[None]:
        """Create a new read-write collection and yield it (single writer)."""
        if self.schema is None:
            raise ValueError("schema is required to build a collection")
        self.collection = create_and_open(
            path=self.path,
            schema=self.schema,
            option=CollectionOption(read_only=False, enable_mmap=self.enable_mmap),
        )
        try:
            yield
        finally:
            self.collection = None

    def insert_batch(
        self,
        keys: np.ndarray,
        vectors: np.ndarray,
        *,
        field: str = VECTOR_FIELD,
    ) -> None:
        if self.collection is None:
            raise RuntimeError("collection not opened; use init() context")
        docs = [
            Doc(id=str(int(key)), vectors={field: embedding})
            for key, embedding in zip(keys, vectors.tolist(), strict=True)
        ]
        results = self.collection.insert(docs)
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if not result.ok():
                raise RuntimeError(f"insert failed: {result.code()}")

    def optimize(self, *, concurrency: int | None = None) -> None:
        if self.collection is None:
            raise RuntimeError("collection not opened; use init() context")
        option = OptimizeOption()
        if concurrency is not None and concurrency > 0:
            option = OptimizeOption(concurrency=concurrency)
        self.collection.optimize(option=option)

    def search_embedding(
        self,
        query: list[float],
        k: int,
    ) -> list[int]:
        """Vector search via collection.query, returning neighbor ids."""
        if self.collection is None:
            raise RuntimeError("collection not opened; use init() context")
        if self.query_param is None:
            raise ValueError("query_param is required for search")

        results = self.collection.query(
            queries=Query(
                field_name=VECTOR_FIELD,
                vector=query,
                param=self.query_param,
            ),
            topk=k,
            output_fields=[],
        )
        if results is None:
            return []
        return [int(doc.id) for doc in results]

    def search_embedding_fast(
        self,
        query: list[float],
        k: int,
    ) -> list[int]:
        """Low-overhead bypass: collection.fast_query -> neighbor ids.

        Single C++ call straight to the segment vector indexer (no SQL planner,
        Arrow pipeline, Doc materialization).
        """
        if self.collection is None:
            raise RuntimeError("collection not opened; use init() context")
        if self.query_param is None:
            raise ValueError("query_param is required for search")

        ids_str, _scores = self.collection.fast_query(
            VECTOR_FIELD, query, topk=k, param=self.query_param
        )
        return [int(x) for x in ids_str]

    def search_embedding_fast_ids(
        self,
        query: list[float],
        k: int,
    ) -> np.ndarray:
        """Cheapest bypass: collection.fast_query_doc_ids_only -> int64 ids.

        Returns stable internal global doc ids directly as a numpy array (no
        USER_ID string fetch, no Python str->int). The ids equal the dataset
        row index when vectors were inserted in row order.
        """
        if self.collection is None:
            raise RuntimeError("collection not opened; use init() context")
        if self.query_param is None:
            raise ValueError("query_param is required for search")

        return self.collection.fast_query_doc_ids_only(
            VECTOR_FIELD, query, topk=k, param=self.query_param
        )

    def search_with_scores(
        self,
        query: list[float],
        k: int,
    ) -> tuple[list[str], list[float]]:
        if self.collection is None:
            raise RuntimeError("collection not opened; use init() context")
        if self.query_param is None:
            raise ValueError("query_param is required for search")

        results = self.collection.query(
            queries=Query(
                field_name=VECTOR_FIELD,
                vector=query,
                param=self.query_param,
            ),
            topk=k,
            output_fields=[],
        )
        if results is None:
            return [], []
        return [doc.id for doc in results], [doc.score for doc in results]
