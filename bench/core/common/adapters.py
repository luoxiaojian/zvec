"""ann-benchmarks-style 3-stage query adapters for zvec bench."""

from __future__ import annotations

import numpy as np
from zvec import HnswQueryParam, VamanaQueryParam

from .schema import VECTOR_FIELD

SEARCH_PATHS = ("fast_query_doc_ids", "ann_bench_doc_ids")


def _param_str(param) -> str:
    if isinstance(param, HnswQueryParam):
        return f"ef={param.ef}"
    if isinstance(param, VamanaQueryParam):
        return f"ef_search={param.ef_search}"
    return str(param)


class ZvecFastQueryDocIdsANN:
    """``fast_query_doc_ids_only`` (or ``fast_query_doc_ids``) on the raw collection."""

    path = "fast_query_doc_ids"

    def __init__(
        self,
        raw,
        base_param,
        field: str = VECTOR_FIELD,
        *,
        ids_only: bool = True,
    ):
        self._raw = raw
        self._field = field
        self._param = base_param
        self._ids_only = ids_only
        method = "fast_query_doc_ids_only" if ids_only else "fast_query_doc_ids"
        self.name = f"zvec({method})"
        self._q = None
        self._n = 0
        self._res = None

    def set_query_arguments(self, param) -> None:
        self._param = param
        self.name = f"zvec(fast_query_doc_ids, {_param_str(param)})"

    def prepare_query(self, v: np.ndarray, n: int) -> None:
        self._q = np.ascontiguousarray(v, dtype=np.float32)
        self._n = n

    def run_prepared_query(self) -> None:
        if self._ids_only:
            self._res = self._raw.fast_query_doc_ids_only(
                self._field, self._q, self._n, self._param
            )
        else:
            ids, _scores = self._raw.fast_query_doc_ids(
                self._field, self._q, self._n, self._param
            )
            self._res = ids

    def get_prepared_query_results(self) -> np.ndarray:
        return self._res


class ZvecAnnBenchDocIdsANN:
    """``ann_bench_prepare`` + ``ann_bench_search_doc_ids_only`` (ann-benchmarks binding)."""

    path = "ann_bench_doc_ids"

    def __init__(self, raw, base_param, field: str = VECTOR_FIELD):
        self._raw = raw
        self._field = field
        self._param = base_param
        self.name = "zvec(ann_bench_doc_ids)"
        self._q = None
        self._n = 0
        self._res = None
        self._raw.ann_bench_prepare(field)
        self._raw.ann_bench_set_query_params(base_param)

    def set_query_arguments(self, param) -> None:
        self._param = param
        self._raw.ann_bench_set_query_params(param)
        self.name = f"zvec(ann_bench_doc_ids, {_param_str(param)})"

    def prepare_query(self, v: np.ndarray, n: int) -> None:
        self._q = np.ascontiguousarray(v, dtype=np.float32)
        self._n = n

    def run_prepared_query(self) -> None:
        self._res = self._raw.ann_bench_search_doc_ids_only(self._q, self._n)

    def get_prepared_query_results(self) -> np.ndarray:
        return self._res


def make_search_adapter(
    path: str,
    raw,
    param,
    *,
    field: str = VECTOR_FIELD,
    legacy_ids: bool = False,
):
    if path == "fast_query_doc_ids":
        return ZvecFastQueryDocIdsANN(
            raw, param, field=field, ids_only=not legacy_ids
        )
    if path == "ann_bench_doc_ids":
        return ZvecAnnBenchDocIdsANN(raw, param, field=field)
    raise ValueError(f"unknown search path {path!r}; choose from {SEARCH_PATHS}")
