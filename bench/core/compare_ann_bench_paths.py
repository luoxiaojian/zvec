#!/usr/bin/env python3
"""Compare zvec search paths: fast_query vs ann_bench bypass vs batch.

Correctness: all single-query paths must return identical doc ids.
Performance: per-query timing (ann-benchmarks style) and whole-loop timing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import zvec
from zvec import CollectionOption, HnswQueryParam, LogLevel, VamanaQueryParam

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from common.io import load_ground_truth_ids, load_query_vectors  # noqa: E402

try:
    zvec.init(log_level=LogLevel.WARN)
except RuntimeError:
    pass

VECTOR_FIELD = "vector"


def recall_at_k(results, gt, k: int) -> float:
    hit = tot = 0
    for res, g in zip(results, gt):
        truth = set(int(x) for x in g[:k])
        for x in res[:k]:
            if int(x) in truth:
                hit += 1
            tot += 1
    return hit / tot if tot else 0.0


def bench_per_query(fn, X, runs: int) -> float:
    best = float("inf")
    for _ in range(runs):
        tsum = 0.0
        for v in X:
            t0 = time.perf_counter()
            fn(v)
            tsum += time.perf_counter() - t0
        best = min(best, tsum / len(X))
    return best


def bench_loop(fn_once, n: int, runs: int) -> float:
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(n):
            fn_once()
        best = min(best, (time.perf_counter() - t0) / n)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--index", choices=["hnsw", "vamana"], default="hnsw")
    ap.add_argument("--ef", type=int, default=24)
    ap.add_argument("--prefetch-offset", type=int, default=64)
    ap.add_argument("--prefetch-lines", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-query", type=int, default=0, help="0 = all queries")
    ap.add_argument("--query", default="/root/zvec_workspace/output/sift_test.txt")
    ap.add_argument("--gt", default="/root/zvec_workspace/output/sift_neighbors.txt")
    args = ap.parse_args()

    queries = load_query_vectors(Path(args.query), first_sep=";", second_sep=" ")
    if args.max_query > 0:
        queries = queries[: args.max_query]
    X = np.ascontiguousarray(np.stack(queries), dtype=np.float32)
    gt = load_ground_truth_ids(Path(args.gt), first_sep=";", second_sep=" ")
    if args.max_query > 0:
        gt = gt[: args.max_query]
    k = args.count

    pf = {"prefetch_offset": args.prefetch_offset, "prefetch_lines": args.prefetch_lines}
    if args.index == "vamana":
        param = VamanaQueryParam(ef_search=args.ef, extra_params=pf)
    else:
        param = HnswQueryParam(ef=args.ef, extra_params=pf)

    col = zvec.open(args.collection, CollectionOption(read_only=True, enable_mmap=True))
    raw = col._obj
    raw.ann_bench_prepare(VECTOR_FIELD)
    raw.ann_bench_set_query_params(param)

    paths = {
        "raw_fast_query_doc_ids_only": lambda v: raw.fast_query_doc_ids_only(
            VECTOR_FIELD, v, k, param),
        "raw_ann_bench_search_doc_ids_only": lambda v: raw.ann_bench_search_doc_ids_only(v, k),
        "collection_fast_query_doc_ids_only": lambda v: col.fast_query_doc_ids_only(
            VECTOR_FIELD, v, topk=k, param=param),
        "collection_ann_bench_search_doc_ids_only": lambda v: col.ann_bench_search_doc_ids_only(
            v, topk=k),
    }

    print(f"collection={args.collection}")
    print(f"queries={len(X)} k={k} ef={args.ef} runs={args.runs}")
    print(f"index={args.index} prefetch={args.prefetch_offset}/{args.prefetch_lines}")
    print()

    # --- correctness ---
    print("=== correctness (ids must match raw_fast_query) ===")
    ref = [paths["raw_fast_query_doc_ids_only"](v) for v in X[: min(500, len(X))]]
    all_ok = True
    for name, fn in paths.items():
        if name == "raw_fast_query_doc_ids_only":
            continue
        mismatch = 0
        for i, v in enumerate(X[: len(ref)]):
            got = fn(v)
            if not np.array_equal(ref[i], got):
                mismatch += 1
        status = "OK" if mismatch == 0 else f"FAIL ({mismatch} mismatches)"
        print(f"  {name:42s}  {status}")
        all_ok = all_ok and mismatch == 0

    ref_full = [paths["raw_fast_query_doc_ids_only"](v) for v in X]
    rec = recall_at_k(ref_full, gt, k)
    print(f"  recall@10 (reference path): {rec:.4f}")
    if not all_ok:
        return 1
    print()

    # --- performance (per-query, ann-benchmarks timing) ---
    print("=== performance (per-query perf_counter, ann-benchmarks style) ===")
    baseline_dt = None
    rows = []
    for name, fn in paths.items():
        dt = bench_per_query(fn, X, args.runs)
        qps = 1.0 / dt
        if baseline_dt is None:
            baseline_dt = dt
            overhead = 0.0
        else:
            overhead = (dt - baseline_dt) / baseline_dt * 100
        rows.append((name, rec, qps, dt * 1e6, overhead))
    rows.sort(key=lambda r: -r[2])
    print(f"{'path':42s}  {'recall':>7s}  {'QPS':>10s}  {'us':>8s}  {'vs fast':>8s}")
    print("-" * 82)
    fast_qps = next(q for n, _, q, _, _ in rows if n == "raw_fast_query_doc_ids_only")
    for name, r, qps, us, _ in rows:
        vs = (qps / fast_qps - 1.0) * 100 if fast_qps else 0.0
        print(f"{name:42s}  {r:7.4f}  {qps:10.1f}  {us:8.1f}  {vs:+7.1f}%")
    print()

    # batch reference (different measurement category)
    print("=== batch reference (not single-query comparable) ===")
    for v in X[:200]:
        raw.fast_query_doc_ids_only(VECTOR_FIELD, v, k, param)
    t0 = time.perf_counter()
    for _ in range(args.runs):
        t_loop = time.perf_counter()
        out = raw.batch_fast_query_doc_ids_only(VECTOR_FIELD, X, k, param)
        dt_batch = (time.perf_counter() - t_loop) / len(X)
    qps_batch = 1.0 / dt_batch
    print(f"  batch_fast_query_doc_ids_only:  QPS={qps_batch:,.1f}  "
          f"(+{(qps_batch/fast_qps-1)*100:.1f}% vs raw_fast)")
    print()

    # ann-benchmarks adapter simulation (3-stage, timed = search only)
    print("=== ann-benchmarks adapter simulation (timed search only) ===")
    sys.path.insert(0, str(Path("/root/py/ann-benchmarks")))
    from ann_benchmarks.algorithms.zvec.module import (  # noqa: E402
        ZvecAnnBenchDocIds,
        ZvecFastQueryDocIds,
    )

    class _OpenAdapter:
        """Use pre-opened collection; skip fit."""

        def __init__(self, cls, collection, param):
            self._col = collection
            self._raw = collection._obj
            self._param = param
            self._q = None
            self._n = 0
            self._res = None
            self._cls = cls

        def prepare_query(self, v, n):
            self._q = np.ascontiguousarray(v, dtype=np.float32)
            self._n = n

        def run_prepared_query(self):
            if self._cls is ZvecAnnBenchDocIds:
                self._res = self._raw.ann_bench_search_doc_ids_only(self._q, self._n)
            else:
                self._res = self._raw.fast_query_doc_ids_only(
                    VECTOR_FIELD, self._q, self._n, self._param)

        def get_prepared_query_results(self):
            return self._res

    for label, cls in [
        ("ZvecFastQueryDocIds path", ZvecFastQueryDocIds),
        ("ZvecAnnBenchDocIds path", ZvecAnnBenchDocIds),
    ]:
        ad = _OpenAdapter(cls, col, param)
        if cls is ZvecAnnBenchDocIds:
            raw.ann_bench_set_query_params(param)
        best = float("inf")
        res = None
        for _ in range(args.runs):
            tsum = 0.0
            results = []
            for v in X:
                ad.prepare_query(v, k)
                t0 = time.perf_counter()
                ad.run_prepared_query()
                tsum += time.perf_counter() - t0
                results.append(ad.get_prepared_query_results())
            dt = tsum / len(X)
            if dt < best:
                best = dt
                res = results
        rec_ad = recall_at_k(res, gt, k)
        print(f"  {label:32s}  recall={rec_ad:.4f}  QPS={1/best:,.1f}  "
              f"({(1/best/fast_qps-1)*100:+.1f}% vs raw_fast)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
