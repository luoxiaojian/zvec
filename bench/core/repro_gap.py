#!/usr/bin/env python3
"""Reproduce the ann-benchmarks zvec point in this independent harness.

Goal: isolate whether the ann-benchmarks 27,325 QPS @ recall>=0.90 vs the
independent ~31,800 QPS gap is (a) measurement-framework overhead or
(b) purely a config choice (Vamana md=64 vs HNSW m16 + coarse ef grid).

Opens a prebuilt collection directly (same raw _obj + fast_query_doc_ids_only
path as ann_benchmarks/algorithms/zvec/module.py) and times an ef sweep.
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


def recall_at_k(results, gt, k=10):
    hit = 0
    tot = 0
    for res, g in zip(results, gt):
        s = set(int(x) for x in g[:k])
        hit += sum(1 for x in res[:k] if int(x) in s)
        tot += k
    return hit / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--index", choices=["hnsw", "vamana"], required=True)
    ap.add_argument("--ef", required=True, help="comma ef list")
    ap.add_argument("--prefetch-offset", type=int, default=64)
    ap.add_argument("--prefetch-lines", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--per-query-timing", action="store_true",
                    help="time each query separately (ann-benchmarks methodology)")
    ap.add_argument("--batch", action="store_true",
                    help="time batch_fast_query_doc_ids_only (one C++ call, GIL released)")
    ap.add_argument("--ann-bench", action="store_true",
                    help="use ann_bench_search_doc_ids_only (cached indexers; params via --ef)")
    ap.add_argument("--query", default="/root/zvec_workspace/output/sift_test.txt")
    ap.add_argument("--gt", default="/root/zvec_workspace/output/sift_neighbors.txt")
    args = ap.parse_args()

    queries = load_query_vectors(Path(args.query), first_sep=";", second_sep=" ")
    X = np.ascontiguousarray(np.stack(queries), dtype=np.float32)
    gt = load_ground_truth_ids(Path(args.gt), first_sep=";", second_sep=" ")
    print(f"queries={len(X)} dim={X.shape[1]} count={args.count} runs={args.runs}")

    col = zvec.open(args.collection, CollectionOption(read_only=True, enable_mmap=True))
    raw = col._obj

    if args.ann_bench:
        raw.ann_bench_prepare(VECTOR_FIELD)

    def make_param(ef):
        pf = {"prefetch_offset": args.prefetch_offset,
              "prefetch_lines": args.prefetch_lines}
        if args.index == "vamana":
            return VamanaQueryParam(ef_search=int(ef), extra_params=pf)
        return HnswQueryParam(ef=int(ef), extra_params=pf)

    # warmup
    p0 = make_param(int(args.ef.split(",")[0]))
    if args.ann_bench:
        raw.ann_bench_set_query_params(p0)
        for v in X[:2000]:
            raw.ann_bench_search_doc_ids_only(v, args.count)
    else:
        for v in X[:2000]:
            raw.fast_query_doc_ids_only(VECTOR_FIELD, v, args.count, p0)

    mode = "ann_bench" if args.ann_bench else ("batch" if args.batch else "fast")
    print(f"mode={mode}")

    print(f"{'ef':>5}  {'recall@10':>9}  {'QPS':>10}  {'mean(us)':>9}")
    for ef in [int(x) for x in args.ef.split(",") if x]:
        param = make_param(ef)
        if args.ann_bench:
            raw.ann_bench_set_query_params(param)
        best_t = float("inf")
        best_res = None
        if args.batch and not args.ann_bench:
            best = float("inf")
            best_res = None
            for _ in range(args.runs):
                t0 = time.perf_counter()
                out = raw.batch_fast_query_doc_ids_only(VECTOR_FIELD, X, args.count, param)
                dt = (time.perf_counter() - t0) / len(X)
                if dt < best:
                    best = dt
                    best_res = [out[i] for i in range(len(out))]
            rec = recall_at_k(best_res, gt, args.count)
            print(f"{ef:5d}  {rec*100:8.2f}%  {1.0/best:10.1f}  {best*1e6:9.2f}  [batch]")
            continue
        for _ in range(args.runs):
            res = []
            if args.per_query_timing:
                tsum = 0.0
                for v in X:
                    start = time.perf_counter()
                    if args.ann_bench:
                        r = raw.ann_bench_search_doc_ids_only(v, args.count)
                    else:
                        r = raw.fast_query_doc_ids_only(
                            VECTOR_FIELD, v, args.count, param)
                    tsum += time.perf_counter() - start
                    res.append(r)
                dt = tsum / len(X)
            else:
                t0 = time.perf_counter()
                for v in X:
                    if args.ann_bench:
                        res.append(raw.ann_bench_search_doc_ids_only(v, args.count))
                    else:
                        res.append(
                            raw.fast_query_doc_ids_only(
                                VECTOR_FIELD, v, args.count, param))
                dt = (time.perf_counter() - t0) / len(X)
            if dt < best_t:
                best_t = dt
                best_res = res
        rec = recall_at_k(best_res, gt, args.count)
        print(f"{ef:5d}  {rec*100:8.2f}%  {1.0/best_t:10.1f}  {best_t*1e6:9.2f}")


if __name__ == "__main__":
    main()
