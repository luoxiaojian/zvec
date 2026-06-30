#!/usr/bin/env python3
"""Benchmark zvec using the ann-benchmarks calling convention.

This is the single, self-contained bench: it covers the full ann-benchmarks
workflow (fit -> query -> recall) for one config:

* ``--build`` performs the fit step (insert + optimize) from ``build.yaml``,
  inserting vectors in row order so the internal doc id equals the dataset row
  index (the contract ann-benchmarks relies on).
* A BaseANN-style adapter (``ZvecANN``) exposes the 3-stage query protocol
  ``prepare_query`` / ``run_prepared_query`` / ``get_prepared_query_results``,
  so only the search itself is timed (argument prep is excluded), exactly like
  the high-scoring leaderboard bindings (glass, kgn, ...).
* ``--path fast_query_doc_ids`` (default) uses ``fast_query_doc_ids_only``.
* ``--path ann_bench_doc_ids`` uses ``ann_bench_prepare`` +
  ``ann_bench_search_doc_ids_only`` (same as ann-benchmarks ``ZvecAnnBenchDocIds``).
* QPS = 1 / best_search_time, taking the best (min mean latency) over
  ``--runs`` repetitions, identical to ann-benchmarks.
* Recall@k = |returned[:k] ∩ ground_truth[:k]| / k, averaged over queries.

This measures the *search-only* path on float32 numpy query rows (zero per-call
list->numpy conversion), matching how ann-benchmarks feeds ``X_test`` rows.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common.adapters import (
    ZvecFastQueryDocIdsANN,
    _param_str,
    make_search_adapter,
)
from common.config import (
    load_build_config,
    load_search_config,
    resolve_workspace_paths,
)
from common.io import iter_vecs_batches, load_ground_truth_ids, load_query_vectors
from common.schema import VECTOR_FIELD
from common.zvec_bench import ZvecBenchClient

from zvec import HnswQueryParam, VamanaQueryParam


# Backward-compatible alias used by compare_ann_bench_paths.py imports.
ZvecANN = ZvecFastQueryDocIdsANN


def _recall(per_query_ids, ground_truth, topk_values):
    """k-recall@k = |returned[:k] ∩ truth[:k]| / k, averaged over queries."""
    out = {}
    nq = len(per_query_ids)
    for k in topk_values:
        total = 0.0
        for res, gt in zip(per_query_ids, ground_truth):
            gt_slice = gt[:k]
            if not gt_slice:
                continue
            knn = set(int(x) for x in res[:k])
            hit = sum(1 for g in gt_slice if g in knn)
            total += hit / k
        out[k] = 100.0 * total / nq if nq else 0.0
    return out


def run_individual_query(algo, X_test, count: int, run_count: int):
    """Faithful port of ann-benchmarks runner.single_query loop (search-only)."""
    best_search_time = float("inf")
    last_results = None
    per_query_times = None
    for r in range(run_count):
        results = []
        times = []
        for v in X_test:
            algo.prepare_query(v, count)
            start = time.perf_counter()
            algo.run_prepared_query()
            elapsed = time.perf_counter() - start
            cand = algo.get_prepared_query_results()
            # ann-benchmarks asserts uniqueness of returned indices
            assert len(cand) == len(set(int(x) for x in cand)), (
                "duplicated candidates returned"
            )
            results.append(cand)
            times.append(elapsed)
        total_time = sum(times)
        search_time = total_time / len(X_test)
        if search_time < best_search_time:
            best_search_time = search_time
            last_results = results
            per_query_times = times
        print(
            f"  run {r + 1}/{run_count}: "
            f"mean={search_time * 1e6:.2f}us qps={1.0 / search_time:.1f}"
        )
    return best_search_time, last_results, per_query_times


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sift", "gist"], default=None)
    parser.add_argument("--workspace-config", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--runs", type=int, default=3, help="run_count (best is reported)."
    )
    parser.add_argument(
        "--count", type=int, default=None, help="topk (default: max of config)."
    )
    parser.add_argument(
        "--ef",
        type=str,
        default=None,
        help="Comma-separated ef sweep (default: ef from search_po_pl.yaml).",
    )
    parser.add_argument(
        "--path",
        choices=["fast_query_doc_ids", "ann_bench_doc_ids"],
        default="fast_query_doc_ids",
        help="Search API path (default: fast_query_doc_ids).",
    )
    parser.add_argument(
        "--legacy-ids",
        action="store_true",
        help="With --path fast_query_doc_ids, use fast_query_doc_ids (ids+scores).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Fit step: build the collection from build.yaml before benchmarking.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --build, drop an existing collection before rebuilding.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Insert batch size for --build."
    )
    return parser.parse_args()


def build_collection(dataset: str, args: argparse.Namespace) -> None:
    """ann-benchmarks fit step: insert (row order) + optimize from build.yaml."""
    cfg = load_build_config(
        dataset,
        config_path=args.config,
        workspace_config_dir=args.workspace_config,
        data_dir=args.data_dir,
        output_dir=Path("./output"),
    )
    if not cfg.build_file.exists():
        raise SystemExit(f"build file not found: {cfg.build_file}")

    collection_path = cfg.collection_path
    if args.force and collection_path.exists():
        print(f"removing existing collection: {collection_path}")
        ZvecBenchClient.from_build_config(cfg).drop_old()
        if collection_path.exists():
            shutil.rmtree(collection_path)
    if collection_path.exists():
        raise SystemExit(
            f"collection already exists: {collection_path} (pass --force)"
        )
    collection_path.parent.mkdir(parents=True, exist_ok=True)

    client = ZvecBenchClient.from_build_config(cfg)
    total = 0
    t0 = time.perf_counter()
    with client.build_session():
        for keys, vectors in iter_vecs_batches(
            cfg.build_file, cfg.dimension, args.batch_size
        ):
            client.insert_batch(keys, vectors)
            total += len(keys)
            print(f"inserted {total} vectors ...")
        print("optimizing index ...")
        client.optimize()
    print(
        f"build finished: {total} vectors -> {collection_path} "
        f"({time.perf_counter() - t0:.1f}s)\n"
    )


def main() -> int:
    args = parse_args()
    if args.workspace_config is None and args.dataset is None:
        print("provide --workspace-config or --dataset", file=sys.stderr)
        return 1

    dataset = args.dataset
    if dataset is None and args.workspace_config is not None:
        dataset, _, _ = resolve_workspace_paths(args.workspace_config)

    if args.build:
        build_collection(dataset, args)

    cfg = load_search_config(
        dataset,
        config_path=args.config,
        workspace_config_dir=args.workspace_config,
        data_dir=args.data_dir,
        output_dir=Path("./output"),
        collection_path=None,
    )

    queries = load_query_vectors(
        cfg.query_file,
        first_sep=cfg.query_first_sep,
        second_sep=cfg.query_second_sep,
    )
    # Mirror ann-benchmarks: a contiguous float32 X_test; iterate rows.
    X_test = np.ascontiguousarray(np.stack(queries), dtype=np.float32)
    gt = load_ground_truth_ids(
        cfg.ground_truth_file,
        first_sep=cfg.ground_truth_first_sep,
        second_sep=cfg.ground_truth_second_sep,
    )
    topk_list = cfg.topk_list
    count = args.count if args.count is not None else max(topk_list)

    # Build the ef sweep (each ef -> one (recall, QPS) leaderboard point).
    base = ZvecBenchClient.from_search_config(cfg)
    if args.ef:
        efs = [int(x) for x in args.ef.split(",") if x]
        params = [HnswQueryParam(ef=e) for e in efs]
    else:
        params = [base.query_param]

    client = ZvecBenchClient.from_search_config(cfg)

    print(f"=== ann-benchmarks-style bench: zvec on {cfg.dataset} "
          f"(path={args.path}) ===")
    print(f"queries={len(X_test)} dim={X_test.shape[1]} count={count} "
          f"runs={args.runs}\n")

    rows = []
    with client.init():
        algo = make_search_adapter(
            args.path,
            client.collection._obj,
            params[0],
            field=VECTOR_FIELD,
            legacy_ids=args.legacy_ids,
        )
        # warmup (JIT caches, page-ins) - not timed
        for v in X_test[: min(2000, len(X_test))]:
            algo.prepare_query(v, count)
            algo.run_prepared_query()

        for param in params:
            algo.set_query_arguments(param)
            print(f"[{_param_str(param)}]")
            best_t, results, times = run_individual_query(
                algo, X_test, count, args.runs
            )
            recall = _recall(results, gt, topk_list)
            qps = 1.0 / best_t
            rows.append((_param_str(param), recall, qps, best_t, times))
            print()

    print("=== results (best of runs, search-only) ===")
    header = "  ".join(f"recall@{k}" for k in topk_list)
    print(f"{'param':22s}  {header}  {'QPS':>10s}  {'mean(us)':>9s}  "
          f"{'p99(us)':>9s}")
    for name, recall, qps, best_t, times in rows:
        rec_str = "  ".join(f"{recall[k]:7.2f}%" for k in topk_list)
        print(
            f"{name:22s}  {rec_str}  {qps:10.1f}  {best_t * 1e6:9.2f}  "
            f"{pct(times, 0.99) * 1e6:9.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
