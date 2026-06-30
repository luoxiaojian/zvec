#!/usr/bin/env python3
"""Benchmark zvec using workspace configs with tuned PO/PL parameters.

Runs all query modes (query, fast_query, fast_query_doc_ids, ann_bench_doc_ids)
workspace index configurations, sweeping ef values to trace the full
recall-QPS curve.  This reproduces the ann-benchmarks methodology
(search-only timing, best of N runs) while applying the workspace's
tuned prefetch_offset/prefetch_lines parameters.

Usage:
    python run_workspace_bench.py --dataset sift --runs 5 --count 10
    python run_workspace_bench.py --dataset gist --runs 5 --count 10
    python run_workspace_bench.py --dataset sift --config-dir /path/to/vamana_d64
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common.adapters import ZvecAnnBenchDocIdsANN, ZvecFastQueryDocIdsANN
from common.config import load_build_config, load_search_config, resolve_workspace_paths
from common.io import iter_vecs_batches, load_ground_truth_ids, load_query_vectors
from common.schema import VECTOR_FIELD, make_query_param
from common.zvec_bench import ZvecBenchClient

import zvec
from zvec import HnswQueryParam, VamanaQueryParam, Query

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path("/root/main/workspace")
COLLECTION_ROOT = Path("/root/py/workspace")  # Python collections cache
DATA_DIR = Path("/root/zvec_workspace/output")

# ef sweeps matching ann-benchmarks config.yml
HNSW_EF_SWEEP = [10, 16, 20, 30, 40, 60, 80, 120, 160, 200, 300, 400]
VAMANA_EF_SWEEP = [10, 16, 25, 40, 60, 80, 120, 160, 250, 400, 600]

QUERY_MODES = [
    "fast_query_doc_ids",
    "ann_bench_doc_ids",
    "fast_query",
    "query",
]


# ---------------------------------------------------------------------------
# Workspace config helpers
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceConfig:
    """Parsed workspace config for one index."""
    name: str
    dataset: str
    index_type: str  # "hnsw" or "vamana"
    config_dir: Path
    collection_path: Path
    # Tuned PO/PL from search_po_pl.yaml
    prefetch_offset: int
    prefetch_lines: int
    # Fixed ef from workspace (for reference)
    fixed_ef: int
    # Build params
    m_or_degree: int


def parse_workspace_config(config_dir: Path) -> WorkspaceConfig:
    """Parse a workspace config directory."""
    import yaml

    dataset = config_dir.parent.name
    name = config_dir.name

    search_yaml = config_dir / "search_po_pl.yaml"
    with search_yaml.open() as f:
        raw = yaml.safe_load(f)

    query_param = json.loads(raw["QueryConfig"]["QueryParam"])
    index_config = json.loads(raw["IndexCommon"]["IndexConfig"])

    index_type_raw = query_param.get("index_type", "")
    if "Vamana" in index_type_raw or "vamana" in index_type_raw:
        index_type = "vamana"
        ef = int(query_param.get("ef_search", 200))
        m_or_degree = int(index_config.get("max_degree", 48))
    else:
        index_type = "hnsw"
        ef = int(query_param.get("ef_search", query_param.get("ef", 300)))
        m_or_degree = int(index_config.get("m", 48))

    po = int(query_param.get("prefetch_offset", 0))
    pl = int(query_param.get("prefetch_lines", 0))

    collection_path = COLLECTION_ROOT / dataset / name / "collection"

    return WorkspaceConfig(
        name=name,
        dataset=dataset,
        index_type=index_type,
        config_dir=config_dir,
        collection_path=collection_path,
        prefetch_offset=po,
        prefetch_lines=pl,
        fixed_ef=ef,
        m_or_degree=m_or_degree,
    )


def discover_workspace_configs(dataset: str) -> list[WorkspaceConfig]:
    """Discover all workspace configs for a dataset."""
    ds_dir = WORKSPACE_ROOT / dataset
    configs = []
    for d in sorted(ds_dir.iterdir()):
        if d.is_dir() and (d / "search_po_pl.yaml").exists():
            configs.append(parse_workspace_config(d))
    return configs


# ---------------------------------------------------------------------------
# Collection build
# ---------------------------------------------------------------------------

def build_collection_if_needed(
    ws: WorkspaceConfig, batch_size: int = 1024, force: bool = False
) -> None:
    """Build the Python collection if it doesn't exist."""
    if ws.collection_path.exists() and not force:
        print(f"  [skip] collection exists: {ws.collection_path}")
        return

    if force and ws.collection_path.exists():
        shutil.rmtree(ws.collection_path)

    cfg = load_build_config(
        ws.dataset,
        workspace_config_dir=ws.config_dir,
        data_dir=DATA_DIR,
    )
    # Override collection path to our cache location
    cfg = cfg.__class__(
        dataset=cfg.dataset,
        build_file=cfg.build_file,
        collection_path=ws.collection_path,
        thread_count=cfg.thread_count,
        dimension=cfg.dimension,
        metric_name=cfg.metric_name,
        index_type=cfg.index_type,
        quantizer=cfg.quantizer,
        disable_id_map=cfg.disable_id_map,
        builder_params=cfg.builder_params,
    )

    ws.collection_path.parent.mkdir(parents=True, exist_ok=True)
    client = ZvecBenchClient.from_build_config(cfg)
    total = 0
    t0 = time.perf_counter()
    with client.build_session():
        for keys, vectors in iter_vecs_batches(
            cfg.build_file, cfg.dimension, batch_size
        ):
            client.insert_batch(keys, vectors)
            total += len(keys)
            if total % 100000 == 0:
                print(f"    inserted {total} vectors ...")
        print("    optimizing index ...")
        client.optimize()
    elapsed = time.perf_counter() - t0
    print(f"  [built] {total} vectors in {elapsed:.1f}s -> {ws.collection_path}")


# ---------------------------------------------------------------------------
# Query runners
# ---------------------------------------------------------------------------

def make_param(ws: WorkspaceConfig, ef: int) -> HnswQueryParam | VamanaQueryParam:
    """Create query param with tuned PO/PL."""
    extra = {}
    if ws.prefetch_offset > 0:
        extra["prefetch_offset"] = ws.prefetch_offset
    if ws.prefetch_lines > 0:
        extra["prefetch_lines"] = ws.prefetch_lines

    if ws.index_type == "vamana":
        return VamanaQueryParam(ef_search=ef, extra_params=extra)
    else:
        return HnswQueryParam(ef=ef, extra_params=extra)


def run_query_mode(
    collection: Any,
    mode: str,
    X_test: np.ndarray,
    count: int,
    param: Any,
    runs: int,
) -> tuple[float, list]:
    """Run one query mode, return (best_mean_time, results_from_best_run)."""
    if mode in ("fast_query_doc_ids", "ann_bench_doc_ids"):
        return _run_adapter_mode(collection, mode, X_test, count, param, runs)

    best_search_time = float("inf")
    best_results = None

    for r in range(runs):
        results = []
        times = []
        for v in X_test:
            q = np.ascontiguousarray(v, dtype=np.float32)

            start = time.perf_counter()
            if mode == "fast_query":
                str_ids, _scores = collection.fast_query(
                    VECTOR_FIELD, q, topk=count, param=param
                )
                res = np.array([int(x) for x in str_ids], dtype=np.int64)
            elif mode == "query":
                docs = collection.query(
                    queries=Query(
                        field_name=VECTOR_FIELD,
                        vector=q.tolist(),
                        param=param,
                    ),
                    topk=count,
                    output_fields=[],
                )
                res = np.array(
                    [int(doc.id) for doc in docs] if docs else [],
                    dtype=np.int64,
                )
            else:
                raise ValueError(f"unknown mode: {mode}")
            elapsed = time.perf_counter() - start

            results.append(res)
            times.append(elapsed)

        mean_time = sum(times) / len(times)
        if mean_time < best_search_time:
            best_search_time = mean_time
            best_results = results

    return best_search_time, best_results


def _run_adapter_mode(
    collection: Any,
    mode: str,
    X_test: np.ndarray,
    count: int,
    param: Any,
    runs: int,
) -> tuple[float, list]:
    """3-stage ann-benchmarks timing for fast_query_doc_ids / ann_bench_doc_ids."""
    raw = collection._obj
    if mode == "ann_bench_doc_ids":
        algo = ZvecAnnBenchDocIdsANN(raw, param)
    else:
        algo = ZvecFastQueryDocIdsANN(raw, param, ids_only=True)

    best_search_time = float("inf")
    best_results = None

    for _ in range(runs):
        results = []
        times = []
        for v in X_test:
            algo.prepare_query(v, count)
            start = time.perf_counter()
            algo.run_prepared_query()
            elapsed = time.perf_counter() - start
            results.append(algo.get_prepared_query_results())
            times.append(elapsed)

        mean_time = sum(times) / len(times)
        if mean_time < best_search_time:
            best_search_time = mean_time
            best_results = results

    return best_search_time, best_results


def recall_at_k(results: list, ground_truth: list, k: int) -> float:
    """Compute recall@k averaged over all queries."""
    nq = len(results)
    total = 0.0
    for res, gt in zip(results, ground_truth):
        gt_set = set(int(x) for x in gt[:k])
        if not gt_set:
            continue
        knn = set(int(x) for x in res[:k])
        hit = len(knn & gt_set)
        total += hit / k
    return total / nq if nq else 0.0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    ws: WorkspaceConfig,
    modes: list[str],
    runs: int,
    count: int,
) -> dict[str, list[dict]]:
    """Run full benchmark for one workspace config."""

    # Load query data
    cfg = load_search_config(
        ws.dataset,
        workspace_config_dir=ws.config_dir,
        data_dir=DATA_DIR,
        collection_path=ws.collection_path,
    )

    queries = load_query_vectors(
        cfg.query_file,
        first_sep=cfg.query_first_sep,
        second_sep=cfg.query_second_sep,
    )
    X_test = np.ascontiguousarray(np.stack(queries), dtype=np.float32)
    gt = load_ground_truth_ids(
        cfg.ground_truth_file,
        first_sep=cfg.ground_truth_first_sep,
        second_sep=cfg.ground_truth_second_sep,
    )

    ef_sweep = VAMANA_EF_SWEEP if ws.index_type == "vamana" else HNSW_EF_SWEEP

    results = {m: [] for m in modes}

    # Open collection
    col = zvec.open(str(ws.collection_path), zvec.CollectionOption(read_only=True, enable_mmap=True))

    # Warmup
    warmup_param = make_param(ws, ef_sweep[len(ef_sweep) // 2])
    for v in X_test[:min(2000, len(X_test))]:
        q = np.ascontiguousarray(v, dtype=np.float32)
        col.fast_query_doc_ids(VECTOR_FIELD, q, topk=count, param=warmup_param)

    for ef in ef_sweep:
        param = make_param(ws, ef)
        ef_label = f"ef={ef}" if ws.index_type == "hnsw" else f"ef_search={ef}"

        for mode in modes:
            best_t, res_list = run_query_mode(
                col, mode, X_test, count, param, runs
            )
            rec = recall_at_k(res_list, gt, count)
            qps = 1.0 / best_t
            results[mode].append({
                "ef": ef,
                "recall": rec,
                "qps": qps,
                "mean_us": best_t * 1e6,
            })
            print(
                f"    {mode:25s} {ef_label:14s} "
                f"recall={rec:.4f}  QPS={qps:,.1f}  mean={best_t*1e6:.1f}us"
            )

    col = None  # release
    return results


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def find_best_qps_at_recall(
    rows: list[dict], threshold: float
) -> dict | None:
    """Find max QPS row where recall >= threshold."""
    candidates = [r for r in rows if r["recall"] >= threshold]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["qps"])


def print_summary(
    all_results: dict[str, dict[str, list[dict]]],
    dataset: str,
    configs: list[WorkspaceConfig],
):
    """Print summary table comparing with benchmark_comparison.md results."""
    thresholds = [0.90, 0.95, 0.98, 0.99]

    print(f"\n{'='*80}")
    print(f"SUMMARY: {dataset.upper()} - Best QPS at recall thresholds (with PO/PL)")
    print(f"{'='*80}\n")

    for mode in QUERY_MODES:
        print(f"\n--- zvec-{mode} ---\n")
        header = f"{'Config':<30s}"
        for t in thresholds:
            header += f" | {'≥'+f'{t:.2f}':>12s}"
        print(header)
        print("-" * len(header))

        # Aggregate best across all configs for each threshold
        best_per_threshold: dict[float, dict | None] = {t: None for t in thresholds}

        for ws in configs:
            key = ws.name
            if key not in all_results or mode not in all_results[key]:
                continue
            rows = all_results[key][mode]
            line = f"{key:<30s}"
            for t in thresholds:
                best = find_best_qps_at_recall(rows, t)
                if best:
                    line += f" | {best['qps']:>10,.1f}"
                    if (best_per_threshold[t] is None or
                            best["qps"] > best_per_threshold[t]["qps"]):
                        best_per_threshold[t] = {**best, "config": key}
                else:
                    line += f" | {'N/A':>10s}"
            print(line)

        # Print overall best
        print()
        line_best = f"{'** BEST **':<30s}"
        for t in thresholds:
            b = best_per_threshold[t]
            if b:
                line_best += f" | {b['qps']:>10,.1f}"
            else:
                line_best += f" | {'N/A':>10s}"
        print(line_best)

        line_cfg = f"{'   (config)':<30s}"
        for t in thresholds:
            b = best_per_threshold[t]
            if b:
                line_cfg += f" | {b['config'][:10]:>10s}"
            else:
                line_cfg += f" | {'':>10s}"
        print(line_cfg)


def save_results_json(
    all_results: dict, dataset: str, output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset}_workspace_bench_results.json"
    with out_file.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["sift", "gist"], required=True
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Run only one specific config dir (e.g. vamana_i8_c500_a1.5_d64)",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of runs (best reported). Default: 5",
    )
    parser.add_argument(
        "--count", type=int, default=10,
        help="Top-k. Default: 10",
    )
    parser.add_argument(
        "--modes", type=str,
        default="fast_query_doc_ids,ann_bench_doc_ids,fast_query,query",
        help="Comma-separated query modes to test.",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build missing collections before benchmarking.",
    )
    parser.add_argument(
        "--force-build", action="store_true",
        help="Force rebuild even if collection exists.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/root/py/zvec/bench/core/results"),
    )
    parser.add_argument(
        "--skip-build-only", action="store_true",
        help="Skip configs whose collections don't exist (no build).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",")]

    if args.config_dir:
        configs = [parse_workspace_config(args.config_dir)]
    else:
        configs = discover_workspace_configs(args.dataset)

    print(f"=== Workspace Bench: {args.dataset.upper()} ===")
    print(f"Configs: {len(configs)}")
    print(f"Modes: {modes}")
    print(f"Runs: {args.runs}, Count: {args.count}")
    print()

    # Build phase
    for ws in configs:
        print(f"[{ws.name}] index_type={ws.index_type} "
              f"{'M' if ws.index_type == 'hnsw' else 'degree'}={ws.m_or_degree} "
              f"PO={ws.prefetch_offset} PL={ws.prefetch_lines}")
        if args.build or args.force_build:
            build_collection_if_needed(ws, force=args.force_build)
        elif not ws.collection_path.exists():
            if args.skip_build_only:
                print(f"  [skip] no collection, skipping benchmark")
            else:
                print(f"  [WARN] collection not found: {ws.collection_path}")
                print(f"         Run with --build to create it.")

    print()

    # Benchmark phase
    all_results = {}
    for ws in configs:
        if not ws.collection_path.exists():
            print(f"[{ws.name}] SKIP (no collection)")
            continue

        print(f"\n{'='*60}")
        print(f"[{ws.name}] Benchmarking (PO={ws.prefetch_offset}, PL={ws.prefetch_lines})")
        print(f"{'='*60}")

        try:
            results = run_benchmark(ws, modes, args.runs, args.count)
            all_results[ws.name] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    if all_results:
        print_summary(all_results, args.dataset, configs)
        save_results_json(all_results, args.dataset, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
