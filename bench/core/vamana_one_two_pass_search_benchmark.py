#!/usr/bin/env python3
"""Compare one-pass and two-pass Vamana search at aligned recall thresholds.

The timing path mirrors ann-benchmarks' ``ZvecAnnBenchDocIds`` adapter:

* open an existing collection read-only with mmap;
* call ``ann_bench_prepare`` once;
* bind each Vamana query parameter once;
* time only ``ann_bench_search_fast`` with a pre-allocated int64 output buffer;
* run every setting multiple times and retain the fastest mean query time.

For each index the benchmark has two phases:

1. With PO=0 and PL=0, binary-search the minimum integer ef that reaches each
   requested recall threshold.
2. At every distinct minimum ef, sweep legal PO/PL combinations and select the
   highest-QPS point that still reaches the corresponding threshold.

Results are checkpointed after every new measurement and can be resumed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zvec
from zvec import CollectionOption, LogLevel, VamanaQueryParam

_SCRIPT_DIR = Path(__file__).resolve().parent

from common.adapters import ZvecAnnBenchFastANN
from common.io import load_ground_truth_ids, load_query_vectors
from common.prefetch_bounds import prefetch_sweep_for_build
from common.schema import VECTOR_FIELD


DEFAULT_INDEX_ROOT = Path("/root/py/zvec/zvec_indices")
DEFAULT_INDEX_PREFIX = (
    "ann_bench_doc_ids_vamana_euclidean_d128_"
    "R64_L500_a1.5_{passes}_cm1_fcm0_uniform_uint8"
)
DEFAULT_DATA_DIR = Path("/root/zvec_workspace/output")
DEFAULT_OUTPUT = (
    _SCRIPT_DIR / "output" / "vamana_one_two_pass_search_results.json"
)
DEFAULT_THRESHOLDS = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999]
DEFAULT_PO_VALUES = [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128]
DEFAULT_PL_VALUES = [0, 1, 2, 4, 6, 8]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SearchPoint:
    ef: int
    prefetch_offset: int
    prefetch_lines: int
    recall: float
    qps: float
    mean_us: float


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_csv_ints(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers")
    return values


def parse_thresholds(raw: str) -> list[float]:
    values = sorted(
        {float(value.strip()) for value in raw.split(",") if value.strip()}
    )
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("recall thresholds must be in (0, 1]")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--one-pass-index",
        type=Path,
        default=DEFAULT_INDEX_ROOT
        / DEFAULT_INDEX_PREFIX.format(passes="1pass"),
    )
    parser.add_argument(
        "--two-pass-index",
        type=Path,
        default=DEFAULT_INDEX_ROOT
        / DEFAULT_INDEX_PREFIX.format(passes="2pass"),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10, help="Recall@k and top-k")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Timed full-query-set runs per point; fastest mean wins",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=100,
        help="Untimed warmup queries after each query-param change",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Use a query prefix for smoke tests; 0 uses all queries",
    )
    parser.add_argument(
        "--max-ef",
        type=int,
        default=2048,
        help="Maximum ef considered by the integer binary search",
    )
    parser.add_argument(
        "--recall-thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--po-values", type=parse_csv_ints, default=DEFAULT_PO_VALUES
    )
    parser.add_argument(
        "--pl-values", type=parse_csv_ints, default=DEFAULT_PL_VALUES
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse compatible point measurements from the output checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned search grid",
    )
    return parser.parse_args()


def recall_at_k(results: list[np.ndarray], ground_truth: list, count: int) -> float:
    """Mean top-k ID intersection, matching existing zvec grid benchmarks."""
    if not results:
        return 0.0
    hits = 0
    denominator = len(results) * count
    for result, truth in zip(results, ground_truth, strict=True):
        expected = {int(value) for value in truth[:count]}
        hits += sum(int(value) in expected for value in result[:count])
    return hits / denominator


def make_query_param(ef: int, po: int, pl: int) -> VamanaQueryParam:
    return VamanaQueryParam(
        ef_search=int(ef),
        extra_params={
            "prefetch_offset": int(po),
            "prefetch_lines": int(pl),
        },
    )


class ExistingIndexBenchmark:
    def __init__(
        self,
        *,
        label: str,
        path: Path,
        expected_two_pass: bool,
        queries: np.ndarray,
        ground_truth: list,
        count: int,
        runs: int,
        warmup_queries: int,
    ) -> None:
        self.label = label
        self.path = path
        self.queries = queries
        self.ground_truth = ground_truth
        self.count = count
        self.runs = runs
        self.warmup_queries = min(max(0, warmup_queries), len(queries))

        self.collection = zvec.open(
            str(path), CollectionOption(read_only=True, enable_mmap=True)
        )
        params = self.collection.schema.vectors[0].index_param.to_dict()
        actual_two_pass = bool(params.get("two_pass_build", False))
        if actual_two_pass != expected_two_pass:
            raise ValueError(
                f"{label}: expected two_pass_build={expected_two_pass}, "
                f"found {actual_two_pass} in {path}"
            )
        doc_count = int(self.collection.stats.doc_count)
        if doc_count <= 0:
            raise ValueError(f"{label}: empty index: {path}")

        initial_param = make_query_param(max(count, 1), 0, 0)
        self.adapter = ZvecAnnBenchFastANN(
            self.collection._obj, initial_param, count=count
        )
        self.metadata = {
            "label": label,
            "path": str(path.resolve()),
            "doc_count": doc_count,
            "two_pass_build": actual_two_pass,
            "index_param": params,
        }

    def measure(self, ef: int, po: int, pl: int) -> SearchPoint:
        param = make_query_param(ef, po, pl)
        self.adapter.set_query_arguments(param)

        for query in self.queries[: self.warmup_queries]:
            self.adapter.prepare_query(query, self.count)
            self.adapter.run_prepared_query()

        best_total_ns: int | None = None
        best_results: list[np.ndarray] | None = None
        for _ in range(self.runs):
            total_ns = 0
            results: list[np.ndarray] = []
            for query in self.queries:
                # ann-benchmarks excludes prepare/get-results from timed search.
                self.adapter.prepare_query(query, self.count)
                started = time.perf_counter_ns()
                self.adapter.run_prepared_query()
                total_ns += time.perf_counter_ns() - started
                # The adapter intentionally reuses one output buffer.
                results.append(self.adapter.get_prepared_query_results().copy())
            if best_total_ns is None or total_ns < best_total_ns:
                best_total_ns = total_ns
                best_results = results

        assert best_total_ns is not None and best_results is not None
        mean_seconds = best_total_ns / len(self.queries) / 1_000_000_000
        return SearchPoint(
            ef=ef,
            prefetch_offset=po,
            prefetch_lines=pl,
            recall=recall_at_k(best_results, self.ground_truth, self.count),
            qps=1.0 / mean_seconds,
            mean_us=mean_seconds * 1_000_000,
        )

    def close(self) -> None:
        self.adapter = None
        self.collection = None


def point_key(ef: int, po: int, pl: int) -> str:
    return f"{ef}:{po}:{pl}"


def point_from_dict(raw: dict[str, Any]) -> SearchPoint:
    return SearchPoint(**raw)


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def find_minimum_integer_ef(
    *,
    threshold: float,
    minimum_ef: int,
    maximum_ef: int,
    measure: Callable[[int], SearchPoint],
) -> int | None:
    """Find the first passing integer ef, assuming the recall curve is monotone."""
    if measure(maximum_ef).recall < threshold:
        return None
    low, high = minimum_ef, maximum_ef
    while low < high:
        middle = (low + high) // 2
        if measure(middle).recall >= threshold:
            high = middle
        else:
            low = middle + 1

    # Explicitly verify the local boundary rather than trusting only the binary
    # search comparisons. This also handles a short passing plateau robustly.
    while low > minimum_ef and measure(low - 1).recall >= threshold:
        low -= 1
    return low


def benchmark_one_index(
    *,
    benchmark: ExistingIndexBenchmark,
    thresholds: list[float],
    max_ef: int,
    po_values: list[int],
    pl_values: list[int],
    previous: dict[str, Any] | None,
    save_progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    summary: dict[str, Any] = previous or {
        "index": benchmark.metadata,
        "ef_points": {},
        "minimum_ef": {},
        "prefetch_points": {},
        "best_by_threshold": {},
        "complete": False,
    }
    if summary.get("complete"):
        log(f"{benchmark.label}: checkpoint complete, skipping")
        return summary

    ef_cache = {
        int(key): point_from_dict(value)
        for key, value in summary.get("ef_points", {}).items()
    }
    prefetch_cache = {
        key: point_from_dict(value)
        for key, value in summary.get("prefetch_points", {}).items()
    }

    def measure_ef(ef: int) -> SearchPoint:
        point = ef_cache.get(ef)
        if point is None:
            point = benchmark.measure(ef, 0, 0)
            ef_cache[ef] = point
            summary["ef_points"][str(ef)] = asdict(point)
            log(
                f"{benchmark.label}: ef={ef:4d} PO=0 PL=0 "
                f"recall={point.recall:.4f} QPS={point.qps:,.1f}"
            )
            save_progress(summary)
        return point

    minimum_ef = max(benchmark.count, 1)
    log(f"{benchmark.label}: phase A — minimum integer ef")
    for threshold in thresholds:
        threshold_key = f"{threshold:.3f}"
        if threshold_key in summary["minimum_ef"]:
            continue
        value = find_minimum_integer_ef(
            threshold=threshold,
            minimum_ef=minimum_ef,
            maximum_ef=max_ef,
            measure=measure_ef,
        )
        summary["minimum_ef"][threshold_key] = value
        if value is None:
            log(f"{benchmark.label}: recall>={threshold_key} unreachable")
        else:
            point = measure_ef(value)
            log(
                f"{benchmark.label}: recall>={threshold_key} "
                f"minimum ef={value}, recall={point.recall:.4f}"
            )
        save_progress(summary)

    distinct_efs = sorted(
        {
            int(value)
            for value in summary["minimum_ef"].values()
            if value is not None
        }
    )
    combinations = list(itertools.product(po_values, pl_values))
    log(
        f"{benchmark.label}: phase B — {len(distinct_efs)} unique ef × "
        f"{len(combinations)} PO/PL points"
    )
    for ef in distinct_efs:
        for po, pl in combinations:
            key = point_key(ef, po, pl)
            if key in prefetch_cache:
                continue
            if po == 0 and pl == 0:
                point = measure_ef(ef)
            else:
                point = benchmark.measure(ef, po, pl)
                log(
                    f"{benchmark.label}: ef={ef:4d} PO={po:2d} PL={pl} "
                    f"recall={point.recall:.4f} QPS={point.qps:,.1f}"
                )
            prefetch_cache[key] = point
            summary["prefetch_points"][key] = asdict(point)
            save_progress(summary)

    for threshold in thresholds:
        threshold_key = f"{threshold:.3f}"
        ef = summary["minimum_ef"][threshold_key]
        if ef is None:
            summary["best_by_threshold"][threshold_key] = None
            continue
        candidates = [
            point
            for point in prefetch_cache.values()
            if point.ef == ef and point.recall >= threshold
        ]
        best = max(candidates, key=lambda point: point.qps) if candidates else None
        summary["best_by_threshold"][threshold_key] = (
            asdict(best) if best is not None else None
        )
        if best is not None:
            log(
                f"{benchmark.label}: BEST recall>={threshold_key}: "
                f"ef={best.ef} PO={best.prefetch_offset} "
                f"PL={best.prefetch_lines} recall={best.recall:.4f} "
                f"QPS={best.qps:,.1f}"
            )

    summary["ef_points"] = {
        str(ef): asdict(point) for ef, point in sorted(ef_cache.items())
    }
    summary["complete"] = True
    save_progress(summary)
    return summary


def build_comparison(
    summaries: dict[str, dict[str, Any]], thresholds: list[float]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for threshold in thresholds:
        key = f"{threshold:.3f}"
        one = summaries.get("one_pass", {}).get("best_by_threshold", {}).get(key)
        two = summaries.get("two_pass", {}).get("best_by_threshold", {}).get(key)
        if one is None or two is None:
            rows[key] = None
            continue
        rows[key] = {
            "one_pass": one,
            "two_pass": two,
            "two_pass_qps_vs_one_pass_pct": (
                float(two["qps"]) / float(one["qps"]) - 1.0
            )
            * 100.0,
        }
    return rows


def print_summary(payload: dict[str, Any]) -> None:
    print("\n=== ONE-PASS vs TWO-PASS ===")
    for threshold, row in payload["comparison"].items():
        if row is None:
            print(f"recall>={threshold}: unreachable in one or both indexes")
            continue
        one = row["one_pass"]
        two = row["two_pass"]
        delta = row["two_pass_qps_vs_one_pass_pct"]
        print(
            f"recall>={threshold}: "
            f"one ef={one['ef']} PO={one['prefetch_offset']} "
            f"PL={one['prefetch_lines']} QPS={one['qps']:,.1f}; "
            f"two ef={two['ef']} PO={two['prefetch_offset']} "
            f"PL={two['prefetch_lines']} QPS={two['qps']:,.1f}; "
            f"two-vs-one={delta:+.1f}%"
        )


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.runs <= 0:
        raise ValueError("count and runs must be positive")
    if args.max_queries < 0 or args.warmup_queries < 0:
        raise ValueError("max-queries and warmup-queries cannot be negative")
    if args.max_ef < args.count:
        raise ValueError("max-ef must be >= count")
    for path in (args.one_pass_index, args.two_pass_index):
        if not path.is_dir():
            raise FileNotFoundError(path)

    query_file = args.data_dir / "sift_test.txt"
    ground_truth_file = args.data_dir / "sift_neighbors.txt"
    queries = np.ascontiguousarray(
        np.stack(
            load_query_vectors(
                query_file, first_sep=";", second_sep=" "
            )
        ),
        dtype=np.float32,
    )
    ground_truth = load_ground_truth_ids(
        ground_truth_file, first_sep=";", second_sep=" "
    )
    if args.max_queries:
        queries = queries[: args.max_queries]
        ground_truth = ground_truth[: args.max_queries]
    if len(queries) != len(ground_truth):
        raise ValueError("query and ground-truth counts differ")

    po_values, pl_values, po_cap, pl_cap = prefetch_sweep_for_build(
        index_type="vamana",
        degree=64,
        dimension=128,
        quantize="uniform_uint8",
        po_candidates=args.po_values,
        pl_candidates=args.pl_values,
    )

    settings = {
        "count": args.count,
        "runs": args.runs,
        "warmup_queries": args.warmup_queries,
        "query_count": len(queries),
        "max_ef": args.max_ef,
        "recall_thresholds": args.recall_thresholds,
        "po_values": po_values,
        "pl_values": pl_values,
        "po_cap": po_cap,
        "pl_cap": pl_cap,
        "recall_metric": "mean_top_k_id_intersection",
        "timing": "ann_benchmarks_prepared_query_search_only_best_of_runs",
        "search_path": "ann_bench_search_fast_preallocated_output",
        "query_file": str(query_file.resolve()),
        "ground_truth_file": str(ground_truth_file.resolve()),
        "one_pass_index": str(args.one_pass_index.resolve()),
        "two_pass_index": str(args.two_pass_index.resolve()),
    }
    log(
        f"queries={len(queries)} topk={args.count} runs={args.runs} "
        f"thresholds={args.recall_thresholds}"
    )
    log(
        f"integer ef=[{args.count},{args.max_ef}], "
        f"PO={po_values}, PL={pl_values} (caps {po_cap}/{pl_cap})"
    )
    if args.dry_run:
        return 0

    try:
        zvec.init(
            log_level=LogLevel.WARN,
            query_thread_binding=False,
            optimize_threads=1,
            optimize_thread_binding=False,
        )
    except RuntimeError:
        pass

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "indexes": {},
        "comparison": {},
    }
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as file:
            previous_payload = json.load(file)
        if (
            previous_payload.get("schema_version") != SCHEMA_VERSION
            or previous_payload.get("settings") != settings
        ):
            raise ValueError("checkpoint settings do not match this run")
        payload = previous_payload

    def save_index(label: str, summary: dict[str, Any]) -> None:
        payload["indexes"][label] = summary
        payload["comparison"] = build_comparison(
            payload["indexes"], args.recall_thresholds
        )
        atomic_save(args.output, payload)

    index_specs = (
        ("one_pass", args.one_pass_index, False),
        ("two_pass", args.two_pass_index, True),
    )
    for label, path, expected_two_pass in index_specs:
        log(f"opening {label}: {path}")
        benchmark = ExistingIndexBenchmark(
            label=label,
            path=path,
            expected_two_pass=expected_two_pass,
            queries=queries,
            ground_truth=ground_truth,
            count=args.count,
            runs=args.runs,
            warmup_queries=args.warmup_queries,
        )
        try:
            previous = payload["indexes"].get(label) if args.resume else None
            summary = benchmark_one_index(
                benchmark=benchmark,
                thresholds=args.recall_thresholds,
                max_ef=args.max_ef,
                po_values=po_values,
                pl_values=pl_values,
                previous=previous,
                save_progress=lambda value, label=label: save_index(label, value),
            )
            save_index(label, summary)
        finally:
            benchmark.close()

    payload["comparison"] = build_comparison(
        payload["indexes"], args.recall_thresholds
    )
    atomic_save(args.output, payload)
    print_summary(payload)
    log(f"results saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
