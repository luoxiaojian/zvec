#!/usr/bin/env python3
"""Build and time a single-threaded Vamana index on SIFT."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import struct
import time
from pathlib import Path

# Keep numerical runtimes single-threaded as well as zvec's optimize pool.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    LogLevel,
    OptimizeOption,
    VamanaIndexParam,
    VectorSchema,
    create_and_open,
    open as zvec_open,
)
from zvec.typing import DataType, MetricType, QuantizeType


VECTOR_FIELD = "vector"
SIFT_DIMENSION = 128
VECS_HEADER_SIZE = 104
VECS_HEADER_PREFIX = struct.Struct("<QHHI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-vecs",
        type=Path,
        default=Path("/root/zvec_workspace/output/sift_train.vecs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Collection path; defaults to a pass-specific ann-benchmarks name.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=0,
        help="Use only a prefix for smoke tests; 0 builds the full SIFT set.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-info",
        action="store_true",
        help="Enable INFO logs for the initial and second Vamana passes.",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Disable full-graph two-pass refinement.",
    )
    parser.add_argument(
        "--no-contiguous-memory",
        action="store_true",
        help="Disable the contiguous graph arena (enabled by default).",
    )
    return parser.parse_args()


def load_sift_vectors(path: Path, max_vectors: int) -> np.ndarray:
    """Load X_train before timing, matching ann-benchmarks' runner boundary."""
    with path.open("rb") as stream:
        prefix = stream.read(VECS_HEADER_PREFIX.size)
        if len(prefix) != VECS_HEADER_PREFIX.size:
            raise ValueError(f"truncated vecs header: {path}")
        total, _meta_size_v1, _version, meta_size = VECS_HEADER_PREFIX.unpack(
            prefix
        )

    count = int(total)
    if max_vectors > 0:
        count = min(count, max_vectors)
    vector_offset = VECS_HEADER_SIZE + int(meta_size)
    mapped = np.memmap(
        path,
        mode="r",
        dtype=np.float32,
        offset=vector_offset,
        shape=(count, SIFT_DIMENSION),
    )
    # ann-benchmarks loads the HDF5 train split into memory before build_index()
    # starts its timer. Force the same owned, contiguous FP32 representation.
    return np.array(mapped, dtype=np.float32, order="C", copy=True)


def build_schema(
    use_contiguous_memory: bool, two_pass_build: bool
) -> CollectionSchema:
    index_param = VamanaIndexParam(
        metric_type=MetricType.L2,
        max_degree=64,
        search_list_size=500,
        alpha=1.5,
        saturate_graph=False,
        use_contiguous_memory=use_contiguous_memory,
        two_pass_build=two_pass_build,
        quantize_type=QuantizeType.UNIFORM_UINT8,
    )
    params = index_param.to_dict()
    if bool(params.get("two_pass_build")) != two_pass_build:
        raise RuntimeError(
            "installed zvec wheel did not preserve two_pass_build="
            f"{two_pass_build}"
        )
    return CollectionSchema(
        name="annb_ann_bench_doc_ids",
        fields=[],
        vectors=[
            VectorSchema(
                VECTOR_FIELD,
                DataType.VECTOR_FP32,
                dimension=SIFT_DIMENSION,
                index_param=index_param,
            )
        ],
    )


def main() -> int:
    args = parse_args()
    two_pass_build = not args.single_pass
    if args.output is None:
        pass_tag = "2pass" if two_pass_build else "1pass"
        args.output = Path(
            "/root/py/zvec/zvec_indices/"
            "ann_bench_doc_ids_vamana_euclidean_d128_"
            f"R64_L500_a1.5_{pass_tag}_cm1_fcm0_uniform_uint8"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.train_vecs.is_file():
        raise FileNotFoundError(args.train_vecs)
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"{args.output} already exists; pass --force to rebuild"
        )

    try:
        zvec.init(
            log_level=LogLevel.INFO if args.log_info else LogLevel.WARN,
            query_thread_binding=False,
            optimize_threads=1,
            optimize_thread_binding=False,
        )
    except RuntimeError:
        pass

    load_start = time.perf_counter()
    vectors = load_sift_vectors(args.train_vecs, args.max_vectors)
    dataset_load_seconds = time.perf_counter() - load_start
    count = len(vectors)

    print(
        json.dumps(
            {
                "dataset": "SIFT",
                "vectors": count,
                "vector_dimension": SIFT_DIMENSION,
                "max_degree": 64,
                "search_list_size": 500,
                "first_pass_alpha": 1.0 if two_pass_build else None,
                "target_alpha": 1.5,
                "two_pass_build": two_pass_build,
                "quantize": "UNIFORM_UINT8",
                "optimize_threads": 1,
                "contiguous_memory": not args.no_contiguous_memory,
                "output": str(args.output),
                "zvec_module": str(Path(zvec.__file__).resolve()),
                "dataset_load_seconds_excluded": dataset_load_seconds,
            },
            indent=2,
        ),
        flush=True,
    )

    # This is the same timing boundary as ann_benchmarks.runner.build_index():
    # X_train is already loaded, then the entire adapter fit() is timed.
    ann_build_start = time.time()

    # ZvecBase.fit() always rebuilds its path and creates the parent directory.
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    schema = build_schema(
        not args.no_contiguous_memory, two_pass_build
    )
    build_collection = create_and_open(
        path=str(args.output),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )

    insert_start = time.perf_counter()
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        docs = [
            Doc(id=str(i), vectors={VECTOR_FIELD: vectors[i].tolist()})
            for i in range(start, end)
        ]
        results = build_collection.insert(docs)
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if not result.ok():
                raise RuntimeError(
                    f"[zvec] insert failed: {result.code()}"
                )
    insert_seconds = time.perf_counter() - insert_start

    optimize_start = time.perf_counter()
    # Match module.py exactly: thread count comes from zvec.init(), while the
    # adapter passes a default OptimizeOption.
    build_collection.optimize(option=OptimizeOption())
    optimize_seconds = time.perf_counter() - optimize_start

    # The adapter releases the writer and reopens mmap/read-only inside fit().
    reopen_start = time.perf_counter()
    build_collection = None
    gc.collect()
    collection = zvec_open(
        str(args.output),
        CollectionOption(read_only=True, enable_mmap=True),
    )
    # ZvecAnnBenchDocIds.fit() additionally caches the native indexers.
    collection._obj.ann_bench_prepare(VECTOR_FIELD)
    reopen_seconds = time.perf_counter() - reopen_start

    ann_bench_build_seconds = time.time() - ann_build_start
    result = {
        "vectors": count,
        "two_pass_build": two_pass_build,
        "dataset_load_seconds_excluded": dataset_load_seconds,
        "insert_seconds": insert_seconds,
        "optimize_seconds": optimize_seconds,
        "reopen_and_ann_bench_prepare_seconds": reopen_seconds,
        "ann_bench_build_seconds": ann_bench_build_seconds,
    }
    collection = None
    gc.collect()
    result_path = args.output.parent / f"{args.output.name}_timing.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"timing_result={result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
