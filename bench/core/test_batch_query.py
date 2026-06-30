#!/usr/bin/env python3
"""Verify batch_fast_query_doc_ids_only correctness and benchmark vs per-query."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common.config import load_search_config
from common.io import load_query_vectors, load_ground_truth_ids
from common.schema import VECTOR_FIELD, make_query_param
import zvec

WORKSPACE_ROOT = Path("/root/main/workspace")
COLLECTION_ROOT = Path("/root/py/workspace")
DATA_DIR = Path("/root/zvec_workspace/output")
RUNS = 5
COUNT = 10


def main():
    # Use SIFT hnsw_i8_m48 (best config from prior tests)
    dataset = "sift"
    config_name = "hnsw_i8_m48"
    config_dir = WORKSPACE_ROOT / dataset / config_name
    collection_path = COLLECTION_ROOT / dataset / config_name / "collection"

    cfg = load_search_config(
        dataset,
        workspace_config_dir=config_dir,
        data_dir=DATA_DIR,
        collection_path=collection_path,
    )
    queries = load_query_vectors(
        cfg.query_file, first_sep=cfg.query_first_sep, second_sep=cfg.query_second_sep
    )
    X_test = np.ascontiguousarray(np.stack(queries), dtype=np.float32)
    gt = load_ground_truth_ids(
        cfg.ground_truth_file,
        first_sep=cfg.ground_truth_first_sep,
        second_sep=cfg.ground_truth_second_sep,
    )
    param = make_query_param(cfg)

    col = zvec.open(
        str(collection_path),
        zvec.CollectionOption(read_only=True, enable_mmap=True),
    )

    nq = len(X_test)
    print(f"=== Batch Query Verification: {dataset}/{config_name} ===")
    print(f"Queries: {nq}, Dim: {X_test.shape[1]}, TopK: {COUNT}, Runs: {RUNS}")
    print()

    # --- Warmup ---
    print("Warming up...", flush=True)
    for v in X_test[: min(2000, nq)]:
        col.fast_query_doc_ids_only(VECTOR_FIELD, v, topk=COUNT, param=param)

    # ================================================================
    # 1. CORRECTNESS: compare batch vs per-query results
    # ================================================================
    print("\n--- Correctness Check ---")

    # Per-query results
    single_ids = []
    for v in X_test:
        ids = col.fast_query_doc_ids_only(VECTOR_FIELD, v, topk=COUNT, param=param)
        single_ids.append(np.asarray(ids, dtype=np.int64))
    single_ids = np.stack(single_ids)  # (nq, topk)

    # Batch results
    batch_ids = col.batch_fast_query_doc_ids_only(
        VECTOR_FIELD, X_test, topk=COUNT, param=param
    )

    assert batch_ids.shape == (nq, COUNT), (
        f"Shape mismatch: {batch_ids.shape} vs expected ({nq}, {COUNT})"
    )
    assert np.array_equal(single_ids, batch_ids), "MISMATCH: batch != single!"
    print(f"  batch_fast_query_doc_ids_only: PASS (shape={batch_ids.shape})")

    # Also check batch_fast_query_doc_ids
    single_ids2 = []
    single_scores = []
    for v in X_test:
        ids, scores = col.fast_query_doc_ids(VECTOR_FIELD, v, topk=COUNT, param=param)
        single_ids2.append(np.asarray(ids, dtype=np.int64))
        single_scores.append(np.asarray(scores, dtype=np.float32))
    single_ids2 = np.stack(single_ids2)
    single_scores = np.stack(single_scores)

    batch_ids2, batch_scores = col.batch_fast_query_doc_ids(
        VECTOR_FIELD, X_test, topk=COUNT, param=param
    )
    assert np.array_equal(single_ids2, batch_ids2), "MISMATCH: batch_doc_ids ids!"
    assert np.allclose(single_scores, batch_scores, atol=1e-5), (
        "MISMATCH: batch_doc_ids scores!"
    )
    print(f"  batch_fast_query_doc_ids:      PASS (ids+scores match)")

    # ================================================================
    # 2. RECALL (sanity check)
    # ================================================================
    print("\n--- Recall Check ---")
    total_hit = 0.0
    for i in range(nq):
        gt_set = set(int(x) for x in gt[i][:COUNT])
        knn = set(int(x) for x in batch_ids[i][:COUNT])
        total_hit += len(knn & gt_set) / COUNT
    recall = total_hit / nq
    print(f"  Recall@{COUNT}: {recall:.4f}")

    # ================================================================
    # 3. PERFORMANCE: batch vs per-query
    # ================================================================
    print("\n--- Performance Comparison ---")

    # Per-query: fast_query_doc_ids_only
    best_single = float("inf")
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for v in X_test:
            col.fast_query_doc_ids_only(VECTOR_FIELD, v, topk=COUNT, param=param)
        elapsed = (time.perf_counter() - t0) / nq
        best_single = min(best_single, elapsed)

    # Batch: batch_fast_query_doc_ids_only
    best_batch = float("inf")
    for _ in range(RUNS):
        t0 = time.perf_counter()
        col.batch_fast_query_doc_ids_only(VECTOR_FIELD, X_test, topk=COUNT, param=param)
        elapsed = (time.perf_counter() - t0) / nq
        best_batch = min(best_batch, elapsed)

    # Per-query: fast_query_doc_ids (with scores)
    best_single_scores = float("inf")
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for v in X_test:
            col.fast_query_doc_ids(VECTOR_FIELD, v, topk=COUNT, param=param)
        elapsed = (time.perf_counter() - t0) / nq
        best_single_scores = min(best_single_scores, elapsed)

    # Batch: batch_fast_query_doc_ids (with scores)
    best_batch_scores = float("inf")
    for _ in range(RUNS):
        t0 = time.perf_counter()
        col.batch_fast_query_doc_ids(VECTOR_FIELD, X_test, topk=COUNT, param=param)
        elapsed = (time.perf_counter() - t0) / nq
        best_batch_scores = min(best_batch_scores, elapsed)

    qps_single = 1.0 / best_single
    qps_batch = 1.0 / best_batch
    qps_single_s = 1.0 / best_single_scores
    qps_batch_s = 1.0 / best_batch_scores

    print(f"\n  {'Mode':<35s} {'QPS':>10s} {'mean(us)':>10s} {'speedup':>8s}")
    print(f"  {'-'*70}")
    print(f"  {'per-query doc_ids_only':<35s} {qps_single:>10,.0f} {best_single*1e6:>10.1f}")
    print(
        f"  {'BATCH doc_ids_only':<35s} {qps_batch:>10,.0f} {best_batch*1e6:>10.1f}"
        f" {qps_batch/qps_single:>7.2f}x"
    )
    print(f"  {'per-query doc_ids (w/ scores)':<35s} {qps_single_s:>10,.0f} {best_single_scores*1e6:>10.1f}")
    print(
        f"  {'BATCH doc_ids (w/ scores)':<35s} {qps_batch_s:>10,.0f} {best_batch_scores*1e6:>10.1f}"
        f" {qps_batch_s/qps_single_s:>7.2f}x"
    )

    print(f"\n  Recall@{COUNT} = {recall:.4f}")
    print(f"\n  benchmark_comparison.md ref (fast_query_doc_ids): 26,058 QPS")
    print(f"  Previous bench/core best (fast_query_doc_ids):    33,068 QPS")
    print(f"  NEW batch_fast_query_doc_ids_only:                {qps_batch:,.0f} QPS")


if __name__ == "__main__":
    main()
