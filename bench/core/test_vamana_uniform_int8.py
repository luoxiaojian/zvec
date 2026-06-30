"""
Reproduce: Vamana index + QuantizeType.UNIFORM_INT8 should work but doesn't.

This script tests that a Vamana index configured with UNIFORM_INT8 quantization
can be created, populated, optimized, and queried — identical to the proven
HNSW + UNIFORM_INT8 code path.
"""
from __future__ import annotations

import tempfile
import shutil
import sys

import numpy as np

sys.path.insert(0, "/root/py/zvec/python")
import zvec
from zvec import (
    Collection,
    CollectionOption,
    CollectionSchema,
    Doc,
    FieldSchema,
    InvertIndexParam,
    VamanaIndexParam,
    VamanaQueryParam,
    Query,
    VectorSchema,
)
from zvec.typing import DataType, IndexType, MetricType, QuantizeType

DIMENSION = 32
NUM_DOCS = 128
TOPK = 5


def build_schema(name: str) -> CollectionSchema:
    """Create a Vamana + UNIFORM_INT8 schema with L2 metric."""
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
                index_param=VamanaIndexParam(
                    metric_type=MetricType.L2,
                    max_degree=32,
                    search_list_size=64,
                    alpha=1.2,
                    quantize_type=QuantizeType.UNIFORM_INT8,
                ),
            ),
        ],
    )


def generate_docs(rng: np.random.Generator, num: int = NUM_DOCS) -> list:
    docs = []
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


def test_create_collection():
    """Test 1: Can we create a collection with Vamana + UNIFORM_INT8?"""
    print("=" * 60)
    print("TEST 1: Create collection with Vamana + UNIFORM_INT8")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp(prefix="zvec_vamana_uint8_")
    schema = build_schema("vamana_uniform_int8")
    option = CollectionOption(read_only=False, enable_mmap=True)

    try:
        coll = zvec.create_and_open(
            path=f"{tmp_dir}/test_create", schema=schema, option=option
        )
        print(f"  [PASS] Collection created successfully")
        print(f"  Schema: {coll.schema.vectors[0].index_param.to_dict()}")
        coll.destroy()
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def test_insert_and_query_before_optimize():
    """Test 2: Insert docs and query before optimize (raw memory path)."""
    print("\n" + "=" * 60)
    print("TEST 2: Insert and query before optimize")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    tmp_dir = tempfile.mkdtemp(prefix="zvec_vamana_uint8_")
    schema = build_schema("vamana_uniform_int8_writer")
    option = CollectionOption(read_only=False, enable_mmap=True)

    try:
        coll = zvec.create_and_open(
            path=f"{tmp_dir}/test_writer", schema=schema, option=option
        )
        docs = generate_docs(rng)
        for r in coll.insert(docs=docs):
            if not r.ok():
                print(f"  [FAIL] Insert failed: code={r.code()}")
                coll.destroy()
                return False

        print(f"  Inserted {coll.stats.doc_count} docs")

        # Query with the first doc's vector
        query_vec = docs[0].vector("dense")
        hits = coll.query(
            Query(
                field_name="dense",
                vector=query_vec,
                param=VamanaQueryParam(ef_search=128),
            ),
            topk=TOPK,
        )
        if hits and len(hits) >= 1:
            print(f"  [PASS] Query returned {len(hits)} hits, top-1 id={hits[0].id}")
            if hits[0].id == "0":
                print(f"  [PASS] Self-recall correct")
            else:
                print(f"  [WARN] Expected top-1 id=0, got {hits[0].id}")
        else:
            print(f"  [FAIL] Query returned no hits")
            coll.destroy()
            return False

        coll.destroy()
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def test_optimize_and_query():
    """Test 3: Optimize (build quantized index) and then query."""
    print("\n" + "=" * 60)
    print("TEST 3: Optimize and query (quantized Vamana index)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    tmp_dir = tempfile.mkdtemp(prefix="zvec_vamana_uint8_")
    schema = build_schema("vamana_uniform_int8_optimize")
    option = CollectionOption(read_only=False, enable_mmap=True)

    try:
        coll = zvec.create_and_open(
            path=f"{tmp_dir}/test_optimize", schema=schema, option=option
        )
        docs = generate_docs(rng)
        for r in coll.insert(docs=docs):
            if not r.ok():
                print(f"  [FAIL] Insert failed: code={r.code()}")
                coll.destroy()
                return False

        print(f"  Inserted {coll.stats.doc_count} docs")

        # Optimize - this should train the UNIFORM_INT8 converter and build
        # the quantized Vamana index
        print(f"  Calling optimize()...")
        coll.optimize()
        print(f"  [PASS] optimize() completed")

        # Query after optimize
        query_vec = docs[0].vector("dense")
        hits = coll.query(
            Query(
                field_name="dense",
                vector=query_vec,
                param=VamanaQueryParam(ef_search=128),
            ),
            topk=TOPK,
        )
        if hits and len(hits) >= 1:
            print(f"  [PASS] Query returned {len(hits)} hits, top-1 id={hits[0].id}")
            if hits[0].id == "0":
                print(f"  [PASS] Self-recall correct after optimize")
            else:
                print(f"  [WARN] Expected top-1 id=0, got {hits[0].id}")
        else:
            print(f"  [FAIL] Query returned no hits after optimize")
            coll.destroy()
            return False

        coll.destroy()
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def test_optimize_reopen_and_query():
    """Test 4: Optimize, close, reopen and query (persistence path)."""
    print("\n" + "=" * 60)
    print("TEST 4: Optimize, reopen and query (persistence)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    tmp_dir = tempfile.mkdtemp(prefix="zvec_vamana_uint8_")
    path_str = f"{tmp_dir}/test_reopen"
    schema = build_schema("vamana_uniform_int8_reopen")
    option = CollectionOption(read_only=False, enable_mmap=True)

    try:
        coll = zvec.create_and_open(
            path=path_str, schema=schema, option=option
        )
        docs = generate_docs(rng)
        for r in coll.insert(docs=docs):
            if not r.ok():
                print(f"  [FAIL] Insert failed: code={r.code()}")
                coll.destroy()
                return False

        print(f"  Inserted {coll.stats.doc_count} docs")
        coll.optimize()
        print(f"  optimize() done")
        del coll

        # Reopen the collection
        print(f"  Reopening collection...")
        reopened = zvec.open(path=path_str, option=option)
        print(f"  [PASS] Reopened, doc_count={reopened.stats.doc_count}")

        query_vec = docs[0].vector("dense")
        hits = reopened.query(
            Query(
                field_name="dense",
                vector=query_vec,
                param=VamanaQueryParam(ef_search=128),
            ),
            topk=TOPK,
        )
        if hits and len(hits) >= 1:
            print(f"  [PASS] Query returned {len(hits)} hits, top-1 id={hits[0].id}")
        else:
            print(f"  [FAIL] Query returned no hits after reopen")
            reopened.destroy()
            return False

        reopened.destroy()
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def test_multi_segment_optimize():
    """Test 5: Multi-segment optimize with flush in between."""
    print("\n" + "=" * 60)
    print("TEST 5: Multi-segment optimize")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    tmp_dir = tempfile.mkdtemp(prefix="zvec_vamana_uint8_")
    schema = build_schema("vamana_uniform_int8_multiseg")
    option = CollectionOption(read_only=False, enable_mmap=True)

    try:
        coll = zvec.create_and_open(
            path=f"{tmp_dir}/test_multiseg", schema=schema, option=option
        )
        docs = generate_docs(rng, num=256)

        # Insert in two batches with flush
        for batch in (docs[:128], docs[128:]):
            for r in coll.insert(docs=batch):
                if not r.ok():
                    print(f"  [FAIL] Insert failed: code={r.code()}")
                    coll.destroy()
                    return False
            coll.flush()

        print(f"  Inserted {coll.stats.doc_count} docs in 2 segments")

        coll.optimize()
        print(f"  optimize() done")

        # Measure recall
        total = 0.0
        num_queries = 10
        for i in range(num_queries):
            query_vec = docs[i].vector("dense")
            hits = coll.query(
                Query(
                    field_name="dense",
                    vector=query_vec,
                    param=VamanaQueryParam(ef_search=200),
                ),
                topk=TOPK,
            )
            if hits:
                got_ids = {h.id for h in hits}
                # compute brute-force exact top-k
                mat = np.asarray([d.vector("dense") for d in docs], dtype=np.float32)
                q = np.asarray(query_vec, dtype=np.float32)
                dists = np.sum((mat - q) ** 2, axis=1)
                order = np.argsort(dists, kind="stable")[:TOPK]
                want_ids = {str(idx) for idx in order}
                total += len(got_ids & want_ids) / float(TOPK)

        mean_recall = total / num_queries
        print(f"  Mean recall@{TOPK} = {mean_recall:.3f}")
        if mean_recall >= 0.5:
            print(f"  [PASS] Recall acceptable")
        else:
            print(f"  [FAIL] Recall too low")
            coll.destroy()
            return False

        coll.destroy()
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def main():
    print("Reproducing: Vamana + UNIFORM_INT8 quantization issue")
    print("=" * 60)

    results = []
    results.append(("Create Collection", test_create_collection()))
    results.append(("Insert & Query (writer)", test_insert_and_query_before_optimize()))
    results.append(("Optimize & Query", test_optimize_and_query()))
    results.append(("Optimize, Reopen & Query", test_optimize_reopen_and_query()))
    results.append(("Multi-segment Optimize", test_multi_segment_optimize()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    if not all(r[1] for r in results):
        print("\nSome tests FAILED - issue reproduced!")
        sys.exit(1)
    else:
        print("\nAll tests PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
