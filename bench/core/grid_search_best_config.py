#!/usr/bin/env python3
"""从索引×量化构建网格中选出各 recall 阈值下的最佳 QPS 配置。

流程（仅 ``ann_bench_doc_ids`` 路径）：

1. **构建网格**（参数叉乘）  
   - Vamana: max_degree × alpha × search_list_size × quantize  
   - HNSW: m × efConstruction × quantize  
   - sift: quantize ∈ {int8, uniform_int8, fp32}  
   - gist: quantize ∈ {int8, fp32}（不含 uniform_int8，与 workspace 一致）

2. **搜索参数**（每个已构建索引）  
   - 阶段 A：固定 ``prefetch_offset=0, prefetch_lines=0``，在有序 ef 候选列表上
     **二分搜索**满足 recall ≥ T 的最小 ef（假设 recall 随 ef 单调不降）。  
   - 阶段 B：在该最小 ef 上枚举 ``prefetch_offset × prefetch_lines``（PO 受节点
     出度限制，PL 受 ``ceil(向量存储字节数/64)`` 限制），取 QPS 最高且
     recall ≥ T 的组合。

3. **汇总**：每个 (dataset, recall_threshold) 输出全局最优的索引构建参数 +
   搜索参数。

用法示例::

    # 预览网格规模
    python grid_search_best_config.py --dry-run

    # 阶段 1：并行构建（独立 Python 进程，避免 fork+zvec 死锁）
    bash run_parallel_build.sh --datasets sift --jobs 4

    # 或 Python 内置 subprocess 调度（等价）
    python grid_search_best_config.py --build-only --build-jobs 4 --datasets sift

    # 阶段 2：串行压测（每个 setting 跑 3 次取最佳 QPS）
    python grid_search_best_config.py --search-only --datasets sift gist --runs 3
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    HnswIndexParam,
    HnswQueryParam,
    LogLevel,
    OptimizeOption,
    VamanaIndexParam,
    VamanaQueryParam,
    VectorSchema,
    create_and_open,
    open as zvec_open,
)
from zvec.typing import DataType, MetricType, QuantizeType

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common.adapters import ZvecAnnBenchDocIdsANN
from common.io import iter_vecs_batches, load_ground_truth_ids, load_query_vectors
from common.prefetch_bounds import effective_element_size, prefetch_sweep_for_build
from common.schema import VECTOR_FIELD

# 不在模块 import 时 init zvec：ProcessPool/fork 会导致 optimize 死锁。
# 各入口（build-one 子进程、串行 build、search）自行调用 init_zvec_safe()。

# ---------------------------------------------------------------------------
# 默认网格（可通过 CLI 覆盖 ef / po / pl 列表）
# ---------------------------------------------------------------------------

RECALL_THRESHOLDS = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]

VAMANA_DEGREES = [16, 24, 32, 48, 64]
VAMANA_ALPHAS = [1.1, 1.3, 1.5]
VAMANA_LIST_SIZES = [100, 300, 500]

HNSW_M_VALUES = [16, 24, 32, 48, 64]
HNSW_EF_CONSTRUCTIONS = [300, 500]

QUANTIZE_CHOICES_SIFT = ["int8", "uniform_int8", "fp32"]
QUANTIZE_CHOICES_GIST = ["int8", "fp32"]
# 默认（未区分 dataset 时的 fallback）
QUANTIZE_CHOICES = QUANTIZE_CHOICES_SIFT


def quantize_choices_for_dataset(dataset: str) -> list[str]:
    if dataset == "sift":
        return QUANTIZE_CHOICES_SIFT
    if dataset == "gist":
        return QUANTIZE_CHOICES_GIST
    raise ValueError(f"unsupported dataset: {dataset}")

# ef 候选上界（二分搜索在此有序列表上找最小 ef）
HNSW_EF_SWEEP = [
    8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 112, 128,
    160, 192, 256, 384, 512, 768, 1024,
]
VAMANA_EF_SWEEP = [
    8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 112, 128,
    160, 192, 256, 384, 512, 768, 1024,
]

PREFETCH_OFFSET_SWEEP = [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64, 96, 128]
PREFETCH_LINES_SWEEP = [0, 1, 2, 4, 6, 8]

DATASET_INFO = {
    "sift": {
        "dimension": 128,
        "train_vecs": "sift_train.vecs",
        "query_file": "sift_test.txt",
        "gt_file": "sift_neighbors.txt",
    },
    "gist": {
        "dimension": 960,
        "train_vecs": "gist_train.vecs",
        "query_file": "gist_test.txt",
        "gt_file": "gist_neighbors.txt",
    },
}

_QUANTIZE_TYPE = {
    "int8": QuantizeType.INT8,
    "uniform_int8": QuantizeType.UNIFORM_INT8,
    "fp32": QuantizeType.UNDEFINED,
}

DEFAULT_RUNS = 3
DEFAULT_OPTIMIZE_THREADS = 8

# zvec collection 名只允许 [a-zA-Z0-9_-]（见 db/common/constants.h）
# alpha=1.1 不能写成 a1.1，改为 a11 / a13 / a15
_ALPHA_SLUG = {1.1: "a11", 1.3: "a13", 1.5: "a15"}


def alpha_slug(alpha: float) -> str:
    if alpha in _ALPHA_SLUG:
        return _ALPHA_SLUG[alpha]
    # fallback: 1.25 -> a125
    return "a" + str(alpha).replace(".", "")


def init_zvec_safe(*, optimize_threads: int = DEFAULT_OPTIMIZE_THREADS) -> None:
    """仅在当前进程内 init zvec（勿在 fork 父进程中调用后再 fork）。"""
    try:
        zvec.init(
            log_level=LogLevel.WARN,
            optimize_threads=max(1, int(optimize_threads)),
        )
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# 日志 / 进度（nohup 友好：带时间戳、flush、progress.json）
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class ProgressTracker:
    """写入 progress.json；多进程构建时用文件锁合并进度。"""

    def __init__(
        self,
        path: Path,
        phase: str,
        total: int,
        *,
        reset: bool = True,
    ):
        self.path = path
        self.phase = phase
        self.total = total
        if reset or not path.exists():
            self.done = 0
            self.failed = 0
            self.skipped = 0
            self.started = time.time()
            self.recent: list[dict[str, Any]] = []
            self._write(current="starting")
        else:
            data = self._read_locked()
            self.done = int(data.get("done", 0))
            self.failed = int(data.get("failed", 0))
            self.skipped = int(data.get("skipped", 0))
            self.started = time.time() - float(data.get("elapsed_s", 0))
            self.recent = list(data.get("recent", []))

    def _read_locked(self) -> dict[str, Any]:
        return ProgressTracker._read_file(self.path)

    @staticmethod
    def _read_file(path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return {}
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _payload(self, *, current: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = time.time() - self.started
        processed = self.done + self.failed + self.skipped
        rate = processed / elapsed if elapsed > 0 and processed > 0 else 0.0
        remaining = max(0, self.total - processed)
        eta = remaining / rate if rate > 0 else None
        payload: dict[str, Any] = {
            "phase": self.phase,
            "total": self.total,
            "done": self.done,
            "failed": self.failed,
            "skipped": self.skipped,
            "current": current,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1) if eta is not None else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recent": self.recent[-20:],
        }
        if extra:
            payload.update(extra)
        return payload

    def _write(self, *, current: str, extra: dict[str, Any] | None = None) -> None:
        payload = self._payload(current=current, extra=extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    @classmethod
    def record_event(
        cls,
        path: Path,
        *,
        phase: str,
        total: int,
        current: str,
        status: str,
        detail: str = "",
    ) -> None:
        """子进程构建完成后追加一条进度（文件锁）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read()
                if raw.strip():
                    data = json.loads(raw)
                else:
                    data = {
                        "phase": phase,
                        "total": total,
                        "done": 0,
                        "failed": 0,
                        "skipped": 0,
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "recent": [],
                    }
                if status == "ok":
                    data["done"] = int(data.get("done", 0)) + 1
                elif status == "skip":
                    data["skipped"] = int(data.get("skipped", 0)) + 1
                elif status == "fail":
                    data["failed"] = int(data.get("failed", 0)) + 1
                started_at = data.get("started_at")
                if started_at:
                    try:
                        started_ts = time.mktime(
                            time.strptime(started_at, "%Y-%m-%d %H:%M:%S")
                        )
                    except ValueError:
                        started_ts = time.time()
                else:
                    started_ts = time.time()
                    data["started_at"] = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(started_ts)
                    )
                elapsed = time.time() - started_ts
                processed = (
                    int(data.get("done", 0))
                    + int(data.get("failed", 0))
                    + int(data.get("skipped", 0))
                )
                rate = processed / elapsed if elapsed > 0 and processed > 0 else 0.0
                remaining = max(0, int(data.get("total", total)) - processed)
                data.update(
                    {
                        "phase": phase,
                        "total": int(data.get("total", total)),
                        "current": current,
                        "elapsed_s": round(elapsed, 1),
                        "eta_s": round(remaining / rate, 1) if rate > 0 else None,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                recent = list(data.get("recent", []))
                recent.append(
                    {
                        "time": time.strftime("%H:%M:%S"),
                        "current": current,
                        "status": status,
                        "detail": detail,
                    }
                )
                data["recent"] = recent[-20:]
                fh.seek(0)
                fh.truncate()
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def event(
        self,
        *,
        current: str,
        status: str,
        detail: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        if status == "ok":
            self.done += 1
        elif status == "skip":
            self.skipped += 1
        elif status == "fail":
            self.failed += 1
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "current": current,
            "status": status,
            "detail": detail,
        }
        self.recent.append(entry)
        processed = self.done + self.failed + self.skipped
        eta_str = ""
        elapsed = time.time() - self.started
        if processed > 0 and elapsed > 0:
            eta = (self.total - processed) * elapsed / processed
            eta_str = f" ETA~{eta / 60:.1f}min"
        log(
            f"{self.phase.upper()} [{processed}/{self.total}] "
            f"{current} — {status}{(': ' + detail) if detail else ''}{eta_str}"
        )
        self._write(current=current, extra=extra)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexBuildSpec:
    """单个索引构建配置。"""

    dataset: str
    index_type: str  # "hnsw" | "vamana"
    quantize: str
    m: int  # HNSW m 或 Vamana max_degree
    ef_construction: int | None = None
    search_list_size: int | None = None
    alpha: float | None = None

    def config_id(self) -> str:
        q = self.quantize
        if self.index_type == "hnsw":
            return f"hnsw_m{self.m}_efc{self.ef_construction}_{q}"
        return (
            f"vamana_d{self.m}_L{self.search_list_size}"
            f"_{alpha_slug(float(self.alpha))}_{q}"
        )

    def collection_dir(self, output_root: Path) -> Path:
        return output_root / self.dataset / self.config_id()


@dataclass
class SearchResult:
    ef: int
    prefetch_offset: int
    prefetch_lines: int
    recall: float
    qps: float
    mean_us: float


@dataclass
class BuildSearchSummary:
    build: IndexBuildSpec
    ef_curve: list[SearchResult] = field(default_factory=list)
    per_recall: dict[str, SearchResult | None] = field(default_factory=dict)


@dataclass
class GlobalBest:
    dataset: str
    recall_threshold: float
    build: IndexBuildSpec
    search: SearchResult


# ---------------------------------------------------------------------------
# 网格生成
# ---------------------------------------------------------------------------


def iter_build_specs(
    datasets: list[str],
    index_types: list[str],
) -> Iterator[IndexBuildSpec]:
    for dataset in datasets:
        quants = quantize_choices_for_dataset(dataset)
        for index_type in index_types:
            if index_type == "vamana":
                for m, alpha, lsize, quant in itertools.product(
                    VAMANA_DEGREES,
                    VAMANA_ALPHAS,
                    VAMANA_LIST_SIZES,
                    quants,
                ):
                    yield IndexBuildSpec(
                        dataset=dataset,
                        index_type="vamana",
                        quantize=quant,
                        m=m,
                        search_list_size=lsize,
                        alpha=alpha,
                    )
            elif index_type == "hnsw":
                for m, efc, quant in itertools.product(
                    HNSW_M_VALUES,
                    HNSW_EF_CONSTRUCTIONS,
                    quants,
                ):
                    yield IndexBuildSpec(
                        dataset=dataset,
                        index_type="hnsw",
                        quantize=quant,
                        m=m,
                        ef_construction=efc,
                    )
            else:
                raise ValueError(f"unknown index_type: {index_type}")


def find_spec_by_config_id(
    config_id: str,
    *,
    datasets: list[str],
    index_types: list[str],
    dataset: str | None = None,
) -> IndexBuildSpec:
    search_datasets = [dataset] if dataset is not None else datasets
    for spec in iter_build_specs(search_datasets, index_types):
        if spec.config_id() == config_id:
            return spec
    scope = dataset or ",".join(datasets)
    raise ValueError(f"config-id not found in grid: {config_id} (datasets={scope})")


def specs_needing_build(
    specs: list[IndexBuildSpec],
    output_root: Path,
    force: bool,
) -> list[IndexBuildSpec]:
    if force:
        return list(specs)
    return [
        s for s in specs
        if not (s.collection_dir(output_root) / "collection").exists()
    ]


def make_collection_schema(spec: IndexBuildSpec, dimension: int) -> CollectionSchema:
    quantize = _QUANTIZE_TYPE[spec.quantize]
    if spec.index_type == "vamana":
        index_param = VamanaIndexParam(
            metric_type=MetricType.L2,
            max_degree=spec.m,
            search_list_size=int(spec.search_list_size),
            alpha=float(spec.alpha),
            use_contiguous_memory=True,
            quantize_type=quantize,
        )
    else:
        index_param = HnswIndexParam(
            metric_type=MetricType.L2,
            m=spec.m,
            ef_construction=int(spec.ef_construction),
            use_contiguous_memory=True,
            quantize_type=quantize,
        )
    return CollectionSchema(
        name=f"{spec.dataset}_grid_{spec.config_id()}",
        fields=[],
        vectors=[
            VectorSchema(
                VECTOR_FIELD,
                DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=index_param,
            )
        ],
    )


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------


def build_index(
    spec: IndexBuildSpec,
    *,
    data_dir: Path,
    output_root: Path,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    """构建单个 collection，返回状态 dict（供并行 worker 与进度汇总）。"""
    info = DATASET_INFO[spec.dataset]
    collection_path = spec.collection_dir(output_root) / "collection"
    config_id = spec.config_id()
    if collection_path.exists() and not force:
        return {
            "config_id": config_id,
            "dataset": spec.dataset,
            "path": str(collection_path),
            "skipped": True,
            "vectors": 0,
            "elapsed_s": 0.0,
        }

    if collection_path.exists() and force:
        shutil.rmtree(collection_path)

    collection_path.parent.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / info["train_vecs"]
    if not train_file.exists():
        raise FileNotFoundError(f"train file not found: {train_file}")

    schema = make_collection_schema(spec, info["dimension"])
    col = create_and_open(
        path=str(collection_path),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    total = 0
    t0 = time.perf_counter()
    try:
        for keys, vectors in iter_vecs_batches(
            train_file, info["dimension"], batch_size
        ):
            docs = [
                Doc(id=str(int(k)), vectors={VECTOR_FIELD: vec})
                for k, vec in zip(keys, vectors.tolist(), strict=True)
            ]
            results = col.insert(docs)
            if not isinstance(results, list):
                results = [results]
            for r in results:
                if not r.ok():
                    raise RuntimeError(f"insert failed: {r.code()}")
            total += len(keys)
        col.optimize(option=OptimizeOption())
    finally:
        col = None

    elapsed = time.perf_counter() - t0
    return {
        "config_id": config_id,
        "dataset": spec.dataset,
        "path": str(collection_path),
        "skipped": False,
        "vectors": total,
        "elapsed_s": round(elapsed, 1),
    }


def run_build_one(
    spec: IndexBuildSpec,
    *,
    data_dir: Path,
    output_root: Path,
    batch_size: int,
    force: bool,
    optimize_threads: int,
    progress_path: Path | None,
    progress_total: int,
) -> int:
    """构建单个配置（独立进程入口）。"""
    init_zvec_safe(optimize_threads=optimize_threads)
    cid = spec.config_id()
    label = f"{spec.dataset}/{cid}"
    try:
        result = build_index(
            spec,
            data_dir=data_dir,
            output_root=output_root,
            batch_size=batch_size,
            force=force,
        )
        if result.get("skipped"):
            detail = "exists"
            status = "skip"
            log(f"BUILD-ONE {label} — skip ({detail})")
        else:
            detail = f"{result['vectors']} vecs in {result['elapsed_s']}s"
            status = "ok"
            log(f"BUILD-ONE {label} — ok ({detail})")
        if progress_path is not None:
            ProgressTracker.record_event(
                progress_path,
                phase="build",
                total=progress_total,
                current=cid,
                status=status,
                detail=detail,
            )
        return 0
    except Exception as exc:
        log(f"BUILD-ONE {label} — fail: {exc}")
        if progress_path is not None:
            ProgressTracker.record_event(
                progress_path,
                phase="build",
                total=progress_total,
                current=cid,
                status="fail",
                detail=str(exc),
            )
        return 1


def _build_subprocess_cmd(
    spec: IndexBuildSpec,
    *,
    script: Path,
    data_dir: Path,
    output_root: Path,
    batch_size: int,
    force: bool,
    optimize_threads: int,
    progress_path: Path,
    progress_total: int,
    index_types: list[str],
) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--build-one",
        "--dataset",
        spec.dataset,
        "--config-id",
        spec.config_id(),
        "--index-types",
        ",".join(index_types),
        "--data-dir",
        str(data_dir),
        "--output-root",
        str(output_root),
        "--batch-size",
        str(batch_size),
        "--optimize-threads",
        str(optimize_threads),
        "--progress-file",
        str(progress_path),
        "--progress-total",
        str(progress_total),
        *(["--force-build"] if force else []),
    ]


def run_build_phase(
    specs: list[IndexBuildSpec],
    *,
    data_dir: Path,
    output_root: Path,
    batch_size: int,
    force: bool,
    build_jobs: int,
    progress_path: Path,
    optimize_threads: int,
    index_types: list[str],
) -> None:
    todo = specs_needing_build(specs, output_root, force)
    skip_count = len(specs) - len(todo)
    if skip_count:
        log(f"BUILD: {skip_count} collections already exist (use --force-build to rebuild)")
    if not todo:
        log("BUILD: nothing to do")
        return

    jobs = max(1, build_jobs)
    progress = ProgressTracker(progress_path, "build", len(todo), reset=True)
    log(f"BUILD: {len(todo)} to build, jobs={jobs}, optimize_threads={optimize_threads}")

    script = Path(__file__).resolve()

    if jobs == 1:
        for spec in todo:
            run_build_one(
                spec,
                data_dir=data_dir,
                output_root=output_root,
                batch_size=batch_size,
                force=force,
                optimize_threads=optimize_threads,
                progress_path=progress_path,
                progress_total=len(todo),
            )
        return

    # 并行：每个 build 独立 Python 子进程（exec），避免 fork+zvec 死锁
    pending = list(todo)
    running: list[tuple[IndexBuildSpec, subprocess.Popen[Any]]] = []
    log_dir = progress_path.parent / "build_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    while pending or running:
        while pending and len(running) < jobs:
            spec = pending.pop(0)
            cid = spec.config_id()
            log_path = log_dir / f"{spec.dataset}_{cid}.log"
            cmd = _build_subprocess_cmd(
                spec,
                script=script,
                data_dir=data_dir,
                output_root=output_root,
                batch_size=batch_size,
                force=force,
                optimize_threads=optimize_threads,
                progress_path=progress_path,
                progress_total=len(todo),
                index_types=index_types,
            )
            log(f"BUILD spawn: {spec.dataset}/{cid} -> {log_path.name}")
            with log_path.open("w", encoding="utf-8") as logfh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=logfh,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
            running.append((spec, proc))

        still: list[tuple[IndexBuildSpec, subprocess.Popen[Any]]] = []
        for spec, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((spec, proc))
                continue
            cid = spec.config_id()
            if rc == 0:
                log(f"BUILD finished: {spec.dataset}/{cid} (exit 0)")
            else:
                log(f"BUILD FAILED: {spec.dataset}/{cid} (exit {rc})")
        running = still

        if running:
            time.sleep(2)

    log(f"BUILD parallel phase complete; see {progress_path} and {log_dir}/")


# ---------------------------------------------------------------------------
# 搜索 / 计时
# ---------------------------------------------------------------------------


def make_query_param(
    spec: IndexBuildSpec,
    ef: int,
    prefetch_offset: int,
    prefetch_lines: int,
) -> HnswQueryParam | VamanaQueryParam:
    extra: dict[str, int] = {}
    if prefetch_offset:
        extra["prefetch_offset"] = prefetch_offset
    if prefetch_lines:
        extra["prefetch_lines"] = prefetch_lines
    if spec.index_type == "vamana":
        return VamanaQueryParam(ef_search=ef, extra_params=extra)
    return HnswQueryParam(ef=ef, extra_params=extra)


def recall_at_k(results: list, ground_truth: list, k: int) -> float:
    nq = len(results)
    if nq == 0:
        return 0.0
    total = 0.0
    for res, gt in zip(results, ground_truth):
        gt_slice = gt[:k]
        if not gt_slice:
            continue
        knn = {int(x) for x in res[:k]}
        hit = sum(1 for g in gt_slice if int(g) in knn)
        total += hit / k
    return total / nq


def bench_ann_bench(
    raw,
    X_test: np.ndarray,
    param: HnswQueryParam | VamanaQueryParam,
    count: int,
    runs: int,
) -> tuple[float, list]:
    """同一 setting 跑 runs 次全量查询，取最优 mean latency。"""
    algo = ZvecAnnBenchDocIdsANN(raw, param)
    best_t = float("inf")
    best_results = None

    for _ in range(runs):
        results = []
        times = []
        for v in X_test:
            algo.prepare_query(v, count)
            t0 = time.perf_counter()
            algo.run_prepared_query()
            times.append(time.perf_counter() - t0)
            results.append(algo.get_prepared_query_results())
        mean_t = sum(times) / len(times)
        if mean_t < best_t:
            best_t = mean_t
            best_results = results

    assert best_results is not None
    return best_t, best_results


def load_dataset_queries(
    dataset: str,
    data_dir: Path,
    count: int,
) -> tuple[np.ndarray, list]:
    info = DATASET_INFO[dataset]
    queries = load_query_vectors(
        data_dir / info["query_file"],
        first_sep=";",
        second_sep=" ",
    )
    gt = load_ground_truth_ids(
        data_dir / info["gt_file"],
        first_sep=";",
        second_sep=" ",
    )
    X_test = np.ascontiguousarray(np.stack(queries), dtype=np.float32)
    return X_test, gt


def measure_ef_recall(
    spec: IndexBuildSpec,
    raw,
    X_test: np.ndarray,
    gt: list,
    *,
    ef: int,
    count: int,
    runs: int,
    cache: dict[int, SearchResult],
) -> SearchResult:
    """在 po=0, pl=0 下测量单个 ef 的 recall/QPS（带缓存）。"""
    cached = cache.get(ef)
    if cached is not None:
        return cached

    param = make_query_param(spec, ef, 0, 0)
    best_t, results = bench_ann_bench(raw, X_test, param, count, runs)
    rec = recall_at_k(results, gt, count)
    row = SearchResult(
        ef=ef,
        prefetch_offset=0,
        prefetch_lines=0,
        recall=rec,
        qps=1.0 / best_t,
        mean_us=best_t * 1e6,
    )
    cache[ef] = row
    log(
        f"      ef={ef:4d}  recall={rec:.4f}  "
        f"QPS={row.qps:,.1f}  mean={row.mean_us:.1f}us  (runs={runs})"
    )
    return row


def binary_search_min_ef(
    ef_candidates: list[int],
    threshold: float,
    measure: Callable[[int], SearchResult],
) -> int | None:
    """在有序 ef 候选上二分找满足 recall ≥ threshold 的最小 ef。

    前提：recall 随 ef 单调不降（HNSW/Vamana 通常成立）。
    """
    if not ef_candidates:
        return None

    if measure(ef_candidates[-1]).recall < threshold:
        return None

    lo, hi = 0, len(ef_candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if measure(ef_candidates[mid]).recall >= threshold:
            hi = mid
        else:
            lo = mid + 1
    return ef_candidates[lo]


def tune_po_pl_at_ef(
    spec: IndexBuildSpec,
    raw,
    X_test: np.ndarray,
    gt: list,
    *,
    ef: int,
    count: int,
    po_sweep: list[int],
    pl_sweep: list[int],
    recall_threshold: float,
    runs: int,
) -> SearchResult | None:
    """阶段 B：固定 ef，枚举 po×pl，取 recall≥T 且 QPS 最高。"""
    best: SearchResult | None = None
    combos = list(itertools.product(po_sweep, pl_sweep))
    for idx, (po, pl) in enumerate(combos, 1):
        param = make_query_param(spec, ef, po, pl)
        best_t, results = bench_ann_bench(raw, X_test, param, count, runs)
        rec = recall_at_k(results, gt, count)
        if rec < recall_threshold:
            continue
        row = SearchResult(
            ef=ef,
            prefetch_offset=po,
            prefetch_lines=pl,
            recall=rec,
            qps=1.0 / best_t,
            mean_us=best_t * 1e6,
        )
        if best is None or row.qps > best.qps:
            best = row
            log(
                f"      po/pl [{idx}/{len(combos)}] PO={po} PL={pl} "
                f"recall={rec:.4f} QPS={row.qps:,.1f}  <== best so far"
            )
    return best


def search_one_build(
    spec: IndexBuildSpec,
    collection_path: Path,
    X_test: np.ndarray,
    gt: list,
    *,
    count: int,
    ef_sweep: list[int],
    po_sweep: list[int],
    pl_sweep: list[int],
    recall_thresholds: list[float],
    runs: int,
) -> BuildSearchSummary:
    summary = BuildSearchSummary(build=spec)
    dimension = DATASET_INFO[spec.dataset]["dimension"]
    po_list, pl_list, po_cap, pl_cap = prefetch_sweep_for_build(
        index_type=spec.index_type,
        degree=spec.m,
        dimension=dimension,
        quantize=spec.quantize,
        po_candidates=po_sweep,
        pl_candidates=pl_sweep,
    )
    elem_bytes = effective_element_size(dimension, spec.quantize)
    log(
        f"    prefetch bounds: PO<={po_cap} PL<={pl_cap} "
        f"(elem={elem_bytes}B)  grid={len(po_list)}x{len(pl_list)}"
    )

    col = zvec_open(
        str(collection_path),
        CollectionOption(read_only=True, enable_mmap=True),
    )
    raw = col._obj

    ef_cache: dict[int, SearchResult] = {}

    def measure(ef: int) -> SearchResult:
        return measure_ef_recall(
            spec, raw, X_test, gt,
            ef=ef, count=count, runs=runs, cache=ef_cache,
        )

    for thr in recall_thresholds:
        key = f"{thr:.3f}"
        log(f"    phase A: recall>={thr:.3f} — binary search min ef (runs={runs})")
        min_ef = binary_search_min_ef(ef_sweep, thr, measure)
        if min_ef is None:
            log(f"    phase B: recall>={thr:.3f} — no ef reached threshold")
            summary.per_recall[key] = None
            continue

        min_row = ef_cache[min_ef]
        log(
            f"      min_ef={min_ef}  recall={min_row.recall:.4f}  "
            f"QPS={min_row.qps:,.1f}"
        )

        log(f"    phase B: recall>={thr:.3f} — ef={min_ef}, tune po×pl (runs={runs})")
        best = tune_po_pl_at_ef(
            spec, raw, X_test, gt,
            ef=min_ef,
            count=count,
            po_sweep=po_list,
            pl_sweep=pl_list,
            recall_threshold=thr,
            runs=runs,
        )
        summary.per_recall[key] = best
        if best:
            log(
                f"      -> ef={best.ef} PO={best.prefetch_offset} "
                f"PL={best.prefetch_lines} recall={best.recall:.4f} "
                f"QPS={best.qps:,.1f}"
            )
        else:
            log(f"      -> no po/pl combo met recall>={thr:.3f} at ef={min_ef}")

    summary.ef_curve = sorted(ef_cache.values(), key=lambda r: r.ef)
    col = None
    return summary


def pick_global_bests(
    summaries: list[BuildSearchSummary],
    recall_thresholds: list[float],
) -> list[GlobalBest]:
    bests: list[GlobalBest] = []
    for thr in recall_thresholds:
        key = f"{thr:.3f}"
        candidates: list[tuple[BuildSearchSummary, SearchResult]] = []
        for sm in summaries:
            row = sm.per_recall.get(key)
            if row is not None:
                candidates.append((sm, row))
        if not candidates:
            continue
        sm, row = max(candidates, key=lambda x: x[1].qps)
        bests.append(
            GlobalBest(
                dataset=sm.build.dataset,
                recall_threshold=thr,
                build=sm.build,
                search=row,
            )
        )
    return bests


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _spec_to_dict(spec: IndexBuildSpec) -> dict[str, Any]:
    return asdict(spec)


def _search_to_dict(row: SearchResult) -> dict[str, Any]:
    return asdict(row)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log(f"Results saved: {path}")


def _dataset_payload(
    summaries: list[BuildSearchSummary],
    recall_thresholds: list[float],
    dataset: str,
) -> dict[str, Any]:
    global_bests = pick_global_bests(summaries, recall_thresholds)
    return {
        "global_best": [
            {
                "recall_threshold": gb.recall_threshold,
                "build": _spec_to_dict(gb.build),
                "search": _search_to_dict(gb.search),
            }
            for gb in global_bests
        ],
        "per_build": [
            {
                "build": _spec_to_dict(sm.build),
                "ef_curve": [_search_to_dict(r) for r in sm.ef_curve],
                "per_recall": {
                    k: _search_to_dict(v) if v else None
                    for k, v in sm.per_recall.items()
                },
            }
            for sm in summaries
        ],
    }


def run_search_phase(
    specs: list[IndexBuildSpec],
    *,
    datasets: list[str],
    data_dir: Path,
    output_root: Path,
    count: int,
    po_sweep: list[int],
    pl_sweep: list[int],
    recall_thresholds: list[float],
    ef_hnsw: list[int],
    ef_vamana: list[int],
    runs: int,
    progress: ProgressTracker,
) -> dict[str, Any]:
    query_cache: dict[str, tuple[np.ndarray, list]] = {}
    for ds in datasets:
        query_cache[ds] = load_dataset_queries(ds, data_dir, count)

    all_payload: dict[str, Any] = {
        "mode": "ann_bench_doc_ids",
        "runs": runs,
        "recall_thresholds": recall_thresholds,
        "datasets": {},
    }

    searchable = [
        s for s in specs
        if (s.collection_dir(output_root) / "collection").exists()
    ]
    missing = len(specs) - len(searchable)
    if missing:
        log(f"SEARCH: {missing} configs missing collection (skipped)")

    for ds in datasets:
        ds_specs = [s for s in searchable if s.dataset == ds]
        log(f"SEARCH dataset={ds.upper()} configs={len(ds_specs)}")
        summaries: list[BuildSearchSummary] = []
        X_test, gt = query_cache[ds]
        partial_path = output_root / f"{ds}_grid_search_results.partial.json"

        for spec in ds_specs:
            cid = spec.config_id()
            collection_path = spec.collection_dir(output_root) / "collection"
            ef_sweep = ef_vamana if spec.index_type == "vamana" else ef_hnsw
            try:
                sm = search_one_build(
                    spec,
                    collection_path,
                    X_test,
                    gt,
                    count=count,
                    ef_sweep=ef_sweep,
                    po_sweep=po_sweep,
                    pl_sweep=pl_sweep,
                    recall_thresholds=recall_thresholds,
                    runs=runs,
                )
                summaries.append(sm)
                progress.event(current=f"{ds}/{cid}", status="ok")
            except Exception as exc:
                progress.event(
                    current=f"{ds}/{cid}",
                    status="fail",
                    detail=str(exc),
                )
                log(traceback.format_exc())

            # 增量 checkpoint，避免 nohup 中断丢结果
            ds_payload = _dataset_payload(summaries, recall_thresholds, ds)
            save_json(partial_path, ds_payload)

        print_global_summary(pick_global_bests(summaries, recall_thresholds))
        ds_payload = _dataset_payload(summaries, recall_thresholds, ds)
        all_payload["datasets"][ds] = ds_payload
        out_file = output_root / f"{ds}_grid_search_results.json"
        save_json(out_file, ds_payload)
        if partial_path.exists():
            partial_path.unlink()

    combined = output_root / "grid_search_all.json"
    save_json(combined, all_payload)
    return all_payload


def print_global_summary(bests: list[GlobalBest]) -> None:
    if not bests:
        print("\n(no global bests)")
        return
    dataset = bests[0].dataset
    print(f"\n{'=' * 72}")
    print(f"GLOBAL BEST — {dataset.upper()} (ann_bench_doc_ids)")
    print(f"{'=' * 72}")
    print(
        f"{'recall≥':>8s}  {'QPS':>10s}  {'ef':>5s}  {'PO':>4s}  {'PL':>3s}  "
        f"{'recall':>7s}  build_config"
    )
    print("-" * 72)
    for gb in bests:
        s = gb.search
        b = gb.build
        print(
            f"{gb.recall_threshold:8.3f}  {s.qps:10,.1f}  {s.ef:5d}  "
            f"{s.prefetch_offset:4d}  {s.prefetch_lines:3d}  "
            f"{s.recall:7.4f}  {b.config_id()}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets", type=str, default="sift,gist",
        help="Comma-separated: sift, gist",
    )
    p.add_argument(
        "--dataset", choices=["sift", "gist"], default=None,
        help="Single dataset (for --build-one).",
    )
    p.add_argument(
        "--index-types", type=str, default="hnsw,vamana",
        help="Comma-separated: hnsw, vamana",
    )
    p.add_argument(
        "--data-dir", type=Path, default=Path("/root/zvec_workspace/output"),
    )
    p.add_argument(
        "--output-root", type=Path,
        default=Path("/root/py/zvec/bench/core/output/grid_search"),
        help="Root dir for built collections and JSON results.",
    )
    p.add_argument("--count", type=int, default=10, help="top-k for recall/QPS")
    p.add_argument(
        "--recall-thresholds", type=str,
        default=",".join(str(x) for x in RECALL_THRESHOLDS),
    )
    p.add_argument(
        "--ef-sweep-hnsw", type=str,
        default=",".join(str(x) for x in HNSW_EF_SWEEP),
    )
    p.add_argument(
        "--ef-sweep-vamana", type=str,
        default=",".join(str(x) for x in VAMANA_EF_SWEEP),
    )
    p.add_argument(
        "--po-sweep", type=str,
        default=",".join(str(x) for x in PREFETCH_OFFSET_SWEEP),
    )
    p.add_argument(
        "--pl-sweep", type=str,
        default=",".join(str(x) for x in PREFETCH_LINES_SWEEP),
    )
    p.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help="Full-query repetitions per setting; best QPS is reported (default: 3).",
    )
    p.add_argument(
        "--build", action="store_true",
        help="Build missing collections (then search unless --skip-search).",
    )
    p.add_argument(
        "--build-only", action="store_true",
        help="Only build (--build-jobs parallel), skip search.",
    )
    p.add_argument(
        "--search-only", action="store_true",
        help="Only search; collections must already exist.",
    )
    p.add_argument(
        "--force-build", action="store_true",
        help="Rebuild even if collection exists.",
    )
    p.add_argument(
        "--skip-search", action="store_true",
        help="Alias for build-only when combined with --build.",
    )
    p.add_argument(
        "--build-jobs", type=int, default=1,
        help="Parallel builds via independent Python subprocesses (not fork).",
    )
    p.add_argument(
        "--optimize-threads", type=int, default=DEFAULT_OPTIMIZE_THREADS,
        help="zvec optimize_threads per build process (default: 8).",
    )
    p.add_argument(
        "--build-one", action="store_true",
        help="Build a single config (--dataset + --config-id); for parallel workers.",
    )
    p.add_argument(
        "--config-id", type=str, default=None,
        help="Build config id, e.g. hnsw_m16_efc300_int8 (with --build-one).",
    )
    p.add_argument(
        "--progress-total", type=int, default=0,
        help="Total build count for progress.json (used by --build-one workers).",
    )
    p.add_argument(
        "--export-build-list", action="store_true",
        help="Print dataset\\tconfig_id lines for pending builds and exit.",
    )
    p.add_argument(
        "--progress-file", type=Path, default=None,
        help="Progress JSON path (default: <output-root>/progress.json).",
    )
    p.add_argument(
        "--max-builds", type=int, default=0,
        help="Limit number of build configs (0 = all). For debugging.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print grid size and config ids, then exit.",
    )
    p.add_argument(
        "--batch-size", type=int, default=1024,
        help="Insert batch size during build.",
    )
    return p.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    index_types = [x.strip() for x in args.index_types.split(",") if x.strip()]
    recall_thresholds = _parse_float_list(args.recall_thresholds)
    po_sweep = _parse_int_list(args.po_sweep)
    pl_sweep = _parse_int_list(args.pl_sweep)
    ef_hnsw = _parse_int_list(args.ef_sweep_hnsw)
    ef_vamana = _parse_int_list(args.ef_sweep_vamana)

    if args.build_only:
        args.build = True
        args.skip_search = True
    if args.search_only:
        args.skip_search = False

    specs = list(iter_build_specs(datasets, index_types))
    if args.max_builds > 0:
        specs = specs[: args.max_builds]

    progress_path = args.progress_file or (args.output_root / "progress.json")

    if args.export_build_list:
        todo = specs_needing_build(specs, args.output_root, args.force_build)
        for spec in todo:
            print(f"{spec.dataset}\t{spec.config_id()}")
        print(
            f"# export-build-list: {len(todo)} pending",
            file=sys.stderr,
        )
        return 0

    if args.build_one:
        if not args.dataset or not args.config_id:
            log("--build-one requires --dataset and --config-id")
            return 1
        try:
            spec = find_spec_by_config_id(
                args.config_id,
                datasets=datasets,
                index_types=index_types,
                dataset=args.dataset,
            )
        except ValueError as exc:
            log(str(exc))
            return 1
        total = args.progress_total or 1
        return run_build_one(
            spec,
            data_dir=args.data_dir,
            output_root=args.output_root,
            batch_size=args.batch_size,
            force=args.force_build,
            optimize_threads=args.optimize_threads,
            progress_path=progress_path,
            progress_total=total,
        )

    if args.dry_run:
        log(f"Grid: {len(specs)} build configs")
        print(f"  datasets={datasets} index_types={index_types}")
        print(f"  recall thresholds={recall_thresholds}")
        print(f"  ef candidates: hnsw={len(ef_hnsw)} vamana={len(ef_vamana)} "
              f"(binary search per threshold, ~O(log N) probes)")
        print(f"  runs={args.runs} (best of full-query passes per setting)")
        print(f"  build_jobs={args.build_jobs}  optimize_threads={args.optimize_threads}")
        print(f"  progress_file={progress_path}")
        print(f"  quantize: sift={QUANTIZE_CHOICES_SIFT}  gist={QUANTIZE_CHOICES_GIST}")
        print(f"  po×pl base grid: {len(po_sweep)}×{len(pl_sweep)} "
              f"(clamped per build by degree & vector width)")
        if specs:
            ex = specs[0]
            dim = DATASET_INFO[ex.dataset]["dimension"]
            po_ex, pl_ex, po_cap, pl_cap = prefetch_sweep_for_build(
                index_type=ex.index_type,
                degree=ex.m,
                dimension=dim,
                quantize=ex.quantize,
                po_candidates=po_sweep,
                pl_candidates=pl_sweep,
            )
            print(f"  example {ex.config_id()}: PO cap={po_cap} -> {po_ex}")
            print(f"                         PL cap={pl_cap} -> {pl_ex}")
        for spec in specs[:10]:
            print(f"    {spec.dataset}/{spec.config_id()}")
        if len(specs) > 10:
            print(f"    ... and {len(specs) - 10} more")
        return 0

    do_build = (args.build or args.force_build or args.build_only) and not args.search_only
    do_search = not args.skip_search and not args.build_only

    if not do_build and not do_search:
        log("Nothing to do: pass --build-only, --search-only, or --build")
        return 1

    if do_build:
        run_build_phase(
            specs,
            data_dir=args.data_dir,
            output_root=args.output_root,
            batch_size=args.batch_size,
            force=args.force_build,
            build_jobs=args.build_jobs,
            progress_path=progress_path,
            optimize_threads=args.optimize_threads,
            index_types=index_types,
        )
        # run_build_phase logs per-item; re-read progress file for summary
        if progress_path.exists():
            with progress_path.open(encoding="utf-8") as fh:
                bp = json.load(fh)
            log(
                f"BUILD summary: ok={bp.get('done')} skip={bp.get('skipped')} "
                f"fail={bp.get('failed')} elapsed={bp.get('elapsed_s', 0) / 60:.1f}min"
            )

    if do_search:
        init_zvec_safe(optimize_threads=args.optimize_threads)
        searchable = [
            s for s in specs
            if (s.collection_dir(args.output_root) / "collection").exists()
        ]
        search_progress = ProgressTracker(progress_path, "search", max(1, len(searchable)))
        run_search_phase(
            specs,
            datasets=datasets,
            data_dir=args.data_dir,
            output_root=args.output_root,
            count=args.count,
            po_sweep=po_sweep,
            pl_sweep=pl_sweep,
            recall_thresholds=recall_thresholds,
            ef_hnsw=ef_hnsw,
            ef_vamana=ef_vamana,
            runs=args.runs,
            progress=search_progress,
        )
        if progress_path.exists():
            with progress_path.open(encoding="utf-8") as fh:
                sp = json.load(fh)
            log(
                f"SEARCH summary: ok={sp.get('done')} skip={sp.get('skipped')} "
                f"fail={sp.get('failed')} elapsed={sp.get('elapsed_s', 0) / 60:.1f}min"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
