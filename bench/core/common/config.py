from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class BuildConfig:
    dataset: str
    build_file: Path
    collection_path: Path
    thread_count: int
    dimension: int
    metric_name: str
    index_type: str
    quantizer: str
    disable_id_map: bool
    builder_params: dict[str, Any]


@dataclass(frozen=True)
class SearchConfig:
    dataset: str
    collection_path: Path
    query_file: Path
    ground_truth_file: Path | None
    query_first_sep: str
    query_second_sep: str
    ground_truth_first_sep: str
    ground_truth_second_sep: str
    topk_list: list[int]
    recall_thread_count: int
    bench_thread_count: int
    bench_iter_count: int
    compare_by_id: bool
    index_config: dict[str, Any]
    query_param: dict[str, Any]
    recall_log_dir: Path | None


def _resolve_path(raw: str, data_dir: Path | None) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if data_dir is not None and path.name.endswith((".vecs", ".txt")):
        candidate = data_dir / path.name
        if candidate.exists():
            return candidate
    return path


def _dimension_from_index_config(index_config: dict[str, Any]) -> int | None:
    dim = index_config.get("dimension")
    return int(dim) if dim is not None else None


def resolve_workspace_paths(workspace_config_dir: Path) -> tuple[str, Path, Path]:
    """Return (dataset, build_yaml, search_yaml) for a workspace config dir."""
    workspace_config_dir = workspace_config_dir.expanduser().resolve()
    dataset = workspace_config_dir.parent.name
    if dataset not in {"sift", "gist"}:
        raise ValueError(
            f"workspace config must live under sift/ or gist/, got: "
            f"{workspace_config_dir}"
        )
    build_yaml = workspace_config_dir / "build.yaml"
    search_yaml = workspace_config_dir / "search_po_pl.yaml"
    if not build_yaml.exists():
        raise FileNotFoundError(f"build.yaml not found: {build_yaml}")
    if not search_yaml.exists():
        raise FileNotFoundError(f"search_po_pl.yaml not found: {search_yaml}")
    return dataset, build_yaml, search_yaml


def _dataset_defaults(dataset: str) -> dict[str, Any]:
    if dataset == "gist":
        return {
            "build_yaml": "gist_vamana_build.yaml",
            "search_yaml": "gist_vamana_search.yaml",
            "dimension": 960,
            "train_vecs": "gist_train.vecs",
            "test_queries": "gist_test.txt",
            "neighbors": "gist_neighbors.txt",
            "collection_subdir": "gist_vamana",
        }
    if dataset == "sift":
        return {
            "build_yaml": "sift_hnsw_build.yaml",
            "search_yaml": "sift_hnsw_search.yaml",
            "dimension": 128,
            "train_vecs": "sift_train.vecs",
            "test_queries": "sift_test.txt",
            "neighbors": "sift_neighbors.txt",
            "collection_subdir": "sift_hnsw",
        }
    raise ValueError(f"unsupported dataset: {dataset}")


def load_build_config(
    dataset: str,
    *,
    config_path: Path | None = None,
    workspace_config_dir: Path | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    collection_path: Path | None = None,
) -> BuildConfig:
    defaults = _dataset_defaults(dataset)
    if workspace_config_dir is not None:
        ws_dataset, build_yaml, search_yaml = resolve_workspace_paths(
            workspace_config_dir
        )
        if ws_dataset != dataset:
            raise ValueError(
                f"--dataset {dataset} does not match workspace parent {ws_dataset}"
            )
        yaml_path = build_yaml
        resolved_collection = workspace_config_dir.expanduser().resolve() / "collection"
    else:
        yaml_path = config_path or (DEFAULT_CONFIG_DIR / defaults["build_yaml"])
        out_root = output_dir or Path("./output")
        resolved_collection = out_root / defaults["collection_subdir"]

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    common = raw["BuilderCommon"]
    build_file = _resolve_path(common["BuildFile"], data_dir)
    if data_dir is not None and not build_file.exists():
        build_file = data_dir / defaults["train_vecs"]

    if collection_path is not None:
        resolved_collection = collection_path
    elif workspace_config_dir is None:
        out_root = output_dir or Path("./output")
        resolved_collection = out_root / defaults["collection_subdir"]

    dimension = int(defaults["dimension"])
    if workspace_config_dir is not None:
        with search_yaml.open("r", encoding="utf-8") as fh:
            search_raw = yaml.safe_load(fh)
        index_config = json.loads(search_raw["IndexCommon"]["IndexConfig"])
        dim_override = _dimension_from_index_config(index_config)
        if dim_override is not None:
            dimension = dim_override

    return BuildConfig(
        dataset=dataset,
        build_file=build_file,
        collection_path=resolved_collection,
        thread_count=int(common.get("ThreadCount", 16)),
        dimension=dimension,
        metric_name=str(common.get("MetricName", "SquaredEuclidean")),
        index_type=str(common.get("BuilderClass", "")),
        quantizer=str(common.get("ConverterName", "")),
        disable_id_map=bool(common.get("DisableIdMap", True)),
        builder_params=dict(raw.get("BuilderParams") or {}),
    )


def load_search_config(
    dataset: str,
    *,
    config_path: Path | None = None,
    workspace_config_dir: Path | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    collection_path: Path | None = None,
) -> SearchConfig:
    defaults = _dataset_defaults(dataset)
    workspace_root: Path | None = None
    if workspace_config_dir is not None:
        ws_dataset, _build_yaml, search_yaml = resolve_workspace_paths(
            workspace_config_dir
        )
        if ws_dataset != dataset:
            raise ValueError(
                f"--dataset {dataset} does not match workspace parent {ws_dataset}"
            )
        yaml_path = search_yaml
        workspace_root = workspace_config_dir.expanduser().resolve()
        default_collection = workspace_root / "collection"
    else:
        yaml_path = config_path or (DEFAULT_CONFIG_DIR / defaults["search_yaml"])
        out_root = output_dir or Path("./output")
        default_collection = out_root / defaults["collection_subdir"]

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    common = raw["IndexCommon"]
    query_cfg = raw.get("QueryConfig") or {}

    resolved_collection = collection_path or default_collection

    query_file = _resolve_path(common["QueryFile"], data_dir)
    if data_dir is not None and not query_file.exists():
        query_file = data_dir / defaults["test_queries"]

    gt_file: Path | None = None
    if common.get("GroundTruthFile"):
        gt_file = _resolve_path(common["GroundTruthFile"], data_dir)
        if data_dir is not None and not gt_file.exists():
            gt_file = data_dir / defaults["neighbors"]

    topk_raw = str(common["TopK"])
    topk_list = [int(x.strip()) for x in topk_raw.split(",") if x.strip()]

    recall_log_dir = None
    if common.get("RecallLogDir"):
        recall_log_dir = Path(common["RecallLogDir"])
        if not recall_log_dir.is_absolute() and workspace_root is not None:
            recall_log_dir = workspace_root / recall_log_dir

    return SearchConfig(
        dataset=dataset,
        collection_path=resolved_collection,
        query_file=query_file,
        ground_truth_file=gt_file,
        query_first_sep=str(common.get("QueryFirstSep", ";")),
        query_second_sep=str(common.get("QuerySecondSep", " ")),
        ground_truth_first_sep=str(common.get("GroundTruthFirstSep", ";")),
        ground_truth_second_sep=str(common.get("GroundTruthSecondSep", " ")),
        topk_list=topk_list,
        recall_thread_count=int(common.get("RecallThreadCount", 1)),
        bench_thread_count=int(common.get("BenchThreadCount", 1)),
        bench_iter_count=int(common.get("BenchIterCount", 10000)),
        compare_by_id=bool(common.get("CompareById", True)),
        index_config=json.loads(common["IndexConfig"]),
        query_param=json.loads(query_cfg["QueryParam"]),
        recall_log_dir=recall_log_dir,
    )
