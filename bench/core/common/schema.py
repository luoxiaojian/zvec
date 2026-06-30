from __future__ import annotations

from typing import Any

import zvec
from zvec import (
    CollectionSchema,
    HnswIndexParam,
    HnswQueryParam,
    VamanaIndexParam,
    VamanaQueryParam,
    VectorSchema,
)
from zvec.typing import DataType, MetricType, QuantizeType

from .config import BuildConfig, SearchConfig

VECTOR_FIELD = "vector"


def _metric_from_name(name: str) -> MetricType:
    normalized = name.strip().lower()
    if normalized in {"squaredeuclidean", "l2", "l2sq"}:
        return MetricType.L2
    if normalized in {"innerproduct", "ip"}:
        return MetricType.IP
    if normalized == "cosine":
        return MetricType.COSINE
    raise ValueError(f"unsupported metric: {name}")


def _quantize_from_converter(converter_name: str) -> QuantizeType:
    normalized = converter_name.strip().lower()
    if "uniformint8" in normalized.replace("_", ""):
        return QuantizeType.UNIFORM_INT8
    if "int8" in normalized:
        return QuantizeType.INT8
    if "fp16" in normalized:
        return QuantizeType.FP16
    return QuantizeType.UNDEFINED


def build_collection_schema(cfg: BuildConfig) -> CollectionSchema:
    metric = _metric_from_name(cfg.metric_name)
    quantize = _quantize_from_converter(cfg.quantizer)
    params = cfg.builder_params

    if "Vamana" in cfg.index_type:
        index_param = VamanaIndexParam(
            metric_type=metric,
            max_degree=int(params.get("proxima.vamana.streamer.max_degree", 48)),
            search_list_size=int(
                params.get("proxima.vamana.streamer.search_list_size", 100)
            ),
            alpha=float(params.get("proxima.vamana.streamer.alpha", 1.2)),
            use_contiguous_memory=True,
            use_id_map=not cfg.disable_id_map,
            quantize_type=quantize,
        )
    elif "Hnsw" in cfg.index_type:
        index_param = HnswIndexParam(
            metric_type=metric,
            m=int(params.get("proxima.hnsw.streamer.max_neighbor_count", 48)),
            use_contiguous_memory=True,
            quantize_type=quantize,
        )
    else:
        raise ValueError(f"unsupported builder class: {cfg.index_type}")

    return CollectionSchema(
        name=f"{cfg.dataset}_bench",
        fields=[],
        vectors=[
            VectorSchema(
                VECTOR_FIELD,
                DataType.VECTOR_FP32,
                dimension=cfg.dimension,
                index_param=index_param,
            )
        ],
    )


def make_query_param(cfg: SearchConfig) -> HnswQueryParam | VamanaQueryParam:
    raw = cfg.query_param
    index_type = str(raw.get("index_type", cfg.index_config.get("index_type", "")))

    extra = {}
    if "prefetch_offset" in raw:
        extra["prefetch_offset"] = int(raw["prefetch_offset"])
    if "prefetch_lines" in raw:
        extra["prefetch_lines"] = int(raw["prefetch_lines"])

    if index_type in {"kVamana", "VAMANA", "vamana"}:
        return VamanaQueryParam(
            ef_search=int(raw.get("ef_search", 200)),
            extra_params=extra,
        )

    return HnswQueryParam(
        ef=int(raw.get("ef_search", raw.get("ef", 300))),
        extra_params=extra,
    )


def init_zvec(*, optimize_threads: int, query_threads: int) -> None:
    zvec.init(
        log_level=zvec.LogLevel.INFO,
        optimize_threads=optimize_threads,
        query_threads=query_threads,
    )
