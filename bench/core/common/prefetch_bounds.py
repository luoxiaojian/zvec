"""PO/PL 搜索上界：与 ~/main/workspace/tune_po_pl.py 一致。

- PO (prefetch_offset): 不超过节点 level-0 出度
  - HNSW: 2 × m
  - Vamana: max_degree (m)
- PL (prefetch_lines): 不超过 ceil(element_size / 64)
  - element_size 为存储中单向量字节数（维度 × 类型宽度 + 量化元数据）
"""

from __future__ import annotations

import math

CACHE_LINE_BYTES = 64
HNSW_L0_MULTIPLIER = 2


def effective_po_cap(index_type: str, degree: int) -> int:
    """节点最大出度上界。"""
    if index_type == "hnsw":
        return degree * HNSW_L0_MULTIPLIER
    if index_type == "vamana":
        return degree
    raise ValueError(f"unknown index_type: {index_type}")


def effective_element_size(dimension: int, quantize: str) -> int:
    """单向量在索引中的存储字节数。"""
    dim = int(dimension)
    q = quantize.strip().lower()
    if q == "int8":
        return dim + 20
    if q == "uniform_int8":
        return dim
    if q == "uniform_uint8":
        return dim + 8
    if q == "fp32":
        return dim * 4
    if q == "fp16":
        return dim * 2
    raise ValueError(f"unknown quantize: {quantize}")


def effective_pl_cap(dimension: int, quantize: str) -> int:
    """PL 上界 = ceil(element_size / 64)。"""
    es = effective_element_size(dimension, quantize)
    return (es + CACHE_LINE_BYTES - 1) // CACHE_LINE_BYTES


def clamp_po_list(po_list: list[int], cap: int | None) -> list[int]:
    """保留 <= cap 的 PO，并确保 cap 在列表中。"""
    if cap is None:
        return sorted(set(po_list))
    kept = [p for p in po_list if p <= cap]
    if cap not in kept:
        kept.append(cap)
    return sorted(set(kept))


def clamp_pl_list(pl_list: list[int], cap: int | None) -> list[int]:
    """保留 <= cap 的 PL（PL=0 始终保留），并确保 cap 在列表中。"""
    if cap is None:
        return sorted(set(pl_list))
    kept = [p for p in pl_list if p == 0 or p <= cap]
    if cap not in kept:
        kept.append(cap)
    return sorted(set(kept))


def prefetch_sweep_for_build(
    *,
    index_type: str,
    degree: int,
    dimension: int,
    quantize: str,
    po_candidates: list[int],
    pl_candidates: list[int],
) -> tuple[list[int], list[int], int, int]:
    """按度数与向量宽度裁剪 PO/PL 候选，返回 (po_list, pl_list, po_cap, pl_cap)。"""
    po_cap = effective_po_cap(index_type, degree)
    pl_cap = effective_pl_cap(dimension, quantize)
    po_list = clamp_po_list(po_candidates, po_cap)
    pl_list = clamp_pl_list(pl_candidates, pl_cap)
    return po_list, pl_list, po_cap, pl_cap
