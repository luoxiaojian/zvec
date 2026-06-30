from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import numpy as np

# Matches tools/core/vecs_common.h VecsHeader (pack=4).
VECS_HEADER_SIZE = 104
VECS_HEADER_PREFIX = struct.Struct("<QHHI")


def read_vecs(
    path: Path,
    dimension: int,
    *,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load dense vectors and keys from a .vecs binary file.

    Layout: [header][meta][dense vectors][uint64 keys]
    """
    element_size = dimension * dtype().itemsize
    with path.open("rb") as fh:
        prefix = fh.read(VECS_HEADER_PREFIX.size)
        if len(prefix) != VECS_HEADER_PREFIX.size:
            raise ValueError(f"truncated vecs header: {path}")

        num_vecs, _meta_size_v1, _version, meta_size = VECS_HEADER_PREFIX.unpack(
            prefix
        )
        fh.seek(VECS_HEADER_SIZE + meta_size)

        vectors = np.fromfile(fh, dtype=dtype, count=num_vecs * dimension)
        if vectors.size != num_vecs * dimension:
            raise ValueError(
                f"unexpected vector payload size in {path}: "
                f"expected {num_vecs * dimension}, got {vectors.size}"
            )
        vectors = vectors.reshape(int(num_vecs), dimension)

        keys = np.fromfile(fh, dtype=np.uint64, count=num_vecs)
        if keys.size != num_vecs:
            raise ValueError(
                f"unexpected key payload size in {path}: "
                f"expected {num_vecs}, got {keys.size}"
            )

    return keys, vectors


def iter_vecs_batches(
    path: Path,
    dimension: int,
    batch_size: int,
    *,
    dtype: np.dtype = np.float32,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    keys, vectors = read_vecs(path, dimension, dtype=dtype)
    total = vectors.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield keys[start:end], vectors[start:end]


def load_query_vectors(
    query_file: Path,
    *,
    first_sep: str = ";",
    second_sep: str = " ",
    as_lists: bool = False,
) -> list[np.ndarray] | list[list[float]]:
    """Parse query txt file used by tools/core recall & bench."""
    queries: list[np.ndarray] = []
    with query_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(first_sep)
            feature_str = parts[1] if len(parts) > 1 else parts[0]
            values = [float(x) for x in feature_str.split(second_sep) if x]
            queries.append(np.asarray(values, dtype=np.float32))
    if as_lists:
        return [query.tolist() for query in queries]
    return queries


def load_ground_truth_ids(
    gt_file: Path,
    *,
    first_sep: str = ";",
    second_sep: str = " ",
) -> list[list[int]]:
    """Load external ground-truth neighbor ids (tools/core txt format)."""
    ground_truth: list[list[int]] = []
    with gt_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(first_sep)
            if len(parts) < 2:
                continue
            ids = [int(x) for x in parts[1].split(second_sep) if x]
            ground_truth.append(ids)
    return ground_truth


def write_txt_vectors(
    output: Path,
    keys: np.ndarray,
    vectors: np.ndarray,
    *,
    first_sep: str = ";",
    second_sep: str = " ",
) -> None:
    """Export vectors to txt2vecs-compatible text format."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for key, vec in zip(keys, vectors, strict=True):
            body = second_sep.join(f"{v:.6g}" for v in vec)
            fh.write(f"{int(key)}{first_sep}{body}\n")
