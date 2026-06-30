#!/usr/bin/env python3
"""Convert txt vector files to .vecs (Python equivalent of tools/core/txt2vecs)."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np

from common.io import VECS_HEADER_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--first-sep", default=";")
    parser.add_argument("--second-sep", default=" ")
    return parser.parse_args()


def _read_txt(path: Path, dimension: int, first_sep: str, second_sep: str):
    keys: list[int] = []
    vectors: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(first_sep)
            key = int(parts[0]) if len(parts) > 1 else len(keys)
            feature_str = parts[1] if len(parts) > 1 else parts[0]
            values = np.fromstring(feature_str, sep=second_sep, dtype=np.float32)
            if values.size != dimension:
                raise ValueError(
                    f"line {line_no}: expected dim={dimension}, got {values.size}"
                )
            keys.append(key)
            vectors.append(values)
    return np.asarray(keys, dtype=np.uint64), np.stack(vectors)


def write_vecs(
    output: Path,
    keys: np.ndarray,
    vectors: np.ndarray,
) -> None:
    num_vecs, dimension = vectors.shape
    meta_size = 0
    header = struct.pack(
        "<QHHIQ",
        num_vecs,
        0,
        1,
        meta_size,
    )
    header += struct.pack(
        "<QQQQQQQQQQ",
        0,
        0,
        0,
        num_vecs * dimension * 4,
        0,
        0,
        (1 << 63) - 1,
        0,
        (1 << 63) - 1,
        0,
    )
    if len(header) != VECS_HEADER_SIZE:
        raise RuntimeError(f"unexpected header size: {len(header)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        fh.write(header)
        vectors.astype(np.float32).tofile(fh)
        keys.astype(np.uint64).tofile(fh)


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    keys, vectors = _read_txt(
        args.input, args.dimension, args.first_sep, args.second_sep
    )
    write_vecs(args.output, keys, vectors)
    print(f"Wrote {vectors.shape[0]} vectors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
