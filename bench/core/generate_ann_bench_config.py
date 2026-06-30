#!/usr/bin/env python3
"""Generate ann-benchmarks zvec config.yml from grid_search results.

Reads sift/gist *_grid_search_results.json, selects a compact set of build
configs that cover top QPS at each recall tier (with redundancy), and writes
query-arg sweeps bracketing the tuned ef / prefetch_offset / prefetch_lines.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

EF_SWEEP = [
    8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 112, 128,
    160, 192, 256, 384, 512, 768, 1024,
]

RECALL_THRESHOLDS = [0.9, 0.95, 0.98, 0.99]
TOP_PER_TIER = 2  # min distinct builds per dataset × recall tier
MAX_GROUPS = 10


def build_key(b: dict[str, Any]) -> tuple:
    if b["index_type"] == "hnsw":
        return ("hnsw", b["m"], b["ef_construction"], b["quantize"])
    return ("vamana", b["m"], b["search_list_size"], b["alpha"], b["quantize"])


def alpha_slug(alpha: float) -> str:
    return f"a{round(alpha * 10)}"


def group_label(k: tuple) -> str:
    if k[0] == "hnsw":
        return f"HNSW-M{k[1]}-efc{k[2]}"
    return f"Vamana-D{k[1]}-L{k[2]}-{alpha_slug(k[3])}"


def bracket_ef(ef: int, margin: int = 1) -> list[int]:
    idx = min(range(len(EF_SWEEP)), key=lambda i: abs(EF_SWEEP[i] - ef))
    lo = max(0, idx - margin)
    hi = min(len(EF_SWEEP), idx + margin + 2)
    return EF_SWEEP[lo:hi]


def format_inline_list(vals: list[Any]) -> str:
    return "[" + ", ".join(format_yaml_scalar(v) for v in vals) + "]"


def load_results(grid_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ("sift", "gist"):
        path = grid_dir / f"{name}_grid_search_results.json"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as fh:
            out[name] = json.load(fh)
    return out


def select_build_keys(results: dict[str, dict]) -> list[tuple]:
    """Greedy: at least TOP_PER_TIER distinct builds per dataset × recall tier."""
    selected: set[tuple] = set()
    scores: dict[tuple, float] = defaultdict(float)
    for ds, data in results.items():
        for thr in RECALL_THRESHOLDS:
            key = f"{thr:.3f}"
            candidates = []
            for pb in data["per_build"]:
                row = pb["per_recall"].get(key)
                if row:
                    candidates.append((pb["build"], row))
            candidates.sort(key=lambda x: -x[1]["qps"])
            for rank, (b, s) in enumerate(candidates[:5]):
                scores[build_key(b)] += (5 - rank) * s["qps"] / 1000.0

    tier_keys = [(ds, thr) for ds in results for thr in RECALL_THRESHOLDS]

    for ds, thr in tier_keys:
        key = f"{thr:.3f}"
        candidates: list[tuple[dict, dict]] = []
        for pb in results[ds]["per_build"]:
            row = pb["per_recall"].get(key)
            if row:
                candidates.append((pb["build"], row))
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[1]["qps"])
        added = 0
        for b, _ in candidates:
            bk = build_key(b)
            if bk not in selected:
                if len(selected) >= MAX_GROUPS:
                    break
                selected.add(bk)
                added += 1
            if added >= TOP_PER_TIER:
                break

    for bk in sorted(scores, key=lambda k: -scores[k]):
        if len(selected) >= MAX_GROUPS:
            break
        selected.add(bk)

    if len(selected) > MAX_GROUPS:
        selected = set(
            sorted(selected, key=lambda k: -scores.get(k, 0.0))[:MAX_GROUPS]
        )

    # Reserve one slot for the best HNSW backup (index-family diversity).
    best_hnsw: tuple | None = None
    best_hnsw_score = 0.0
    for bk, sc in scores.items():
        if bk[0] == "hnsw" and sc > best_hnsw_score:
            best_hnsw = bk
            best_hnsw_score = sc
    if best_hnsw is not None and best_hnsw not in selected:
        if len(selected) >= MAX_GROUPS:
            drop = min(selected, key=lambda k: scores.get(k, 0.0))
            if scores.get(drop, 0.0) < best_hnsw_score:
                selected.discard(drop)
        selected.add(best_hnsw)

    return sorted(selected, key=lambda k: (k[0], k[1], k[2] if k[0] == "hnsw" else k[2]))


def collect_query_args(
    bk: tuple,
    results: dict[str, dict],
) -> tuple[list[int], list[int], list[int]]:
    efs: set[int] = set()
    pos: set[int] = set()
    pls: set[int] = set()

    for ds, data in results.items():
        for thr in RECALL_THRESHOLDS:
            key = f"{thr:.3f}"
            ranked: list[tuple[dict, dict]] = []
            for pb in data["per_build"]:
                row = pb["per_recall"].get(key)
                if row:
                    ranked.append((pb["build"], row))
            ranked.sort(key=lambda x: -x[1]["qps"])
            for rank, (b, row) in enumerate(ranked[:5]):
                if build_key(b) != bk:
                    continue
                if rank >= 3:
                    break
                for ef in bracket_ef(int(row["ef"])):
                    efs.add(ef)
                pos.add(int(row["prefetch_offset"]))
                pls.add(int(row["prefetch_lines"]))

    if not efs:
        for ds, data in results.items():
            for pb in data["per_build"]:
                if build_key(pb["build"]) != bk:
                    continue
                for thr in RECALL_THRESHOLDS:
                    row = pb["per_recall"].get(f"{thr:.3f}")
                    if row:
                        efs.add(int(row["ef"]))
                        pos.add(int(row["prefetch_offset"]))
                        pls.add(int(row["prefetch_lines"]))

    if not efs:
        efs.update([16, 32, 64, 128, 256])
    if not pos:
        pos.add(0)
    if not pls:
        pls.add(0)

    return sorted(efs), sorted(pos), sorted(pls)


def build_run_group(k: tuple, results: dict[str, dict]) -> tuple[str, dict]:
    label = group_label(k)
    ef_list, po_list, pl_list = collect_query_args(k, results)

    if k[0] == "hnsw":
        args: dict[str, Any] = {
            "index": "hnsw",
            "M": k[1],
            "efConstruction": k[2],
            "quantize": ["int8", "uniform_int8"],
        }
    else:
        args = {
            "index": "vamana",
            "max_degree": k[1],
            "search_list_size": k[2],
            "alpha": k[3],
            "quantize": ["int8"],
        }

    query_args = {
        "ef": ef_list,
        "prefetch_offset": po_list,
        "prefetch_lines": pl_list,
    }
    return label, {"args": args, "query_args": query_args}


def format_yaml_value(val: Any) -> str:
    if isinstance(val, list):
        return format_inline_list(val)
    return format_yaml_scalar(val)


def format_yaml_scalar(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, float):
        return str(val)
    return str(val)


def emit_config(run_groups: dict[str, dict]) -> str:
    constructors = [
        ("ZvecQuery", "zvec-query"),
        ("ZvecFastQuery", "zvec-fast_query"),
        ("ZvecFastQueryDocIds", "zvec-fast_query_doc_ids"),
        ("ZvecAnnBenchDocIds", "zvec-ann_bench_doc_ids"),
    ]

    lines = [
        "# Generated from grid_search results (generate_ann_bench_config.py).",
        "# Build configs cover top QPS at recall tiers 0.9–0.99 with redundancy.",
        "float:",
        "  any:",
        "  - base_args: ['@metric', '@dimension']",
        "    constructor: ZvecQuery",
        "    disabled: false",
        "    docker_tag: ann-benchmarks-zvec",
        "    module: ann_benchmarks.algorithms.zvec",
        "    name: zvec-query",
        "    run_groups: &zvec_run_groups",
    ]

    for label, rg in run_groups.items():
        lines.append(f"      {label}:")
        lines.append("        args:")
        for ak, av in rg["args"].items():
            lines.append(f"          {ak}: {format_yaml_value(av)}")
        lines.append("        query_args:")
        for qk, qv in rg["query_args"].items():
            lines.append(f"          {qk}: {format_yaml_value(qv)}")

    for ctor, name in constructors[1:]:
        lines.extend([
            "  - base_args: ['@metric', '@dimension']",
            f"    constructor: {ctor}",
            "    disabled: false",
            "    docker_tag: ann-benchmarks-zvec",
            "    module: ann_benchmarks.algorithms.zvec",
            f"    name: {name}",
            "    run_groups: *zvec_run_groups",
        ])

    return "\n".join(lines) + "\n"


def summarize(run_groups: dict[str, dict]) -> str:
    lines = ["Run groups:", f"  count: {len(run_groups)}"]
    total = 0
    for label, rg in run_groups.items():
        qa = rg["query_args"]
        qargs = rg["args"].get("quantize", [])
        n = len(qa["ef"]) * len(qa["prefetch_offset"]) * len(qa["prefetch_lines"])
        if isinstance(qargs, list):
            n *= len(qargs)
        total += n
        lines.append(
            f"  {label}: ef×po×pl×quant = "
            f"{len(qa['ef'])}×{len(qa['prefetch_offset'])}×"
            f"{len(qa['prefetch_lines'])}×{len(qargs) if isinstance(qargs, list) else 1} "
            f"= {n}"
        )
    lines.append(f"  total query combos per constructor: {total}")
    lines.append(f"  total across 4 constructors: {total * 4}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output/grid_search",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/root/py/ann-benchmarks/ann_benchmarks/algorithms/zvec/config.yml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = load_results(args.grid_dir)
    keys = select_build_keys(results)
    run_groups: dict[str, dict] = {}
    for k in keys:
        label, rg = build_run_group(k, results)
        run_groups[label] = rg

    summary = summarize(run_groups)
    print(summary)

    if args.dry_run:
        print(emit_config(run_groups))
        return 0

    out = args.output
    if out.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = out.with_name(f"config.yml.bak.{stamp}")
        shutil.copy2(out, backup)
        print(f"Backed up: {backup}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(emit_config(run_groups), encoding="utf-8")
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
