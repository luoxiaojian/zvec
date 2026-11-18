#!/usr/bin/env python3
"""End-to-end filtered HNSW QPS: stile vs Gandiva JIT vs Arrow per row.

Every run is a fresh process. Variant order rotates over repetitions; the
summary reports the median QPS and median p50 API latency per cell.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def variants(args):
    # Every forward-only filter is evaluated per candidate (measurement
    # switch in both builds); without it zvec first builds a full bitmap
    # for filters on columns without an inverted index.
    single = {"ZVEC_FORWARD_SINGLE_STAGE": "1"}
    return {
        # PR 913's build: Gandiva single-row JIT (same Gandiva patch as PR 889).
        "jit": (args.jit_binary, dict(single, ZVEC_SCALAR_JIT="1")),
        # The same binary with the JIT disabled: Arrow per-row evaluation.
        "arrow": (args.jit_binary, dict(single, ZVEC_SCALAR_JIT="0")),
        # zvec on the same base with stile evaluating forward predicates.
        "stile": (args.stile_binary, dict(single)),
        # zvec's default routing (bitmap first for non-indexed columns).
        "default": (args.stile_binary, {}),
    }


def run_one(binary, env, flags, out_path):
    cmd = [binary] + [str(x) for pair in flags.items() for x in pair]
    full_env = dict(os.environ, **env)
    started = time.time()
    proc = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"run failed: {' '.join(cmd)}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    result["wall_seconds"] = time.time() - started
    result["env"] = env
    result.pop("raw_api_ms", None)
    result.pop("raw_request_ms", None)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jit-binary", required=True)
    p.add_argument("--stile-binary", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predicates", default="time,and,range,or_like,not_in")
    p.add_argument("--variants", default="jit,arrow,stile")
    p.add_argument("--clients", default="1,8")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--count", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--unfiltered", action="store_true")
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=False)
    table = variants(args)
    names = args.variants.split(",")
    results = []
    cells = [(pred, int(c)) for pred in args.predicates.split(",")
             for c in args.clients.split(",")]
    for rep in range(args.reps):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for pred, clients in cells:
            for name in order:
                binary, env = table[name]
                flags = {"--mode": "run", "--path": args.collection,
                         "--predicate": pred, "--clients": clients,
                         "--count": args.count, "--warmup": args.warmup,
                         "--workers": args.workers, "--variant": name}
                out = os.path.join(args.output,
                                   f"{pred}_{name}_c{clients}_r{rep}.json")
                r = run_one(binary, env, flags, out)
                results.append((pred, name, clients, r))
                print(f"rep{rep} {pred:8s} {name:6s} c{clients} "
                      f"qps={r['qps']:.1f} p50={r['p50_api_ms']:.3f}ms",
                      flush=True)
        if args.unfiltered:
            for clients in sorted({c for _, c in cells}):
                binary, env = table["stile"]
                flags = {"--mode": "run", "--path": args.collection,
                         "--strategy": "none", "--clients": clients,
                         "--count": args.count, "--warmup": args.warmup,
                         "--workers": args.workers, "--variant": "none"}
                out = os.path.join(args.output, f"none_c{clients}_r{rep}.json")
                r = run_one(binary, env, flags, out)
                results.append(("none", "none", clients, r))
                print(f"rep{rep} unfiltered c{clients} qps={r['qps']:.1f}",
                      flush=True)
    summary = {}
    for pred, name, clients, r in results:
        summary.setdefault((pred, clients, name), []).append(r)
    lines = ["predicate,clients,variant,median_qps,median_p50_ms,runs"]
    for (pred, clients, name), rs in sorted(summary.items()):
        lines.append(",".join([
            pred, str(clients), name,
            f"{statistics.median(r['qps'] for r in rs):.1f}",
            f"{statistics.median(r['p50_api_ms'] for r in rs):.3f}",
            str(len(rs))]))
    with open(os.path.join(args.output, "summary.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
