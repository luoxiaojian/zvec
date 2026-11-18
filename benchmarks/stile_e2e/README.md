# stile vs Gandiva JIT vs Arrow: end-to-end filtered HNSW

Measurement only, not for merging. Adapted from PR #889's Linux harness
(commit e8b65783): selectable predicates, each validated per returned row,
and no PR 889-only cache statistics.

| `--predicate` | filter | matching rows |
|---|---|---|
| `time` | `dataTime > 536870911` | 75% |
| `and` | `dataTime > t and pid < 7500000` | 56% |
| `range` | `dataTime > 268435455 and dataTime < 1879048192` | 75% |
| `or_like` | `(dataTime > t and pid < 5000000) or mail like '%7@%'` | 60% |
| `not_in` | `dataTime > t and pid not in (100 values)` | 75% |

## Variants

Both branches start from `feat/forward-filter-read-view` (22fa6bb1) and
carry the same `ZVEC_FORWARD_SINGLE_STAGE` switch and this directory.

| variant | branch | environment |
|---|---|---|
| `stile` | `feat/stile-forward-predicate` | `ZVEC_FORWARD_SINGLE_STAGE=1` |
| `jit` | `bench/jit-e2e` (PR 913, the same Gandiva patch as PR 889) | `ZVEC_FORWARD_SINGLE_STAGE=1 ZVEC_SCALAR_JIT=1` |
| `arrow` | `bench/jit-e2e` | `ZVEC_FORWARD_SINGLE_STAGE=1 ZVEC_SCALAR_JIT=0` |

`ZVEC_FORWARD_SINGLE_STAGE=1` evaluates every forward-only filter per
candidate. Without it zvec does that only when every condition is on an
inverted-indexed column that the optimizer moves to forward evaluation;
filters on other columns first build a full bitmap, and LIKE, IN, = and !=
are never moved.

## Linux

```sh
# stile variant (stile is a submodule; its own Arrow submodule is skipped)
git clone -b feat/stile-forward-predicate https://github.com/luoxiaojian/zvec zvec-stile
cd zvec-stile && git submodule update --init --recursive && cd ..
cmake -S zvec-stile -B build-stile -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PROJECT_zvec_INCLUDE=$PWD/zvec-stile/benchmarks/stile_e2e/hook.cmake
cmake --build build-stile --target zvec_e2e_bench

# JIT and Arrow variants (needs LLVM 20 and OpenSSL, as in PR 889)
git clone -b bench/jit-e2e https://github.com/luoxiaojian/zvec zvec-jit
cd zvec-jit && git submodule update --init --recursive && cd ..
cmake -S zvec-jit -B build-jit -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DZVEC_ENABLE_SCALAR_JIT=ON -DLLVM_ROOT=<llvm-20> \
  -DCMAKE_PROJECT_zvec_INCLUDE=$PWD/zvec-jit/benchmarks/stile_e2e/hook.cmake
cmake --build build-jit --target zvec_e2e_bench

# Data (35 minutes on 12 Apple M6 cores), then the matrix
build-stile/bin/zvec_e2e_bench --mode build --path c10m --rows 10000000 \
  --build-workers 16 --output build_manifest.json
python3 zvec-stile/benchmarks/stile_e2e/run_e2e.py \
  --jit-binary build-jit/bin/zvec_e2e_bench \
  --stile-binary build-stile/bin/zvec_e2e_bench \
  --collection c10m --output results-linux --unfiltered
```

`run_e2e.py` runs each cell in a fresh process, rotates the variant order
over `--reps` (3) repetitions, and writes per-run JSON plus `summary.csv`
(median QPS and median p50 API latency). Add `--debug 1` to a single
`--mode run` to see which path each segment takes: `Forward predicate
typed` (stile) or `Scalar JIT forward filter ready` (JIT).

`results/summary_m6.csv`: Apple M6 (2+4+6 cores), macOS 27, 8 engine
workers, 4096 timed and 1024 warmup queries per client; stile f3a6b6d.
