# Turbo batch distance kernels

Turbo uses the existing pointer batch interface for one query against many
candidates. Rows may be independently allocated, reordered, or padded with
quantizer metadata. IVF and Flat gather bounded pointer arrays (at most 512
rows); FlatStreamer never crosses a physical storage block. BatchTopK, filtering,
key validity and deferred result-key mapping remain independent of the kernel.
HNSW uses the same batch kernels for stored FP32 datapoint construction and
query distances, including graph traversal and brute-force fallback. Provider
inputs keep their typed input-distance operations. Other quantizers retain a
datapoint-to-datapoint pairwise fallback unless they implement native batching;
query and datapoint encodings must not be assumed interchangeable.

## Scope and evidence

This implementation keeps Turbo's existing pointer batch interface. It adds
no GEMV provider, BLAS dependency, strided-distance API or storage-format change.
It incorporates the bounded scan and top-k improvements developed in PR #838,
and the stored-datapoint batch integration from PR #839.

The NEON FP32 default shares query loads across four rows and uses two
accumulators per row. The later `gemv_bench` tuning experiment compared 30
layouts/prefetch variants and found no stable overall improvement over this
4x2 layout; PR #839's earlier Apple-specific prefetch kernel is not substituted
for that default. The evidence is under
`results/machines/mac-m5pro-01/neon-ip-tuning-2026-10-26/` in `gemv_bench`.
SME2 measurements are under `sme2-ip-2026-10-26/` in the same machine directory.
These are kernel experiments; end-to-end build and query results must be
measured separately. Performance crossover points are workload- and
machine-dependent.

## Implementations and dispatch

- NEON shares query loads across four FP32 rows or eight FP16 rows, with two
  FP32 accumulators per row. Inner product and squared L2 use this implementation;
  cosine delegates to inner product. FP16 widening is shared across rows.
- SME2 implements FP32 inner product and cosine, sharing query loads through
  16-row ZA accumulator tiles. It requires no packing or contiguous-row contract.
  Pairwise distances, squared L2 and other types retain existing backends.
- Automatic SME2 selection requires a successful compiler/linker probe and
  `hw.optional.arm.FEAT_SME2 == 1`. Missing OS capabilities or older toolchains
  retain NEON. `CpuArchType::kNEON` still selects NEON explicitly; an unavailable
  explicit `kSME2` request returns empty kernels, like other explicit ISAs.
- The selected batch wrapper uses NEON for fewer than 16 rows or dimensions
  below 256. Larger batches handle all tails within one SME2 streaming region;
  cosine's `+1` is fused into the streaming reduction. This is a conservative
  crossover informed by local rotating measurements, not a cache-aware or
  universally optimal tuning rule. The raw SME2 kernel still supports all tails
  and queries the actual streaming vector length.
- Only SME2 translation units receive `-march=armv8.6-a+sme2`. Public functions
  have the ordinary ABI; the implementation uses `__arm_locally_streaming` and
  `__arm_new("za")`. Do not replace this flag with `armv9-a+sme2`: Apple supports
  streaming SVE through SME, not ordinary SVE instructions in a function prologue.
- `-DZVEC_TURBO_ENABLE_SME2=OFF` builds the fallback without SME2 objects.

## Mac coverage

| Machines | Distance backend |
| --- | --- |
| M1/M2/M3 Macs | NEON |
| M4/M5 and other SME2-capable ARM64 Macs | Runtime-gated SME2 for FP32 IP/cosine batches with at least 16 rows and dimension at least 256; NEON otherwise |
| Intel Macs | Best compiled SSE2/AVX2/AVX512 family allowed by CPU features and OS register-state support |
| Other ARM64 targets (Linux, Android, iOS) | Existing NEON / native FP16 dispatch; this SME2 backend is enabled only on macOS |
| Other x86 targets | Same compiled CPU/OS-gated dispatch as Intel Macs |
| Unsupported SIMD targets or toolchains | Scalar fallback |

Dispatch uses OS capabilities, never chip names. Older macOS SDKs, a failed
SME2 compiler/linker probe, or a disabled SME2 build leave the ordinary NEON
path available even on new hardware. An ARMv9 architecture label alone does
not imply SME2 support. Ordinary SVE/SVE2 is not required by this implementation;
Apple's streaming SVE instructions are confined to the SME2 function body.

The crossover is tuned on M5 Pro, not benchmarked on every Mac generation.
Correct feature selection guarantees an executable compatible kernel, not
universal performance optimality on unmeasured hardware. FP32 squared L2 stays
on NEON; FP16 and quantized representations retain their own compiled backends.
An explicit distance-kernel ISA request is honored or returns empty kernels;
automatic mode falls back through supported implementations. The legacy PQ and
rotation factories retain their scalar fallback for unavailable explicit ISAs.

The x86 dispatcher checks the compiled kernel family, CPUID features and OS
YMM/ZMM register-state support. ISA flags are confined to their implementation
files, using explicit feature sets rather than an entire CPU model. On Intel
macOS, `hw.optional.avx512f` also recognizes Darwin's lazy AVX-512 state enablement;
usable AVX state is still mandatory. BatchTopK follows the same CPU/OS gating
with separately compiled AVX2/AVX-512 loops and a baseline dispatcher.

Primary references: [Arm device support and runtime detection](https://learn.arm.com/insights/sme/),
[Apple XNU SME execution rules](https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md),
[LLVM M5 feature definition](https://github.com/llvm/llvm-project/commit/f85494f6afeb52c18b0fd55fbe340b4b6df630bd),
[Clang streaming attributes](https://clang.llvm.org/docs/AttributeReference.html#arm-locally-streaming).

## Validation

`turbo_neon_batch_test` and `turbo_sme2_batch_test` compare against independent
FP64 references, including reordered/disjoint rows, unaligned SIMD addresses,
empty inputs and row/dimension tails. Guard pages detect overreads and output
overwrites. SME2 checks that SM/ZA state is restored before ordinary NEON code
runs. Public dispatch tests cover both unavailable hardware and a disabled build.

The scan regressions are `flat_quantizer_distance_test`,
`flat_streamer_batch_test`, `ivf_searcher_test` and `ivf_index_provider_test`.
`turbo_batch_distance_test` checks encoded metadata and pointer ordering across
quantizer types. Existing quantizer, CPU-feature and BatchTopK tests also apply.

`turbo_kernel_dispatch_test` exercises compiled-versus-runtime capabilities,
missing OS state, compiler-implied feature requirements, fallback ordering and
actual numerical output from every available backend. CPU tests include the
Intel macOS lazy-state case.
