// Copyright 2026-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "db/sqlengine/planner/ops/bitmap_compare.h"
#include <cstring>
#include <arrow/compute/exec.h>
#include <arrow/scalar.h>
#include <arrow/util/bit_util.h>
#include "ailego/internal/cpu_features.h"

// Keep the translation unit at the baseline ISA. Other architectures and
// compilers retain Arrow's generic comparison implementation.
#if !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#define ZVEC_BITMAP_TARGETS 1
#include <cpuid.h>
#include <immintrin.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

namespace zvec::sqlengine::bitmap_internal {
namespace {

void pack_scalar(const int32_t *values, int32_t threshold, int64_t length,
                 uint8_t *output, int64_t offset) {
  for (int64_t i = 0; i < length; ++i) {
    arrow::bit_util::SetBitTo(output, offset + i, values[i] > threshold);
  }
}

#if defined(ZVEC_BITMAP_TARGETS)
__attribute__((target("avx2"), noinline)) void pack_avx2(
    const int32_t *__restrict__ values, int32_t threshold, int64_t length,
    uint8_t *__restrict__ output, int64_t offset) {
  if (length == 0) return;
  int64_t i = 0;
  while (i < length && ((offset + i) & 7)) {
    arrow::bit_util::SetBitTo(output, offset + i, values[i] > threshold);
    ++i;
  }
  const auto threshold_vector = _mm256_set1_epi32(threshold);
  auto *bytes = output + ((static_cast<uint64_t>(offset) + i) >> 3);
  const auto *ptr = values + i;
  for (; i + 32 <= length; i += 32, ptr += 32, bytes += 4) {
    const auto a = _mm256_cmpgt_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)),
        threshold_vector);
    const auto b = _mm256_cmpgt_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + 8)),
        threshold_vector);
    const auto c = _mm256_cmpgt_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + 16)),
        threshold_vector);
    const auto d = _mm256_cmpgt_epi32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + 24)),
        threshold_vector);
    const uint32_t bits =
        static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(a))) |
        (static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(b)))
         << 8) |
        (static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(c)))
         << 16) |
        (static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(d)))
         << 24);
    std::memcpy(bytes, &bits, sizeof(bits));
  }
  for (; i + 8 <= length; i += 8, ptr += 8, ++bytes) {
    const auto values_vector =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    *bytes = static_cast<uint8_t>(_mm256_movemask_ps(_mm256_castsi256_ps(
        _mm256_cmpgt_epi32(values_vector, threshold_vector))));
  }
  for (; i < length; ++i) {
    arrow::bit_util::SetBitTo(output, offset + i, values[i] > threshold);
  }
}

__attribute__((target("avx512f"), noinline)) void pack_avx512(
    const int32_t *__restrict__ values, int32_t threshold, int64_t length,
    uint8_t *__restrict__ output, int64_t offset) {
  if (length == 0) return;
  int64_t i = 0;
  while (i < length && ((offset + i) & 7)) {
    arrow::bit_util::SetBitTo(output, offset + i, values[i] > threshold);
    ++i;
  }
  const auto threshold_vector = _mm512_set1_epi32(threshold);
  auto *bytes = output + ((static_cast<uint64_t>(offset) + i) >> 3);
  const auto *ptr = values + i;
  for (; i + 64 <= length; i += 64, ptr += 64, bytes += 8) {
    const uint64_t a =
        _mm512_cmpgt_epi32_mask(_mm512_loadu_si512(ptr), threshold_vector);
    const uint64_t b =
        _mm512_cmpgt_epi32_mask(_mm512_loadu_si512(ptr + 16), threshold_vector);
    const uint64_t c =
        _mm512_cmpgt_epi32_mask(_mm512_loadu_si512(ptr + 32), threshold_vector);
    const uint64_t d =
        _mm512_cmpgt_epi32_mask(_mm512_loadu_si512(ptr + 48), threshold_vector);
    const uint64_t bits = a | (b << 16) | (c << 32) | (d << 48);
    std::memcpy(bytes, &bits, sizeof(bits));
  }
  for (; i + 16 <= length; i += 16, ptr += 16, bytes += 2) {
    const uint16_t bits =
        _mm512_cmpgt_epi32_mask(_mm512_loadu_si512(ptr), threshold_vector);
    std::memcpy(bytes, &bits, sizeof(bits));
  }
  for (; i < length; ++i) {
    arrow::bit_util::SetBitTo(output, offset + i, values[i] > threshold);
  }
}
#endif

Int32BitmapKernel detect_kernel() {
  bool avx2 = false, avx512 = false;
#if defined(ZVEC_BITMAP_TARGETS)
  const auto &features = ailego::internal::CpuFeatures::static_flags_;
  uint32_t eax, ebx, ecx, edx;
  uint64_t xcr0 = 0;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) &&
      (ecx & (3u << 26)) == (3u << 26)) {
    // XGETBV requires both XSAVE and OSXSAVE. CPUID alone is insufficient.
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    xcr0 = (static_cast<uint64_t>(edx) << 32) | eax;
  }
  avx2 = features.AVX && features.AVX2 && (xcr0 & 0x6u) == 0x6u;
  bool avx512_state = (xcr0 & 0xe6u) == 0xe6u;
#if defined(__APPLE__)
  // Darwin may enable opmask/ZMM lazily, as in turbo's batch_topk dispatch.
  int enabled = 0;
  size_t bytes = sizeof(enabled);
  avx512_state = avx512_state || (sysctlbyname("hw.optional.avx512f", &enabled,
                                               &bytes, nullptr, 0) == 0 &&
                                  bytes == sizeof(enabled) && enabled != 0);
#endif
  avx512 = avx2 && features.AVX512F && avx512_state;
#endif
  return select_int32_bitmap_kernel(avx2, avx512);
}

struct CompareData : arrow::compute::KernelState {
  explicit CompareData(PackInt32Greater function) : pack(function) {}
  PackInt32Greater pack;
};

arrow::Status compare(arrow::compute::KernelContext *ctx,
                      const arrow::compute::ExecSpan &batch,
                      arrow::compute::ExecResult *result) {
  auto *output = result->array_span_mutable();
  auto *bits = output->buffers[1].data;
  const auto &left = batch[0];
  const auto &right = batch[1];
  if (left.is_array() && right.is_scalar()) {
    const auto *data =
        static_cast<const CompareData *>(ctx->kernel()->data.get());
    data->pack(left.array.GetValues<int32_t>(1),
               right.scalar_as<arrow::Int32Scalar>().value, batch.length, bits,
               output->offset);
  } else {
    // Binding selects array > literal. Keep all argument shapes correct for
    // Arrow's scalar promotion and direct function invocation as well.
    for (int64_t i = 0; i < batch.length; ++i) {
      const int32_t l = left.is_array()
                            ? left.array.GetValues<int32_t>(1)[i]
                            : left.scalar_as<arrow::Int32Scalar>().value;
      const int32_t r = right.is_array()
                            ? right.array.GetValues<int32_t>(1)[i]
                            : right.scalar_as<arrow::Int32Scalar>().value;
      arrow::bit_util::SetBitTo(bits, output->offset + i, l > r);
    }
  }
  return arrow::Status::OK();
}
}  // namespace

Int32BitmapKernel select_int32_bitmap_kernel(bool avx2, bool avx512) {
#if defined(ZVEC_BITMAP_TARGETS)
  if (avx2 && avx512) return {Int32BitmapIsa::kAvx512, pack_avx512};
  if (avx2) return {Int32BitmapIsa::kAvx2, pack_avx2};
#else
  (void)avx2;
  (void)avx512;
#endif
  return {Int32BitmapIsa::kScalar, pack_scalar};
}

const Int32BitmapKernel &int32_bitmap_kernel() {
  static const auto kernel = detect_kernel();
  return kernel;
}

const char *int32_bitmap_isa_name(Int32BitmapIsa isa) {
  switch (isa) {
    case Int32BitmapIsa::kAvx512:
      return "avx512";
    case Int32BitmapIsa::kAvx2:
      return "avx2";
    default:
      return "scalar";
  }
}

arrow::Result<std::shared_ptr<arrow::compute::ScalarFunction>>
make_int32_bitmap_function(const Int32BitmapKernel &selected) {
  namespace cp = arrow::compute;
  auto function = std::make_shared<cp::ScalarFunction>(
      "greater", cp::Arity::Binary(),
      cp::FunctionDoc{"Compare signed int32 values",
                      "Write a predicate bitmap",
                      {"left", "right"}});
  cp::ScalarKernel kernel({arrow::int32(), arrow::int32()}, arrow::boolean(),
                          compare);
  kernel.data = std::make_shared<CompareData>(selected.pack);
  // Default validity intersection preserves Arrow's three-valued semantics.
  // Supporting sliced output lets the executor allocate one bitmap even when
  // its input chunks have arbitrary, non-byte-aligned row counts.
  ARROW_RETURN_NOT_OK(function->AddKernel(std::move(kernel)));
  return function;
}

}  // namespace zvec::sqlengine::bitmap_internal
