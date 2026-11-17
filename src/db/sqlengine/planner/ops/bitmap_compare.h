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

#pragma once

#include <cstdint>
#include <memory>
#include <arrow/compute/function.h>

namespace zvec::sqlengine::bitmap_internal {

enum class Int32BitmapIsa { kScalar, kAvx2, kAvx512 };

// Input/output buffers must not overlap. The output slice can start at any
// bit offset; bits outside it are preserved. Values need no SIMD alignment.
using PackInt32Greater = void (*)(const int32_t *, int32_t, int64_t, uint8_t *,
                                  int64_t);

struct Int32BitmapKernel {
  Int32BitmapIsa isa;
  PackInt32Greater pack;
};

// Select among compiled kernels. Callers must supply OS-usable capabilities,
// not just hardware CPUID bits. Also used to exercise lower ISAs in tests.
Int32BitmapKernel select_int32_bitmap_kernel(bool avx2, bool avx512);

// Cached on first use, using ailego CPU features and OS vector-state support.
const Int32BitmapKernel &int32_bitmap_kernel();
const char *int32_bitmap_isa_name(Int32BitmapIsa isa);

// An owned, stateless Arrow function; never replaces the global "greater".
arrow::Result<std::shared_ptr<arrow::compute::ScalarFunction>>
make_int32_bitmap_function(const Int32BitmapKernel &kernel);

}  // namespace zvec::sqlengine::bitmap_internal
