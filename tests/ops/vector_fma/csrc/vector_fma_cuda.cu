#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int64_t kSupportedAccumulator4 = 4;
constexpr int64_t kSupportedAccumulator8 = 8;
constexpr int64_t kSupportedAccumulator16 = 16;

__device__ __forceinline__ void fma_f16_scalar(
    uint16_t& accumulator,
    const uint16_t value_a,
    const uint16_t value_b) {
  asm volatile(
      "fma.rn.f16 %0, %0, %1, %2;"
      : "+h"(accumulator)
      : "h"(value_a), "h"(value_b));
}

__device__ __forceinline__ void fma_bf16_scalar(
    uint16_t& accumulator,
    const uint16_t value_a,
    const uint16_t value_b) {
  asm volatile(
      "fma.rn.bf16 %0, %0, %1, %2;"
      : "+h"(accumulator)
      : "h"(value_a), "h"(value_b));
}

template <bool kBf16>
__device__ __forceinline__ void fma_x2(
    uint32_t& accumulator,
    const uint32_t value_a,
    const uint32_t value_b) {
  uint32_t result;
  if constexpr (kBf16) {
    asm volatile(
        "fma.rn.bf16x2 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(accumulator), "r"(value_a), "r"(value_b));
  } else {
    asm volatile(
        "fma.rn.f16x2 %0, %1, %2, %3;"
        : "=r"(result)
        : "r"(accumulator), "r"(value_a), "r"(value_b));
  }
  accumulator = result;
}

#define DECLARE_CHAINS(T) \
  T a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15; \
  T b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15; \
  T acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11, \
      acc12, acc13, acc14, acc15

#define LOAD_CHAIN(INDEX, STRIDE, LANE) \
  if constexpr (kAccumulators > INDEX) { \
    const int64_t offset = static_cast<int64_t>(INDEX) * (STRIDE) + (LANE); \
    a##INDEX = input_a[offset]; \
    b##INDEX = input_b[offset]; \
    acc##INDEX = output[offset]; \
  }

#define APPLY_SCALAR_CHAIN(INDEX, FUNCTION) \
  if constexpr (kAccumulators > INDEX) { \
    FUNCTION(acc##INDEX, a##INDEX, b##INDEX); \
  }

#define APPLY_X2_CHAIN(INDEX, BF16) \
  if constexpr (kAccumulators > INDEX) { \
    fma_x2<BF16>(acc##INDEX, a##INDEX, b##INDEX); \
  }

#define STORE_CHAIN(INDEX, STRIDE, LANE) \
  if constexpr (kAccumulators > INDEX) { \
    const int64_t offset = static_cast<int64_t>(INDEX) * (STRIDE) + (LANE); \
    output[offset] = acc##INDEX; \
  }

#define LOAD_ALL_CHAINS(STRIDE, LANE) \
  LOAD_CHAIN(0, STRIDE, LANE)  \
  LOAD_CHAIN(1, STRIDE, LANE)  \
  LOAD_CHAIN(2, STRIDE, LANE)  \
  LOAD_CHAIN(3, STRIDE, LANE)  \
  LOAD_CHAIN(4, STRIDE, LANE)  \
  LOAD_CHAIN(5, STRIDE, LANE)  \
  LOAD_CHAIN(6, STRIDE, LANE)  \
  LOAD_CHAIN(7, STRIDE, LANE)  \
  LOAD_CHAIN(8, STRIDE, LANE)  \
  LOAD_CHAIN(9, STRIDE, LANE)  \
  LOAD_CHAIN(10, STRIDE, LANE) \
  LOAD_CHAIN(11, STRIDE, LANE) \
  LOAD_CHAIN(12, STRIDE, LANE) \
  LOAD_CHAIN(13, STRIDE, LANE) \
  LOAD_CHAIN(14, STRIDE, LANE) \
  LOAD_CHAIN(15, STRIDE, LANE)

#define STORE_ALL_CHAINS(STRIDE, LANE) \
  STORE_CHAIN(0, STRIDE, LANE)  \
  STORE_CHAIN(1, STRIDE, LANE)  \
  STORE_CHAIN(2, STRIDE, LANE)  \
  STORE_CHAIN(3, STRIDE, LANE)  \
  STORE_CHAIN(4, STRIDE, LANE)  \
  STORE_CHAIN(5, STRIDE, LANE)  \
  STORE_CHAIN(6, STRIDE, LANE)  \
  STORE_CHAIN(7, STRIDE, LANE)  \
  STORE_CHAIN(8, STRIDE, LANE)  \
  STORE_CHAIN(9, STRIDE, LANE)  \
  STORE_CHAIN(10, STRIDE, LANE) \
  STORE_CHAIN(11, STRIDE, LANE) \
  STORE_CHAIN(12, STRIDE, LANE) \
  STORE_CHAIN(13, STRIDE, LANE) \
  STORE_CHAIN(14, STRIDE, LANE) \
  STORE_CHAIN(15, STRIDE, LANE)

#define APPLY_ALL_SCALAR_CHAINS(FUNCTION) \
  APPLY_SCALAR_CHAIN(0, FUNCTION)  \
  APPLY_SCALAR_CHAIN(1, FUNCTION)  \
  APPLY_SCALAR_CHAIN(2, FUNCTION)  \
  APPLY_SCALAR_CHAIN(3, FUNCTION)  \
  APPLY_SCALAR_CHAIN(4, FUNCTION)  \
  APPLY_SCALAR_CHAIN(5, FUNCTION)  \
  APPLY_SCALAR_CHAIN(6, FUNCTION)  \
  APPLY_SCALAR_CHAIN(7, FUNCTION)  \
  APPLY_SCALAR_CHAIN(8, FUNCTION)  \
  APPLY_SCALAR_CHAIN(9, FUNCTION)  \
  APPLY_SCALAR_CHAIN(10, FUNCTION) \
  APPLY_SCALAR_CHAIN(11, FUNCTION) \
  APPLY_SCALAR_CHAIN(12, FUNCTION) \
  APPLY_SCALAR_CHAIN(13, FUNCTION) \
  APPLY_SCALAR_CHAIN(14, FUNCTION) \
  APPLY_SCALAR_CHAIN(15, FUNCTION)

#define APPLY_ALL_X2_CHAINS(BF16) \
  APPLY_X2_CHAIN(0, BF16)  \
  APPLY_X2_CHAIN(1, BF16)  \
  APPLY_X2_CHAIN(2, BF16)  \
  APPLY_X2_CHAIN(3, BF16)  \
  APPLY_X2_CHAIN(4, BF16)  \
  APPLY_X2_CHAIN(5, BF16)  \
  APPLY_X2_CHAIN(6, BF16)  \
  APPLY_X2_CHAIN(7, BF16)  \
  APPLY_X2_CHAIN(8, BF16)  \
  APPLY_X2_CHAIN(9, BF16)  \
  APPLY_X2_CHAIN(10, BF16) \
  APPLY_X2_CHAIN(11, BF16) \
  APPLY_X2_CHAIN(12, BF16) \
  APPLY_X2_CHAIN(13, BF16) \
  APPLY_X2_CHAIN(14, BF16) \
  APPLY_X2_CHAIN(15, BF16)

template <bool kBf16, int kAccumulators>
__global__ void vector_fma_scalar_kernel(
    const uint16_t* input_a,
    const uint16_t* input_b,
    uint16_t* output,
    const int64_t elements,
    const int64_t fma_depth) {
  const int64_t lane =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lane >= elements) {
    return;
  }

  DECLARE_CHAINS(uint16_t);
  LOAD_ALL_CHAINS(elements, lane);

  for (int64_t iteration = 0; iteration < fma_depth; ++iteration) {
    if constexpr (kBf16) {
      APPLY_ALL_SCALAR_CHAINS(fma_bf16_scalar);
    } else {
      APPLY_ALL_SCALAR_CHAINS(fma_f16_scalar);
    }
  }

  STORE_ALL_CHAINS(elements, lane);
}

template <bool kBf16, int kAccumulators>
__global__ void vector_fma_x2_kernel(
    const uint32_t* input_a,
    const uint32_t* input_b,
    uint32_t* output,
    const int64_t packed_elements,
    const int64_t fma_depth) {
  const int64_t packed_lane =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (packed_lane >= packed_elements) {
    return;
  }

  DECLARE_CHAINS(uint32_t);
  LOAD_ALL_CHAINS(packed_elements, packed_lane);

  for (int64_t iteration = 0; iteration < fma_depth; ++iteration) {
    APPLY_ALL_X2_CHAINS(kBf16);
  }

  STORE_ALL_CHAINS(packed_elements, packed_lane);
}

void check_common(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    const torch::Tensor& output,
    const int64_t elements,
    const int64_t fma_depth,
    const int64_t accumulators,
    const int64_t block_size,
    const int64_t num_programs,
    const at::ScalarType expected_dtype,
    const int64_t instruction_lane_width) {
  TORCH_CHECK(input_a.is_cuda(), "input_a must be a CUDA tensor");
  TORCH_CHECK(input_b.is_cuda(), "input_b must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(
      input_a.get_device() == input_b.get_device() &&
          input_a.get_device() == output.get_device(),
      "all tensors must reside on the same CUDA device");
  TORCH_CHECK(
      input_a.scalar_type() == expected_dtype &&
          input_b.scalar_type() == expected_dtype &&
          output.scalar_type() == expected_dtype,
      "all tensors must have the provider's exact dtype");
  TORCH_CHECK(
      input_a.is_contiguous() && input_b.is_contiguous() &&
          output.is_contiguous(),
      "all tensors must be contiguous");
  TORCH_CHECK(elements > 0, "elements must be positive");
  TORCH_CHECK(fma_depth > 0, "fma_depth must be positive");
  TORCH_CHECK(
      accumulators == kSupportedAccumulator4 ||
          accumulators == kSupportedAccumulator8 ||
          accumulators == kSupportedAccumulator16,
      "accumulators must be 4, 8, or 16");
  TORCH_CHECK(
      block_size > 0 && block_size <= 1024,
      "block_size must be in [1, 1024]");
  TORCH_CHECK(num_programs > 0, "num_programs must be positive");
  TORCH_CHECK(
      block_size * num_programs * instruction_lane_width == elements,
      "launch threads and instruction lane width must exactly cover elements");
  TORCH_CHECK(
      elements % instruction_lane_width == 0,
      "elements must be divisible by the instruction lane width");
  const int64_t expected_numel = elements * accumulators;
  TORCH_CHECK(
      input_a.numel() == expected_numel &&
          input_b.numel() == expected_numel &&
          output.numel() == expected_numel,
      "tensor numel must equal elements * accumulators");
}

template <bool kBf16, int kAccumulators>
void launch_scalar_typed(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    torch::Tensor& output,
    const int64_t elements,
    const int64_t fma_depth,
  const int64_t block_size,
  const int64_t num_programs) {
  const c10::cuda::CUDAGuard device_guard(input_a.device());
  const auto* input_a_ptr =
      reinterpret_cast<const uint16_t*>(input_a.data_ptr());
  const auto* input_b_ptr =
      reinterpret_cast<const uint16_t*>(input_b.data_ptr());
  auto* output_ptr = reinterpret_cast<uint16_t*>(output.data_ptr());
  const auto stream = at::cuda::getCurrentCUDAStream(input_a.get_device());
  vector_fma_scalar_kernel<kBf16, kAccumulators>
      <<<num_programs, block_size, 0, stream.stream()>>>(
          input_a_ptr,
          input_b_ptr,
          output_ptr,
          elements,
          fma_depth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kBf16, int kAccumulators>
void launch_x2_typed(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    torch::Tensor& output,
    const int64_t elements,
    const int64_t fma_depth,
    const int64_t block_size,
    const int64_t num_programs) {
  const c10::cuda::CUDAGuard device_guard(input_a.device());
  const auto* input_a_ptr =
      reinterpret_cast<const uint32_t*>(input_a.data_ptr());
  const auto* input_b_ptr =
      reinterpret_cast<const uint32_t*>(input_b.data_ptr());
  auto* output_ptr = reinterpret_cast<uint32_t*>(output.data_ptr());
  const auto stream = at::cuda::getCurrentCUDAStream(input_a.get_device());
  vector_fma_x2_kernel<kBf16, kAccumulators>
      <<<num_programs, block_size, 0, stream.stream()>>>(
          input_a_ptr,
          input_b_ptr,
          output_ptr,
          elements / 2,
          fma_depth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Launch4, typename Launch8, typename Launch16>
void dispatch_accumulators(
    const int64_t accumulators,
    Launch4&& launch4,
    Launch8&& launch8,
    Launch16&& launch16) {
  switch (accumulators) {
    case kSupportedAccumulator4:
      launch4();
      return;
    case kSupportedAccumulator8:
      launch8();
      return;
    case kSupportedAccumulator16:
      launch16();
      return;
    default:
      TORCH_CHECK(false, "accumulators must be 4, 8, or 16");
  }
}

void launch_f16_scalar(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    torch::Tensor output,
    const int64_t elements,
    const int64_t fma_depth,
    const int64_t accumulators,
    const int64_t block_size,
    const int64_t num_programs) {
  check_common(
      input_a,
      input_b,
      output,
      elements,
      fma_depth,
      accumulators,
      block_size,
      num_programs,
      at::kHalf,
      1);
  dispatch_accumulators(
      accumulators,
      [&] {
        launch_scalar_typed<false, 4>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      },
      [&] {
        launch_scalar_typed<false, 8>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      },
      [&] {
        launch_scalar_typed<false, 16>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      });
}

void launch_bf16_scalar(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    torch::Tensor output,
    const int64_t elements,
    const int64_t fma_depth,
    const int64_t accumulators,
    const int64_t block_size,
    const int64_t num_programs) {
  check_common(
      input_a,
      input_b,
      output,
      elements,
      fma_depth,
      accumulators,
      block_size,
      num_programs,
      at::kBFloat16,
      1);
  dispatch_accumulators(
      accumulators,
      [&] {
        launch_scalar_typed<true, 4>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      },
      [&] {
        launch_scalar_typed<true, 8>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      },
      [&] {
        launch_scalar_typed<true, 16>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      });
}

template <bool kBf16>
void launch_x2(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    torch::Tensor output,
    const int64_t elements,
    const int64_t fma_depth,
    const int64_t accumulators,
    const int64_t block_size,
    const int64_t num_programs,
    const at::ScalarType dtype) {
  check_common(
      input_a,
      input_b,
      output,
      elements,
      fma_depth,
      accumulators,
      block_size,
      num_programs,
      dtype,
      2);
  dispatch_accumulators(
      accumulators,
      [&] {
        launch_x2_typed<kBf16, 4>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      },
      [&] {
        launch_x2_typed<kBf16, 8>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      },
      [&] {
        launch_x2_typed<kBf16, 16>(
            input_a, input_b, output, elements, fma_depth, block_size, num_programs);
      });
}

void launch_f16x2(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    torch::Tensor output,
    const int64_t elements,
    const int64_t fma_depth,
    const int64_t accumulators,
    const int64_t block_size,
    const int64_t num_programs) {
  launch_x2<false>(
      input_a,
      input_b,
      output,
      elements,
      fma_depth,
      accumulators,
      block_size,
      num_programs,
      at::kHalf);
}

void launch_bf16x2(
    const torch::Tensor& input_a,
    const torch::Tensor& input_b,
    torch::Tensor output,
    const int64_t elements,
    const int64_t fma_depth,
    const int64_t accumulators,
    const int64_t block_size,
    const int64_t num_programs) {
  launch_x2<true>(
      input_a,
      input_b,
      output,
      elements,
      fma_depth,
      accumulators,
      block_size,
      num_programs,
      at::kBFloat16);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "launch_f16_scalar",
      &launch_f16_scalar,
      "Launch one FP16 scalar PTX Vector FMA kernel");
  module.def(
      "launch_f16x2",
      &launch_f16x2,
      "Launch one packed FP16x2 PTX Vector FMA kernel");
  module.def(
      "launch_bf16_scalar",
      &launch_bf16_scalar,
      "Launch one BF16 scalar PTX Vector FMA kernel");
  module.def(
      "launch_bf16x2",
      &launch_bf16x2,
      "Launch one packed BF16x2 PTX Vector FMA kernel");
}
