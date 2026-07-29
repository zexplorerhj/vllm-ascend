# H20 FP8 Linear and GroupGemm Benchmark Design

## Goal

Add H20-native FP8 W8A8 Linear and GroupGemm curves to
`tests/ops` without changing the OperatorTestFramework V2 timing or fresh
storage protocol.

## Scope

This phase covers only:

- NVIDIA H20 (SM90).
- Linear FP8 W8A8 E4M3 with BF16 output.
- GroupGemm FP8 W8A8 E4M3 with BF16 output.
- Correctness smoke tests, provider diagnostics, full Framework V2 CSVs,
  and PNG curves.

MXFP8 and Ascend 950PR are explicitly deferred.

## Quantization Semantics

Both operators benchmark the quantized GEMM core. Quantization is performed
while preparing each fresh payload and is excluded from Device Event timing.

- Activations: E4M3, one FP32 dequantization scale per token/row.
- Weights: E4M3, one FP32 dequantization scale per output channel.
- Accumulation/output: BF16.
- Bias: disabled.
- FLOP accounting: `2 * M * N * K`.

The CPU source tensors are FP32. A symmetric max-absolute-value quantizer
produces E4M3 values and dequantization scales. Zero rows/channels use a scale
of one to avoid division by zero.

## Providers

### Linear

Use the low-level vLLM CUTLASS operator:

```python
torch.ops._C.cutlass_scaled_mm(
    output, activation_fp8, weight_fp8_kn,
    activation_scale_m1, weight_scale_1n, None
)
```

The low-level entry is selected because it accepts a caller-owned output
tensor. The public `vllm._custom_ops.cutlass_scaled_mm` wrapper allocates its
output and therefore does not satisfy the strict preallocated-output contract.

### GroupGemm

H20 runtime validation changed the provider policy.  The grouped CUTLASS
kernel advertises SM90 support but fails internally at the first formal point
(`M=64`, `E=8`, eight rows per expert).  To keep one implementation across
all ten points, the formal provider is therefore a loop over the low-level
caller-output `cutlass_scaled_mm` entry:

```python
for expert in range(num_experts):
    torch.ops._C.cutlass_scaled_mm(
        expert_output, expert_activation_fp8, expert_weight_fp8_kn,
        expert_activation_scale, expert_weight_scale, None
    )
```

The input is already sorted into contiguous expert ranges, so no routing
permutation is timed. All metadata is constructed during payload preparation.
The implementation executes eight expert launches into disjoint views of one
preallocated `[M, N]` BF16 output tensor.

The vLLM CUTLASS grouped GEMM remains a diagnostic-only provider for shapes
where every expert has at least 16 rows and is 16-row aligned. It does not
silently mix providers point-by-point in the formal curve.

## Framework Integration

- Add a unique `PrecisionType.FP8` value.
- Add focused FP8 operator classes rather than changing the behavior of the
  existing FP16/BF16/INT8 classes.
- Reuse `run_core_operator_performance_test_v2` unchanged.
- Each repeat prepares `warmup + iterations` independent activation, weight,
  scale, metadata, and output buffers.
- Both providers declare the strict preallocated-output contract.
- Device Events enclose only repeated core operator calls.
- Existing stabilization-repeat and median aggregation behavior remains
  unchanged.

## Curve Shapes

Linear reuses the current formal matrix:

- `M=N=K`: 256 through 4096 inclusive, step 128.
- Warmup: 10.
- Requested measured iterations: existing adaptive fresh-storage plan with
  base 50, using FP8-specific storage bytes.
- Measured repeats: 3, plus the framework's formal stabilization repeats.

GroupGemm reuses the current formal matrix:

- Experts: 8.
- `K=7168`, `N=4096`.
- Total tokens `M`: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
  32768.
- Warmup: 10.
- Measured iterations: 30.
- Measured repeats: 3, plus the framework's formal stabilization repeats.

## Correctness

Before the full curves:

1. Verify runtime capability probes on H20.
2. Run small and representative shapes.
3. Compare CUTLASS output with a dequantized FP32 matrix-multiplication
   reference converted to BF16.
4. Check finite output, shape, provider name, scale layout, and output alias.
5. Reject unsupported K/N alignment rather than adding hidden padding.

## Results

The H20 result directory contains:

- Linear FP8 CSV and PNG.
- GroupGemm FP8 CSV and PNG.
- Provider-diagnostic CSV.
- A manifest containing host, GPU, driver, CUDA, Torch, vLLM commit/version,
  provider, quantization semantics, W/I/R counts, and Framework V2 protocol.

The user reviews these H20 results before any 950PR work begins.
