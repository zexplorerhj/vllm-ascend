# Ascend 950PR FP8 and MXFP8 operator curves

## Scope

Add two semantically independent Ascend 950PR benchmark paths for both
Linear and pure GroupGemm while preserving the existing H20 FP8 providers.
All timing continues to use `OperatorTestFramework V2`.

The four 950PR curve entries are:

- Linear plain FP8
- Linear MXFP8
- GroupGemm plain FP8
- GroupGemm MXFP8

Each entry has its own provider name, precision metadata, CSV filename and
PNG filename. Plain FP8 and MXFP8 results must never be merged under one
precision label.

## Quantization contracts

Plain FP8 uses E4M3 activation and weight tensors. Activations use one FP32
scale per token and weights use one FP32 scale per output channel. Linear
calls `npu_quant_matmul`; GroupGemm calls pure `npu_grouped_matmul`.

MXFP8 uses E4M3 activation and weight tensors with group size 32. Scales use
the E8M0 semantic dtype and the runtime pair-packed layout
`[..., K / 64, 2]`. Linear calls `npu_quant_matmul` with
`group_sizes=[1, 1, 32]`; GroupGemm calls pure `npu_grouped_matmul`. The
fused SwiGLU operator is outside this benchmark because its output and FLOP
semantics differ from pure GroupGemm.

All four paths produce BF16 output and use no bias.

## Framework and timing boundary

Source BF16 tensors, dynamic quantization, scale generation and weight layout
conversion are prepared before the device Event timing region. Timed
execution contains only the cached native matmul callable.

The 950PR native APIs do not expose caller-owned `out=` tensors. Providers
therefore do not claim the framework's preallocated-output contract. Framework
V2 still prepares independent input payloads and retains every returned BF16
output until the repeat ends, preventing output storage reuse from changing
the benchmark.

## Device and dispatch rules

The new providers are formal only when the selected NPU reports a device name
beginning with `Ascend950PR` and the required runtime symbols exist.

- `fp8` on CUDA keeps selecting the existing H20 CUTLASS provider.
- `fp8` on Ascend 950PR selects the independent plain-FP8 NPU provider.
- `mxfp8` is an independent precision token and is accepted only on Ascend
  950PR NPU.
- Default formal matrices do not implicitly add either opt-in precision.

## Shapes and artifacts

Linear keeps the existing square matrix sweep, `M=N=K=256..4096` in steps of
128. GroupGemm keeps `E=8`, `K=7168`, `N=4096` and total tokens
`64..32768`.

Artifact stems contain both operator and precision:

- `linear_tflops_fp8_*` and `linear_tflops_curve_fp8_*`
- `linear_tflops_mxfp8_*` and `linear_tflops_curve_mxfp8_*`
- `groupgemm_tflops_fp8_*` and `groupgemm_tflops_curve_fp8_*`
- `groupgemm_tflops_mxfp8_*` and `groupgemm_tflops_curve_mxfp8_*`

CSV rows record the exact implementation/provider, device, precision,
quantization semantics, Framework V2 provenance, warmup, iteration and repeat
counts.

## Validation

Local tests cover precision-token non-aliasing, 950PR device gates, exact
native call arguments, scale layouts, H20 provider preservation, curve
dispatch and artifact separation.

Remote validation proceeds in this order:

1. Import and runtime-symbol smoke.
2. Small-shape correctness against BF16 references.
3. Representative formal endpoints.
4. Full Framework V2 curves.
5. Pull CSV, PNG and logs to the local workspace and audit all rows.
