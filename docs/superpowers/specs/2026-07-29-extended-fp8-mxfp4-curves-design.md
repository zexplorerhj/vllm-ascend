# Extended FP8 and Ascend 950PR MXFP4 operator curves

## Scope

Extend the existing square Linear curves from `M=N=K<=4096` to
`M=N=K<=32768`, remeasure the existing H20/Ascend 950PR FP8 variants, and add
one native Ascend 950PR MXFP4 variant.

The Linear shape matrix is exactly:

- `256..4096` in steps of `128`;
- `8192`;
- `16384`;
- `32768`.

No other Linear shapes are part of this curve.

The measured Linear series are:

- H20 plain FP8 through the existing vLLM CUTLASS provider;
- Ascend 950PR plain FP8 through `npu_quant_matmul`;
- Ascend 950PR MXFP8 through group-32 `npu_quant_matmul`;
- Ascend 950PR MXFP4 through group-32 `npu_quant_matmul`.

MXFP4 is also added to the existing pure GroupGemm shape matrix if a single
native `npu_grouped_matmul` call passes runtime-symbol, correctness, and
kernel-path validation. Fused SwiGLU and per-expert Python loops are not valid
GroupGemm providers.

Plain FP4 and INT4 are outside this change. The current 950PR runtime exposes
native MXFP4 but rejects non-MX E2M1 dynamic quantization and E2M1 matmul with
ordinary FP32 per-token scales. No unsupported path will be simulated or
relabeled.

## MXFP4 contract

MXFP4 uses packed E2M1 activation and weight data, two values per byte.
Activation and weight scales use group size 32 and E8M0 semantics in the
runtime's pair-packed layout.

Preparation performs the following work outside Event timing:

- create BF16 source tensors;
- quantize activation and weight with `npu_dynamic_mx_quant`;
- normalize and transpose packed weight and scale layouts;
- resolve and cache the native callable.

The timed Linear region contains exactly one `npu_quant_matmul` call with:

- `x1_dtype=x2_dtype=float4_e2m1fn_x2`;
- `scale_dtype=pertoken_scale_dtype=float8_e8m0fnu`;
- `group_sizes=[1, 1, 32]`;
- BF16 output;
- no bias.

The timed GroupGemm region, when supported, contains exactly one pure
`npu_grouped_matmul` call with the equivalent packed dtypes and E8M0 scale
metadata. It must not contain activation quantization, fused activation, or an
expert loop.

## Framework and memory protocol

All measurements use `OperatorTestFramework V2` device Event timing. Every
invocation within one repeat receives independent prepared input storage.
Every returned output is retained until that repeat ends. Inputs or outputs
must not be reused to make the largest shapes fit.

Linear uses two stabilization repeats and five measured repeats. Invocation
counts are selected per shape from a provider-independent worst-case retained
byte estimate, shared by all four Linear series:

1. Set the fresh-storage budget to `40 GiB`.
2. Compute `capacity=floor(40 GiB / worst_case_bytes_per_invocation)`.
3. Set `total_invocations=min(60, capacity)`.
4. Set `warmup=min(10, max(2, floor(total_invocations / 5)))`.
5. Set `iterations=total_invocations-warmup`.

The planner must reject a shape if it cannot provide at least two warmups and
one measured invocation. With the current estimators this gives:

- `8192`: `W10/I50`;
- `16384`: `W7/I32`;
- `32768`: `W2/I7`.

Repeats execute sequentially and release retained storage between repeats.
CSV rows record the effective warmup, iterations, repeats, estimated retained
bytes, verified storage counts, output allocation mode, Event semantics, and
all repeat samples.

The existing GroupGemm protocol and shape matrix remain unchanged. MXFP4 uses
the same per-shape warmup, iteration, repeat, and fresh-storage rules as the
existing GroupGemm series.

## Correctness and native-path validation

Before formal curves:

1. Run import, dtype, device-name, and runtime-symbol gates on
   `Ascend950PR`.
2. Compare small MXFP4 Linear and GroupGemm outputs with a BF16 reference.
   Require matching shape and dtype, finite output, cosine similarity of at
   least `0.95`, and normalized RMSE below `0.25`.
3. Verify independent prepared input storage and retained output storage with
   Framework V2.
4. Profile representative Linear and GroupGemm shapes with Level1
   `PipeUtilization`.
5. Accept a provider only when the profile contains the native quantized
   matmul task and contains no dequantize-to-BF16 fallback matmul sequence.

If pure MXFP4 GroupGemm fails any gate, its CSV records the failure and no
GroupGemm MXFP4 performance curve is published. Linear MXFP4 remains
independent.

## Artifacts

MXFP4 has an independent precision token, provider, CSV, log, and curve:

- `linear_tflops_mxfp4_npu_*.csv`;
- `linear_tflops_curve_mxfp4_npu_*.png`;
- `groupgemm_tflops_mxfp4_npu_*.csv`, only after all GroupGemm gates pass;
- `groupgemm_tflops_curve_mxfp4_npu_*.png`, only after all GroupGemm gates
  pass.

The final outputs are:

- refreshed FP8 Linear and combined Linear/GroupGemm comparison plots;
- an Ascend 950PR Linear precision comparison containing plain FP8, MXFP8,
  and MXFP4 on the exact 34-point square grid;
- an Ascend 950PR GroupGemm precision comparison containing only validated
  single-call native providers;
- source CSVs, run logs, profile exports, and a validation summary.

Plots must expose protocol changes at large shapes in their metadata or
caption and must not silently mix incomplete, failed, simulated, or
semantically different providers.
