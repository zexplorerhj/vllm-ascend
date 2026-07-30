# Norm / Activation Quantized Operator Benchmark Design

Date: 2026-07-30

## Goal

Extend `tests/ops` with reproducible correctness, eager-service, graph-replay,
and diagnostic-profile coverage for the fused RMSNorm + quantization operators
available on NVIDIA H20-3e and Ascend 950PR.

The benchmark must separate device/kernel capability from framework dispatch
overhead. It must not mix profiler duration, eager Event latency, and graph
replay latency in one series.

## Scope

The requested operator and precision matrix is:

| Operator | H20 Plain FP8 | 950PR Plain FP8 | 950PR MXFP8 | 950PR MXFP4 |
|---|---:|---:|---:|---:|
| RmsNormQuant | yes | yes | no | no |
| AddRmsNormQuant | yes | capability gate | no | no |
| AddRmsNormDynamicQuant | yes | capability gate | no | no |
| RmsNormDynamicMxQuant | no | no | yes | yes |
| AddRmsNormDynamicMxQuant | no | no | yes | yes |

HiFP8 is explicitly excluded. H20 MXFP4 is unsupported and remains absent;
INT4, Marlin, or dequantized fallbacks must not be reported as MXFP4.

An Ascend capability gate is a real native smoke test of the requested E4M3
contract. A failed gate produces an explicit `unsupported` row with the
original error. It must not fall back to INT8 or another operator.

## Provider contracts

### NVIDIA H20-3e

H20 providers use raw, preallocated-output interfaces. Python convenience
wrappers that allocate output tensors on every invocation are not allowed in
the timed region.

#### vLLM

- `RmsNormQuant`

  ```python
  torch.ops._C.rms_norm_static_fp8_quant(
      out_fp8, x, weight, static_scale_fp32, epsilon
  )
  ```

- `AddRmsNormQuant`

  ```python
  torch.ops._C.fused_add_rms_norm_static_fp8_quant(
      out_fp8, x, residual, weight, static_scale_fp32, epsilon
  )
  ```

  `residual` is updated in place.

- `AddRmsNormDynamicQuant`

  ```python
  torch.ops._C.rms_norm_dynamic_per_token_quant(
      out_fp8,
      x,
      weight,
      per_token_scale_fp32,
      epsilon,
      None,
      residual,
  )
  ```

  The output and `[tokens, 1]` FP32 scale are preallocated. Passing a residual
  selects the fused add path and updates the residual in place.

#### FlashInfer

FlashInfer CuTe/PDL participates as a challenger for the two static-scale
operators:

- `flashinfer.rmsnorm_quant`
- `flashinfer.fused_add_rmsnorm_quant`

FlashInfer module generation, JIT loading, PDL capability checks, and warmup
must complete before graph capture and before timing. Static scale is a
precreated, contiguous, one-element CUDA FP32 tensor.

SGLang is not an independent CUDA provider because its H20 RMSNorm path
reuses vLLM custom operators. It is therefore not emitted as a duplicate
series.

The H20 best-provider envelope is selected independently for each shape from
successful vLLM and FlashInfer rows. The CSV retains the chosen provider for
every point. Provider-specific CSV files are preserved for audit.

### Ascend 950PR

950PR providers call one native `torch.ops.npu` operator per logical
invocation:

- Plain FP8 `RmsNormQuant`:
  `npu_rms_norm_quant(..., dst_dtype=torch.float8_e4m3fn)`.
- Plain FP8 `AddRmsNormQuant`:
  `npu_add_rms_norm_quant(..., dst_type=<E4M3>)`, only after a successful
  native E4M3 capability gate.
- Plain FP8 `AddRmsNormDynamicQuant`:
  `npu_add_rms_norm_dynamic_quant(..., y_dtype=torch.float8_e4m3fn)`, only
  after a successful native E4M3 capability gate.
- MXFP8 `RmsNormDynamicMxQuant`:
  `npu_rms_norm_dynamic_mx_quant(..., dst_type=292)`.
- MXFP4 `RmsNormDynamicMxQuant`:
  `npu_rms_norm_dynamic_mx_quant(..., dst_type=296)`.
- MXFP8 `AddRmsNormDynamicMxQuant`:
  `npu_add_rms_norm_dynamic_mx_quant(..., dst_type=292)`.
- MXFP4 `AddRmsNormDynamicMxQuant`:
  `npu_add_rms_norm_dynamic_mx_quant(..., dst_type=296)`.

MX scale is E8M0 with logical group size 32. MXFP4 uses pair-packed E2M1
output and must pass integer dtype code `296` on the validated 950PR image.

The current 950PR image has already shown that E4M3
`npu_add_rms_norm_dynamic_quant` can reject parameters. The benchmark treats
this as an expected capability-gate outcome, not as permission to substitute
INT8.

Native NPU APIs allocate and return their outputs. The timed eager path
retains all outputs until the repeat ends and does not claim an `out=`
contract.

## Shape matrix

All inputs are two-dimensional `[tokens, hidden]` BF16 tensors. Epsilon is
`1e-6`.

### Token sweep

The primary curve fixes `hidden=7168` and uses:

```text
tokens = 1, 2, 4, 8, 16, 32, 64, 128,
         256, 512, 1024, 2048, 4096
```

### Hidden-size sweep

The secondary curve fixes `tokens=128` and uses:

```text
hidden = 4096, 7168, 8192
```

All hidden sizes are multiples of 32 and satisfy MXFP4 packing constraints.

## Timing modes

### 1. `graph_chain_event`

This is the primary hardware/kernel comparison curve.

- Build `I` independent prepared payloads for one shape.
- Capture a chain containing exactly one operator invocation for each of the
  `I` payloads.
- Each captured invocation has independent input and output storage. Capturing
  one fixed-address invocation and replaying it `I` times is forbidden.
- Capture and warmup are not timed.
- Keep an immutable, preallocated baseline for every input or residual that
  the operator mutates. Restore the graph payloads after capture and before
  every graph warmup or measured repeat. Every restore completes outside the
  Event window.
- Record start Event, replay the captured chain once, record end Event, and
  divide elapsed time by the number of captured invocations.
- CUDA uses `torch.cuda.CUDAGraph`.
- NPU uses `torch_npu.npu.NPUGraph` and
  `torch_npu.npu.graph`.
- If a provider cannot be captured, emit an `unsupported_graph_capture` row.
  Profiler kernel time must not be inserted into the graph series.

The graph chain has one host graph launch and preserves independent-address
device work within the captured chain. It represents execution inside a model
graph, not ordinary eager Python dispatch.

### 2. `eager_fresh_event`

This is the secondary service-stack curve.

- Reuse `OperatorTestFramework.run_core_operator_performance_test_v2`.
- Prepare independent warmup and measured payloads before timing.
- Dispatch one prepared payload per logical invocation from the Python loop.
- Use one device Event pair around the complete measured loop.
- Preserve all framework provenance and output-storage checks.
- The result intentionally includes stream-idle gaps when the host/runtime
  cannot submit the next eager invocation quickly enough.

The eager and graph modes use the same shape, provider, input values,
warmup/iteration schedule, repeat count, and aggregation policy.

### Window size and aggregation

- Stabilization repeats: 2.
- Measured repeats: 5.
- Center: median of the five repeat means.
- No point or repeat outlier is removed.
- The iteration planner targets a 20–50 ms Event window where memory capacity
  permits.
- Fresh-address storage is hard-bounded. When the shape cannot fit the target
  iteration count, the planner reduces graph width and eager iterations
  symmetrically while keeping at least two warmups and one measured
  invocation.

## Diagnostic profiling

Profiler data is diagnostic only and never supplies a formal curve value.

For each successful provider, profile representative token counts:

```text
tokens = 1, 128, 4096
hidden = 7168
```

The profile active window calls the already prepared core operator. It must
not call `run_device_implementation`, perform H2D/D2H copies, call `.cpu()`,
or construct quantization parameters.

Record:

- logical invocation count;
- device kernel count and kernel names;
- allocator/runtime calls in the active window;
- CPU launch spacing and device stream gaps;
- H20 kernel duration and memory/SM evidence available from the trace;
- 950PR Vector, MTE, scalar, cycles, active bandwidth, frequency, power, and
  temperature evidence exposed by the profiler and `npu-smi`.

The expected native contract is one fused device kernel per logical
invocation. Any split implementation is retained but labeled with its actual
kernel count.

## Correctness

Correctness runs before performance for every provider and shape family.

- Compute BF16 RMSNorm and Add+RMSNorm references in FP32 accumulation.
- Validate static FP8 by dequantizing with the exact provider scale semantics.
- Validate dynamic FP8 output and its per-token FP32 scale separately.
- Validate MXFP8 and MXFP4 after decoding the E8M0 group-32 scales.
- Unpack both E2M1 values from each MXFP4 byte before comparison.
- Validate Add operators' updated residual or `x_out` independently against
  `x1 + x2`.
- Validate all output shapes, dtypes, scale layouts, and storage mutation
  contracts.
- Use deterministic inputs bounded to avoid FP8 saturation when validating
  fused and reference paths.

A provider that fails correctness is excluded from performance and produces a
failure artifact containing the error and observed metadata.

## Metrics and plots

Every CSV row records:

- operator, precision, platform, device model, provider, shape;
- latency in microseconds;
- logical effective bandwidth in GB/s;
- dispatch mode;
- graph capture width and graph replay count;
- eager warmup, iterations, stabilization repeats, measured repeats;
- Event-window samples and repeat spread;
- output allocation and storage policy;
- kernel count from the matching diagnostic profile when available;
- capability and graph-capture status.

Logical effective bandwidth counts bytes read and written by the fused
operator contract, including residual, quantized output, dynamic scale, and
MX scale traffic as applicable. Latency remains the primary metric because
the operators have different output contracts.

Artifacts are independent:

- provider-specific eager CSV files;
- provider-specific graph CSV files;
- H20 best-provider envelope CSV;
- one latency plot per operator with graph as the solid primary series and
  eager as the dashed secondary series;
- one effective-bandwidth plot per operator;
- raw H20 and 950PR profile directories;
- a Markdown summary explaining graph/eager gaps and unsupported cells.

## Failure and publication rules

- No HiFP8 row or legend entry is created.
- No unsupported precision is replaced by another dtype.
- No profiler-derived latency is mixed into Event curves.
- No eager point is mixed into a graph series.
- No provider is labeled H20 unless the captured hardware name is
  `NVIDIA H20-3e`.
- No provider is labeled 950PR unless the captured hardware name is
  `Ascend950PR`.
- A plot is published only from successful, correctness-validated rows with
  complete protocol provenance.
