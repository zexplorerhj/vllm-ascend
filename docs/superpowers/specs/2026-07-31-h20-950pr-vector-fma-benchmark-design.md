# H20 / Ascend 950PR Vector FMA Benchmark Design

Date: 2026-07-31

## Goal

Measure and explain the device-level non-Tensor FP16/BF16 Vector throughput of
NVIDIA H20-3e and Ascend 950PR without using GEMM, Tensor Core, or Cube
instructions.

The benchmark must answer two different questions:

1. How fast do the two devices execute the same register/UB-resident FMA
   workload through their current compiler stacks?
2. What architecture-native Vector throughput can each device sustain when
   its native packed/vector instruction form is used?

The benchmark must not infer Vector peak from Add, RMSNorm, NormQuant, or
another bandwidth/reduction-heavy operator.

## Scope and terminology

The measured devices are:

- `NVIDIA H20-3e`, compute capability 9.0, 78 SMs.
- `Ascend950PR_957b`, 56 Vector cores and 28 Cube cores as reported by
  `torch_npu.npu.get_device_properties(0)`.

SM count and AIV core count are recorded but are not directly comparable
units. Aggregate TFLOP/s is the primary cross-platform result.

One fused multiply-add on one scalar lane counts as two FLOPs. A CUDA
`f16x2` or `bf16x2` instruction therefore counts as four FLOPs per issuing
thread instruction. The same per-element two-FLOP convention is used for the
Ascend AIV kernel.

The user-supplied `H20=44T` and `A5 DT=54T` figures are retained as an
internal-spec reference only. They are not acceptance thresholds because:

- the measured Ascend device is 950PR rather than 950DT;
- the H20 44T figure may use a scalar/unpacked or different FMA convention;
- public product specifications do not establish one shared counting
  convention for these two values.

## Provider matrix

### Common-semantic providers

The common-semantic kernel uses the same recurrence on both platforms:

```text
for r in range(R):
    acc[j] = fma(acc[j], a[j], b[j])
```

Each program/thread owns 4 to 8 independent accumulator chains. Inputs are
loaded once from GM/HBM, all `R` FMAs execute in registers or UB, and one
checksum/output is stored at the end.

The initial implementation uses one raw Triton kernel source for CUDA and
the installed Ascend Triton backend. It must not use `tl.dot`, matrix
instructions, or a PyTorch allocating wrapper.

The generated lowering determines how a result is labeled:

- native FP16/BF16 Vector lowering: publish as common FP16/BF16 FMA;
- FP32 promotion plus conversion: publish as a conversion-path result, not
  as native FP16/BF16 peak;
- Tensor/Cube lowering: reject the provider.

### H20 architecture-native providers

H20 uses a small CUDA extension with inline PTX:

- `cuda_ptx_f16_scalar_fma`
- `cuda_ptx_f16x2_fma`
- `cuda_ptx_bf16_scalar_fma`
- `cuda_ptx_bf16x2_fma`

The packed providers are the primary native-peak rows. Inline assembly is
volatile, accumulators are runtime initialized, and the final values are
stored so the compiler cannot remove the loop.

The generated binary must contain the requested FP16/BF16 FMA instruction
form and must not contain HMMA, MMA, or WGMMA instructions.

### 950PR architecture-native providers

950PR uses an AscendC AIV-only kernel:

- `npu_ascendc_fp16_vector_fma`
- `npu_ascendc_bf16_vector_fma`

Each AIV core loads its tile from GM to UB once, performs the repeated
Vector FMA/Mla workload in UB, and writes one final tile/checksum to GM.
Cube execution is forbidden.

Ascend has no requirement to imitate CUDA's scalar-versus-x2 ABI. The
published native result is aggregate FP16/BF16 AIV TFLOP/s.

## Workload and tuning

The tuning dimensions are:

- independent accumulators: `4, 8, 16`;
- H20 threads per block: `128, 256`;
- H20 grid: multiples of the 78-SM count selected from occupancy evidence;
- 950PR block count: 56 AIV cores, with an optional two-blocks-per-core
  challenger only if supported by the compiler/runtime;
- 950PR UB tile: `2048, 4096, 8192` elements subject to UB capacity;
- FMA depth `R`: calibrated from `8192, 32768, 65536, 131072`.

The selected configuration must:

- run one kernel for 5 to 20 ms;
- have no local-memory or UB spill;
- reach a stable throughput plateau as `R` increases;
- keep HBM/GM traffic negligible compared with arithmetic work;
- avoid overflow and non-finite outputs.

The FLOP formula is derived from the actual launched lanes:

```text
FLOPs = 2 * active_scalar_lanes * independent_accumulators * R
```

For the common semantic kernel, the CSV also records the logical
`2 * N * R` formula and the exact mapping from `N` to active lanes.

## Framework integration

Add a self-contained `VectorFmaOperatorTest` suite under `tests/ops` and
reuse the enhanced `OperatorTestFramework` already used by the NormQuant
work.

Preparation must:

- resolve and compile the provider before timing;
- allocate every input and output before timing;
- initialize runtime values that prevent constant folding;
- verify output aliasing for providers with an `out=` contract;
- record device name, SM/AIV count, dtype, instruction form, launch shape,
  accumulator count, and `R`.

Execution must launch exactly one cached raw kernel per invocation. It must
not import modules, allocate tensors, compile code, copy to the host, or
perform correctness work inside the Event interval.

Fresh-address inputs are intentionally not required for this compute-peak
microbenchmark. A single GM/HBM load is amortized by thousands of
register/UB FMAs, and profiler counters must prove that memory traffic is not
the limiting resource. This benchmark remains separate from the fresh-input
operator curves.

## Correctness

Correctness uses a small `R` before performance:

- compare the common semantic result with an FP32 host reference followed by
  the exact destination rounding;
- compare architecture-native results with a device or host recurrence using
  the same dtype semantics;
- require finite output and a nontrivial checksum for the long performance
  loop;
- verify that changing the runtime inputs changes the checksum.

A failed provider is excluded from performance and produces an error artifact.

## Timing protocol

Formal throughput uses device Events around one long kernel:

- sustained preconditioning/warmup: at least 5 seconds and at least 20
  untimed launches;
- measured samples: 30;
- independent processes: 3;
- aggregation: median of process medians;
- publish P5, P95, coefficient of variation, and all raw samples;
- no outlier removal;
- no profiler duration substituted for an Event result.

Graph replay is not required for the peak number because one 5-to-20-ms
kernel makes host launch overhead negligible. A supplemental Graph result
may be recorded to demonstrate this, but it must not replace the Event
series.

Clock, temperature, and power are sampled before and after every process.
Clock locking is optional and must be recorded explicitly if used.

## Profile verification

Profile three representative launches for every published native provider.

### H20

Use `cuobjdump`/SASS and Nsight Compute to verify:

- requested FP16/BF16 scalar or x2 FMA instructions;
- no HMMA, MMA, or WGMMA instructions;
- Tensor-pipe activity is zero;
- register count, achieved occupancy, active warps, and local spill;
- SM clock, kernel cycles/duration, L2 and DRAM bytes.

Metric names are queried from the installed Nsight Compute version instead
of being hard-coded.

### 950PR

Use CANN profiling with PipeUtilization to verify:

- task type `AI_VECTOR_CORE`;
- Cube utilization is zero;
- Vector, Scalar, MTE2, and MTE3 activity;
- no GM transfer inside the repeated arithmetic loop;
- AIV frequency, kernel cycles/duration, power, and temperature where
  available.

Vector pipe active ratio is diagnostic and is not treated as theoretical
TFLOP/s utilization by itself.

## Artifacts

Produce:

- one raw Event CSV per platform, dtype, provider, process, and tuning point;
- one selected-configuration CSV with the full selection rationale;
- H20 SASS and Nsight Compute reports;
- 950PR CANN profile directories and extracted kernel/Pipe CSV files;
- a combined comparison CSV;
- a Markdown report containing:
  - H20 versus 950PR common-semantic throughput;
  - architecture-native peak throughput;
  - FP16 versus BF16 results;
  - scalar versus packed H20 results;
  - measured clocks, variability, and profiler proof;
  - comparison with the internal 44T/54T reference under an explicit
    counting caveat.

## Publication and failure rules

- A kernel that uses Tensor Core or Cube is not a Vector result.
- A promoted FP32 path is not labeled native FP16/BF16.
- A spilled or memory-bound point is not selected as the native peak.
- Profiler time is never mixed into the formal Event series.
- 950PR results are not relabeled as 950DT.
- SM count and AIV core count are never presented as directly comparable
  execution-unit counts.
- The AddRMS/NormQuant results remain operator-competitiveness evidence and
  are not used to infer peak Vector TFLOP/s.
