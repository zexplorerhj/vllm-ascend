# H20 / Ascend 950PR Vector FMA Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an auditable FP16/BF16 non-Tensor Vector FMA benchmark on NVIDIA H20-3e and Ascend 950PR.

**Architecture:** A platform-independent operator contract supplies workload configuration, FLOP accounting, correctness, and OperatorTestFramework V2 integration. A common Triton provider runs the same register-resident recurrence on CUDA and NPU; platform-native providers and profiler gates prove that published results use CUDA Core/AIV rather than Tensor Core/Cube. Formal Event samples and profiler artifacts remain separate.

**Tech Stack:** Python 3.11, PyTorch 2.10/2.11, Triton 3.2, OperatorTestFramework V2, CUDA 12.9/PTX, CANN 9.1, torch_npu, Nsight Compute, msprof.

## Global Constraints

- Count one scalar-lane FMA as exactly 2 FLOPs.
- Publish only kernels with no Tensor Core, MMA, WGMMA, HMMA, or Cube execution.
- Label FP32-promoted FP16/BF16 lowering as a conversion path, never as native FP16/BF16 peak.
- Allocate inputs and outputs before timing; timed execution launches exactly one cached raw kernel.
- Tune each selected kernel to 5–20 ms, warm for at least 5 seconds and 20 launches, then collect 30 samples in each of 3 processes.
- Formal latency comes from device Events; profiler duration is diagnostic only.
- Reuse the enhanced `tests/ops/operator_test_framework.py` from this worktree.
- Record H20 as 78 SMs and the measured `Ascend950PR_957b` as 56 AIV/28 Cube cores, but never compare SM and AIV counts as equivalent units.
- Keep the user-supplied 44T/54T figures as an explicitly caveated internal reference, not a pass criterion.
- If a required compiler/profiler is unavailable, emit an unsupported artifact with the exact command and error; do not substitute GEMM, Tensor Core, Cube, Add, or RMSNorm data.

---

### Task 1: Platform-independent Vector FMA contract

**Files:**
- Create: `tests/ops/vector_fma/__init__.py`
- Create: `tests/ops/vector_fma/base.py`
- Create: `tests/ut/ops/test_vector_fma_contract.py`

**Interfaces:**
- Produces: `VectorFmaConfig`, `VectorFmaOperatorTestBase`, `vector_fma_flops`, `vector_fma_arithmetic_intensity`.
- Consumes: `BaseOperatorTest`, `DeviceType`, and `PrecisionType` from `operator_test_framework`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_fma_flops_counts_scalar_lanes():
    config = VectorFmaConfig(
        elements=4096,
        fma_depth=1024,
        accumulators=8,
        block_size=256,
        num_programs=78,
    )
    assert vector_fma_flops(config) == 2 * 4096 * 8 * 1024


def test_fma_config_rejects_bool_and_non_positive_values():
    with pytest.raises(ValueError, match="elements"):
        VectorFmaConfig(
            elements=True,
            fma_depth=1024,
            accumulators=8,
            block_size=256,
            num_programs=78,
        )
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_contract.py
```

Expected: import failure because `tests.ops.vector_fma.base` does not exist.

- [ ] **Step 3: Implement the immutable configuration and formulas**

```python
@dataclass(frozen=True)
class VectorFmaConfig:
    elements: int
    fma_depth: int
    accumulators: int
    block_size: int
    num_programs: int

    def __post_init__(self) -> None:
        for name in (
            "elements",
            "fma_depth",
            "accumulators",
            "block_size",
            "num_programs",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive non-bool int")


def vector_fma_flops(config: VectorFmaConfig) -> int:
    return 2 * config.elements * config.accumulators * config.fma_depth


def vector_fma_arithmetic_intensity(config: VectorFmaConfig, element_size: int) -> float:
    transferred = config.elements * config.accumulators * element_size * 3
    return vector_fma_flops(config) / transferred
```

`VectorFmaOperatorTestBase` must generate deterministic finite CPU inputs,
calculate TFLOP/s from `avg_time_ms`, expose the configuration in prepared
payloads, and require subclasses to implement one cached raw launch.

- [ ] **Step 4: Run the focused tests**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_contract.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/ops/vector_fma tests/ut/ops/test_vector_fma_contract.py
git commit -m "test(ops): define vector fma benchmark contract"
```

---

### Task 2: Common Triton register-resident provider

**Files:**
- Create: `tests/ops/vector_fma/triton_impl.py`
- Create: `tests/ut/ops/test_vector_fma_triton_provider.py`

**Interfaces:**
- Consumes: `VectorFmaConfig`, `VectorFmaOperatorTestBase`.
- Produces: `TritonVectorFmaOperatorTest`, provider names `triton_common_fp16_fma` and `triton_common_bf16_fma`.

- [ ] **Step 1: Write failing provider tests**

The tests mock Triton launch and verify:

```python
def test_execute_launches_one_cached_raw_kernel(prepared, fake_kernel):
    operator = TritonVectorFmaOperatorTest(PrecisionType.BF16)
    prepared["kernel"] = fake_kernel
    result = operator._execute_core_operator(prepared, "triton_common_bf16_fma")
    fake_kernel.assert_called_once()
    assert result is prepared["output"]


def test_prepared_output_contract_is_declared():
    operator = TritonVectorFmaOperatorTest(PrecisionType.FP16)
    assert operator._declares_preallocated_output_contract(
        {"implementation": "triton_common_fp16_fma"}
    )
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_triton_provider.py
```

Expected: import failure for `triton_impl`.

- [ ] **Step 3: Implement the raw Triton kernel**

The kernel must:

```python
@triton.jit
def vector_fma_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    fma_depth: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.5)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.25)
    acc = tl.load(output_ptr + offsets, mask=mask, other=0.125)
    for _ in tl.range(0, fma_depth, loop_unroll_factor=1):
        acc = tl.fma(acc, a, b)
    tl.store(output_ptr + offsets, acc, mask=mask)
```

The provider must allocate `a`, `b`, and `output` in
`_prepare_data_for_core_operator`, resolve/compile during an untimed probe,
and launch exactly this cached kernel in `_execute_core_operator`. If the
installed Triton backend rejects `tl.fma`, record that provider as
unsupported rather than replacing it with separate PyTorch kernels.

- [ ] **Step 4: Run unit tests and syntax checks**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_vector_fma_contract.py \
  tests/ut/ops/test_vector_fma_triton_provider.py
python -m py_compile tests/ops/vector_fma/base.py tests/ops/vector_fma/triton_impl.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/ops/vector_fma/triton_impl.py tests/ut/ops/test_vector_fma_triton_provider.py
git commit -m "feat(ops): add common triton vector fma provider"
```

---

### Task 3: Formal CLI and OperatorTestFramework V2 output

**Files:**
- Create: `tests/ops/tests/test_vector_fma.py`
- Create: `tests/ut/ops/test_vector_fma_runner.py`
- Modify: `tests/ops/run_tests.sh`

**Interfaces:**
- Consumes: provider classes and `OperatorTestFramework.run_core_operator_performance_test_v2`.
- Produces: formal CSV rows and commands under the `vector-fma` operator name.

- [ ] **Step 1: Write failing runner tests**

Cover exact CLI parsing, FLOP-derived TFLOP/s, 30 samples, three process
indices, device provenance, and fail-closed behavior:

```python
def test_formal_defaults_are_exact():
    args = parse_args(["--mode", "formal", "--device", "cuda:0"])
    assert args.warmup == 20
    assert args.samples == 30
    assert args.process_index in (0, 1, 2)


def test_csv_rejects_unverified_tensor_or_cube_provider():
    with pytest.raises(RuntimeError, match="profile verification"):
        build_formal_row(metrics, profile_status="unverified")
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_runner.py
```

Expected: import failure for `tests.ops.tests.test_vector_fma`.

- [ ] **Step 3: Implement the formal runner**

The formal path must call:

```python
metrics = framework.run_core_operator_performance_test_v2(
    operator,
    data,
    args.device,
    precision,
    implementation=provider,
    num_warmup=args.warmup,
    num_iterations=1,
    num_repeats=args.samples,
    retain_outputs=True,
    verify_independent_storage=True,
    num_stabilization_repeats=0,
    dispatch_mode="eager_direct",
)
```

Write one row per process/provider/config with raw repeat samples,
configuration, FLOPs, median latency, TFLOP/s, device properties, clocks,
temperature, power, and an initially `unverified` profile status. Formal
aggregation must refuse to publish a peak row until a matching verified
profile manifest exists.

Add `run_tests.sh vector-fma ...` dispatch without changing existing operator
commands.

- [ ] **Step 4: Run runner and dispatcher tests**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_vector_fma_runner.py \
  tests/ut/ops/test_operator_curve_entries.py
bash -n tests/ops/run_tests.sh
```

Expected: all tests pass and shell syntax is valid.

- [ ] **Step 5: Commit**

```bash
git add tests/ops/tests/test_vector_fma.py tests/ut/ops/test_vector_fma_runner.py tests/ops/run_tests.sh
git commit -m "feat(ops): add vector fma formal runner"
```

---

### Task 4: H20 native packed PTX provider

**Files:**
- Create: `tests/ops/vector_fma/csrc/vector_fma_cuda.cu`
- Create: `tests/ops/vector_fma/cuda_impl.py`
- Create: `tests/ut/ops/test_vector_fma_cuda_provider.py`

**Interfaces:**
- Consumes: `VectorFmaConfig`, `VectorFmaOperatorTestBase`.
- Produces: scalar and x2 FP16/BF16 PTX providers with one raw launch method.

- [ ] **Step 1: Write failing provider tests**

Verify exact device-name gating, build-cache reuse, one module call per
execution, output aliasing, and the FLOP lane multiplier for x2 providers.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_cuda_provider.py
```

Expected: import failure for `cuda_impl`.

- [ ] **Step 3: Implement volatile PTX kernels and Python loader**

The CUDA source must use the exact instruction forms:

```cpp
asm volatile("fma.rn.f16x2 %0, %1, %2, %3;"
             : "=r"(acc0) : "r"(acc0), "r"(a0), "r"(b0));
asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;"
             : "=r"(acc0) : "r"(acc0), "r"(a0), "r"(b0));
```

Use 4/8/16 independent accumulators, a runtime loop depth, and a single
final store per accumulator. Expose one PyBind launch function per dtype and
instruction form. `cuda_impl.py` compiles once before timing with
`torch.utils.cpp_extension.load`, caches the module, and rejects any device
whose exact name is not `NVIDIA H20-3e`.

- [ ] **Step 4: Run unit tests**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_cuda_provider.py
```

Expected: all mocked loader and contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/ops/vector_fma/csrc/vector_fma_cuda.cu \
  tests/ops/vector_fma/cuda_impl.py \
  tests/ut/ops/test_vector_fma_cuda_provider.py
git commit -m "feat(ops): add h20 packed ptx vector fma provider"
```

---

### Task 5: 950PR native AIV provider

**Files:**
- Create: `tests/ops/vector_fma/ascendc/vector_fma.cpp`
- Create: `tests/ops/vector_fma/ascendc/vector_fma.h`
- Create: `tests/ops/vector_fma/npu_impl.py`
- Create: `tests/ut/ops/test_vector_fma_npu_provider.py`

**Interfaces:**
- Consumes: `VectorFmaConfig`, `VectorFmaOperatorTestBase`.
- Produces: `npu_ascendc_fp16_vector_fma` and `npu_ascendc_bf16_vector_fma`.

- [ ] **Step 1: Write failing provider tests**

Verify exact `Ascend950PR` name gating, `vector_core_num=56`, compiled-symbol
availability, one cached launch per execution, and explicit unsupported
status when the standalone AscendC build tool is unavailable.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_npu_provider.py
```

Expected: import failure for `npu_impl`.

- [ ] **Step 3: Implement the AIV-only kernel**

The kernel must use `__aicore__`, GM-to-UB copy before the arithmetic loop,
AscendC Vector FMA/Mla or explicit Mul+Add in UB, and UB-to-GM copy after the
loop. It must not instantiate Matmul/Cube APIs. The Python provider builds
or loads the kernel before timing, records whether the compiled path is
FMA or Mul+Add, and exposes one cached launch.

If CANN 9.1 on 950PR rejects a standalone custom AscendC build, write an
unsupported manifest containing the compiler command, stdout/stderr, CANN
version, and SoC name. The common Triton AIV provider remains valid but must
not be relabeled as the AscendC native provider.

- [ ] **Step 4: Run unit tests**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_npu_provider.py
```

Expected: all mocked provider tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/ops/vector_fma/ascendc tests/ops/vector_fma/npu_impl.py \
  tests/ut/ops/test_vector_fma_npu_provider.py
git commit -m "feat(ops): add 950pr aiv vector fma provider"
```

---

### Task 6: Local verification and remote source synchronization

**Files:**
- Modify only if tests expose defects in Tasks 1–5.
- Create artifacts under:
  `/Users/yucheng/Documents/2027/A5/vector_fma_20260731/`

**Interfaces:**
- Consumes: complete test suite and runner.
- Produces: immutable source archive, SHA-256 manifest, local unit-test log.

- [ ] **Step 1: Run the complete focused unit suite**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_vector_fma_contract.py \
  tests/ut/ops/test_vector_fma_triton_provider.py \
  tests/ut/ops/test_vector_fma_runner.py \
  tests/ut/ops/test_vector_fma_cuda_provider.py \
  tests/ut/ops/test_vector_fma_npu_provider.py
```

- [ ] **Step 2: Run framework regression and static checks**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_operator_test_framework.py
python -m py_compile tests/ops/vector_fma/*.py tests/ops/tests/test_vector_fma.py
bash -n tests/ops/run_tests.sh
git diff --check
```

- [ ] **Step 3: Build a source archive and hash manifest**

Archive only the committed source state, record `git rev-parse HEAD`, and
write SHA-256 values for every synchronized file. Do not copy remote code
back into the worktree.

- [ ] **Step 4: Synchronize local source to isolated remote directories**

Use:

```text
H20:   /root/codex_vector_fma_20260731
950PR: /root/codex_vector_fma_20260731 inside container 25988abdefd2
```

Synchronize local-to-remote only. Remote logs flow back to the local artifact
directory only.

---

### Task 7: H20 Event sweep and profile gate

**Files:**
- No source edits unless a reproducible H20 defect is fixed locally first.
- Pull logs to `vector_fma_20260731/h20/`.

**Interfaces:**
- Produces: per-process Event CSVs, SASS, NCU reports, verified provider manifest.

- [ ] **Step 1: Probe compilers and profilers**

Record `CUDA_HOME`, `nvcc`, `cuobjdump`, `ncu`, PyTorch/CUDA versions, exact
device properties, clocks, temperature, power, and free-memory state.

- [ ] **Step 2: Run correctness and tuning**

Run FP16 and BF16 common Triton plus scalar/x2 PTX providers over the tuning
grid. Select only configurations with finite correct output and 5–20-ms
single-kernel latency.

- [ ] **Step 3: Run three formal processes**

For each selected provider/dtype, run process indices 0, 1, and 2 with
`W20/I1/R30`. Preserve every Event sample.

- [ ] **Step 4: Collect SASS and NCU evidence**

Query installed metric names first. Profile three launches and verify no
HMMA/MMA/WGMMA, Tensor pipe zero, no local spill, stable occupancy, and low
DRAM/L2 contribution.

- [ ] **Step 5: Pull all logs**

Copy only remote artifacts to
`/Users/yucheng/Documents/2027/A5/vector_fma_20260731/h20/` and verify hashes.

---

### Task 8: 950PR Event sweep and profile gate

**Files:**
- No source edits unless a reproducible 950PR defect is fixed locally first.
- Pull logs to `vector_fma_20260731/950pr/`.

**Interfaces:**
- Produces: per-process Event CSVs, CANN profiles, verified provider manifest.

- [ ] **Step 1: Probe compiler/runtime and device state**

Record Triton backend, CANN/torch_npu versions, Bisheng/compiler paths,
`Ascend950PR_957b`, 56 AIV/28 Cube properties, clocks, temperature, power,
and free memory.

- [ ] **Step 2: Run correctness and tuning**

Run FP16/BF16 common Triton providers and, if the build succeeds, native
AscendC providers. Select 5–20-ms configurations with finite correct output.

- [ ] **Step 3: Run three formal processes**

For every selected provider/dtype run process indices 0, 1, and 2 with
`W20/I1/R30`.

- [ ] **Step 4: Collect three CANN profiles**

Enable PipeUtilization and verify `AI_VECTOR_CORE`, Cube utilization zero,
high Vector activity, no GM copies in the arithmetic loop, and consistent
kernel duration.

- [ ] **Step 5: Pull all logs**

Copy only remote artifacts to
`/Users/yucheng/Documents/2027/A5/vector_fma_20260731/950pr/` and verify
hashes.

---

### Task 9: Cross-platform aggregation and report

**Files:**
- Create: `tests/ops/vector_fma/aggregate.py`
- Create: `tests/ut/ops/test_vector_fma_aggregate.py`
- Create: `/Users/yucheng/Documents/2027/A5/vector_fma_20260731/H20_950PR_VECTOR_FMA_REPORT.md`

**Interfaces:**
- Consumes: verified Event CSVs and profiler manifests.
- Produces: combined CSV, selected peaks, Markdown report, plots.

- [ ] **Step 1: Write failing aggregation tests**

Test fail-closed profile status, dtype/provider grouping, median-of-process-
medians, P5/P95/CV calculations, and exclusion of conversion/Tensor/Cube or
spilled rows.

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_aggregate.py
```

- [ ] **Step 3: Implement aggregation**

For each verified provider/dtype:

```text
process center = median(30 raw Event samples)
formal center  = median(three process centers)
TFLOP/s        = exact FLOPs / formal center
```

Keep all raw rows, emit selected and rejected configuration tables, and
include rejection reasons.

- [ ] **Step 4: Write the report**

Report common-semantic and architecture-native ratios separately. Include
H20 scalar/x2 differences, FP16/BF16 differences, measured clocks and
variability, profiler evidence, and the explicit internal 44T/54T counting
caveat.

- [ ] **Step 5: Run final verification**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_vector_fma_*.py
git diff --check
```

Verify every published row has a matching profile manifest and every linked
artifact exists.

- [ ] **Step 6: Commit source and report generator**

```bash
git add tests/ops/vector_fma/aggregate.py tests/ut/ops/test_vector_fma_aggregate.py
git commit -m "feat(ops): aggregate verified vector fma results"
```
