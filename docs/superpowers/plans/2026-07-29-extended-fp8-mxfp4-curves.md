# Extended FP8 and MXFP4 Curves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remeasure the H20/Ascend 950PR FP8 Linear curves through square size 32768 and add native Ascend 950PR MXFP4 Linear and pure GroupGemm curves.

**Architecture:** Keep `OperatorTestFramework V2` as the only timing and fresh-storage engine. Add one generic hard-bounded invocation planner, then implement MXFP4 as independent Linear and GroupGemm providers that prepare packed E2M1/E8M0 payloads before timing and issue one native matmul API call inside timing. Keep plotting as an audited analysis artifact built only from successful CSV rows with matching shape and protocol metadata.

**Tech Stack:** Python 3, PyTorch, torch_npu, vLLM CUTLASS ops, pytest, Bash, Matplotlib, SSH/rsync.

## Global Constraints

- The Linear shape matrix is exactly `256..4096` in steps of `128`, followed by `8192`, `16384`, and `32768`.
- Linear uses `S2/R5`, a `40 GiB` fresh-storage hard budget, independent prepared inputs, and outputs retained until the repeat ends.
- All compared Linear providers use the same per-shape warmup and iteration counts, calculated from the worst retained-byte estimate.
- The expected large-shape plans are `8192=W10/I50`, `16384=W7/I32`, and `32768=W2/I7`.
- MXFP4 is packed E2M1 data with group-32 pair-packed E8M0 scales and BF16 output.
- Dynamic quantization, scale generation, and layout preparation remain outside Event timing.
- Timed Linear contains one `npu_quant_matmul`; timed GroupGemm contains one pure `npu_grouped_matmul`.
- Plain FP4, INT4, fused SwiGLU, per-expert Python loops, simulated FP4, and H20 W4A16 fallbacks are outside scope.
- Local code is synchronized only from local to remote. Logs, CSVs, profiles, and images are pulled only from remote to local.
- Preserve the existing dirty `tests/ops/flashattention/__init__.py`; it is unrelated and must not be staged or modified.

---

### Task 1: Checkpoint the Existing FP8/MXFP8 Baseline

**Files:**
- Existing baseline only: `tests/ops/groupgemm/groupgemm_fp8.py`
- Existing baseline only: `tests/ops/groupgemm/groupgemm_fp8_npu.py`
- Existing baseline only: `tests/ops/groupgemm/groupgemm_mxfp8_npu.py`
- Existing baseline only: `tests/ops/linear/linear_fp8_npu_common.py`
- Existing baseline only: `tests/ops/linear/linear_fp8_npu_operator.py`
- Existing baseline only: `tests/ops/linear/linear_mxfp8_npu_operator.py`
- Existing baseline only: `tests/ops/operator_test_framework.py`
- Existing baseline only: `tests/ops/run_tests.sh`
- Existing baseline only: `tests/ops/tests/test_groupgemm.py`
- Existing baseline only: `tests/ops/tests/test_linear.py`
- Existing baseline only: `tests/ut/ops/test_groupgemm_fp8_npu_providers.py`
- Existing baseline only: `tests/ut/ops/test_linear_fp8_pr_providers.py`
- Existing baseline only: `tests/ut/ops/test_operator_curve_entries.py`
- Existing baseline only: `tests/ut/ops/test_operator_curve_providers.py`
- Existing baseline only: `tests/ut/ops/test_operator_dispatcher.py`
- Existing baseline only: `docs/superpowers/specs/2026-07-29-950pr-fp8-mxfp8-ops-design.md`

**Interfaces:**
- Consumes: the existing uncommitted FP8/MXFP8 providers and Framework V2 integration.
- Produces: a tested baseline commit that later tasks can review against.

- [ ] **Step 1: Run the focused baseline tests**

Run:

```bash
pytest -q \
  tests/ut/ops/test_linear_fp8_pr_providers.py \
  tests/ut/ops/test_groupgemm_fp8_npu_providers.py \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_dispatcher.py
```

Expected: all collected tests pass. If a test fails, diagnose the baseline before staging anything.

- [ ] **Step 2: Verify the staged file boundary**

Run:

```bash
git status --short
git diff --check -- tests/ops tests/ut/ops
```

Expected: `tests/ops/flashattention/__init__.py` remains modified but unstaged; no whitespace errors exist in the baseline files.

- [ ] **Step 3: Commit only the verified baseline**

```bash
git add \
  docs/superpowers/specs/2026-07-29-950pr-fp8-mxfp8-ops-design.md \
  tests/ops/groupgemm/groupgemm_fp8.py \
  tests/ops/groupgemm/groupgemm_fp8_npu.py \
  tests/ops/groupgemm/groupgemm_mxfp8_npu.py \
  tests/ops/linear/linear_fp8_npu_common.py \
  tests/ops/linear/linear_fp8_npu_operator.py \
  tests/ops/linear/linear_mxfp8_npu_operator.py \
  tests/ops/operator_test_framework.py \
  tests/ops/run_tests.sh \
  tests/ops/tests/test_groupgemm.py \
  tests/ops/tests/test_linear.py \
  tests/ut/ops/test_groupgemm_fp8_npu_providers.py \
  tests/ut/ops/test_linear_fp8_pr_providers.py \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_dispatcher.py
git commit -m "test(ops): add Ascend FP8 and MXFP8 curve providers"
```

Expected: the commit excludes `tests/ops/flashattention/__init__.py`.

---

### Task 2: Add the Hard-Bounded Linear Invocation Plan and 34-Point Grid

**Files:**
- Modify: `tests/ops/operator_test_framework.py`
- Modify: `tests/ops/tests/test_linear.py`
- Modify: `tests/ops/run_tests.sh`
- Modify: `tests/ut/ops/test_operator_curve_entries.py`
- Modify: `tests/ut/ops/test_operator_dispatcher.py`

**Interfaces:**
- Consumes: `build_fresh_iteration_plan`, `FRESH_ITERATION_PLAN_FIELDS`, and `run_core_operator_performance_test_v2`.
- Produces:
  `build_memory_bounded_fresh_invocation_plan(...) -> dict[str, Any]`,
  `LINEAR_QUANTIZED_FORMAL_SIZES`, and per-row effective `warmup`/`iterations`.

- [ ] **Step 1: Write failing planner and shape tests**

Add literal, hand-derived assertions:

```python
@pytest.mark.parametrize(
    ("bytes_per_invocation", "expected_warmup", "expected_iterations"),
    [
        (272_629_760, 10, 50),       # 8192 MXFP8 retained bytes
        (1_090_519_040, 7, 32),      # 16384 MXFP8 retained bytes
        (4_362_076_160, 2, 7),       # 32768 MXFP8 retained bytes
    ],
)
def test_memory_bounded_plan_preserves_fresh_storage_within_40_gib(
    bytes_per_invocation,
    expected_warmup,
    expected_iterations,
):
    plan = build_memory_bounded_fresh_invocation_plan(
        requested_warmup=10,
        requested_iterations=None,
        base_iterations=50,
        estimated_unique_bytes_per_invocation=bytes_per_invocation,
        fresh_storage_hard_limit_bytes=40 * 1024**3,
    )
    assert plan["effective_warmup"] == expected_warmup
    assert plan["effective_iterations"] == expected_iterations
    assert plan["estimated_fresh_storage_bytes_per_repeat"] <= 40 * 1024**3
```

Also assert the formal quantized grid equals:

```python
[*range(256, 4096 + 1, 128), 8192, 16384, 32768]
```

and has exactly `34` unique ascending points. Add dispatcher coverage proving
`run_tests.sh --formal --operator linear --precision fp8|mxfp8` does not
replace that grid with a dense `4096..32768` range. Task 3 adds the equivalent
`mxfp4` dispatcher assertion when that precision token is introduced.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest -q \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_dispatcher.py
```

Expected: failure because `build_memory_bounded_fresh_invocation_plan` or the
34-point formal grid does not yet exist.

- [ ] **Step 3: Implement the minimal generic planner**

Add this public boundary in `operator_test_framework.py`:

```python
def build_memory_bounded_fresh_invocation_plan(
    *,
    requested_warmup: int,
    requested_iterations: Optional[int],
    base_iterations: int,
    estimated_unique_bytes_per_invocation: int,
    fresh_storage_hard_limit_bytes: int,
    minimum_warmup: int = 2,
    minimum_iterations: int = 1,
) -> Dict[str, Any]:
    capacity = (
        fresh_storage_hard_limit_bytes
        // estimated_unique_bytes_per_invocation
    )
    if requested_iterations is None:
        total = min(requested_warmup + base_iterations, capacity)
        effective_warmup = min(
            requested_warmup,
            max(minimum_warmup, total // 5),
        )
        effective_iterations = total - effective_warmup
    else:
        effective_warmup = requested_warmup
        effective_iterations = requested_iterations
    if (
        effective_warmup < minimum_warmup
        or effective_iterations < minimum_iterations
        or (
            effective_warmup + effective_iterations
        ) * estimated_unique_bytes_per_invocation
        > fresh_storage_hard_limit_bytes
    ):
        raise ValueError("fresh-storage hard limit cannot satisfy protocol")
    return {
        "effective_warmup": effective_warmup,
        "effective_iterations": effective_iterations,
        "estimated_fresh_storage_bytes_per_repeat": (
            (effective_warmup + effective_iterations)
            * estimated_unique_bytes_per_invocation
        ),
        "fresh_storage_hard_limit_bytes": (
            fresh_storage_hard_limit_bytes
        ),
        "fresh_storage_hard_limit_overflow": False,
    }
```

Retain full integer/bool validation and the existing provenance fields in the actual implementation. Existing callers of `build_fresh_iteration_plan` must remain unchanged.

- [ ] **Step 4: Apply the planner to quantized Linear curves**

Define:

```python
LINEAR_QUANTIZED_FORMAL_SIZES = [
    *range(256, 4096 + 1, 128),
    8192,
    16384,
    32768,
]
LINEAR_FRESH_STORAGE_HARD_LIMIT_BYTES = 40 * 1024**3
```

For `fp8` and `mxfp8`, use the MXFP8 retained-byte estimator as the
provider-independent worst case and pass the returned effective warmup and
iterations to Framework V2. Task 3 opts `mxfp4` into this same plan when that
precision is added. FP16/BF16 keep their existing 31-point formal grid and
planner.

- [ ] **Step 5: Run GREEN and regression tests**

```bash
pytest -q \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_dispatcher.py \
  tests/ut/ops/test_operator_curve_providers.py
```

Expected: all tests pass, including exact `34`, `W10/I50`, `W7/I32`, and `W2/I7` assertions.

- [ ] **Step 6: Commit**

```bash
git add \
  tests/ops/operator_test_framework.py \
  tests/ops/tests/test_linear.py \
  tests/ops/run_tests.sh \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_dispatcher.py
git commit -m "test(ops): bound fresh storage for extended Linear curves"
```

---

### Task 3: Add the Ascend 950PR MXFP4 Linear Provider

**Files:**
- Create: `tests/ops/linear/linear_mxfp4_npu_operator.py`
- Modify: `tests/ops/operator_test_framework.py`
- Modify: `tests/ops/tests/test_linear.py`
- Modify: `tests/ops/run_tests.sh`
- Modify: `tests/ut/ops/test_linear_fp8_pr_providers.py`
- Modify: `tests/ut/ops/test_operator_curve_entries.py`
- Modify: `tests/ut/ops/test_operator_curve_providers.py`
- Modify: `tests/ut/ops/test_operator_dispatcher.py`

**Interfaces:**
- Consumes: `LinearFp8NpuBaseOperatorTest`, `npu_dynamic_mx_quant`, and `npu_quant_matmul`.
- Produces: `PrecisionType.MXFP4`, `LinearMxFp4NpuOperatorTest`, precision token `mxfp4`, and provider `npu_quant_matmul_mxfp4_e2m1_e8m0_group32_bf16`.

- [ ] **Step 1: Write the failing provider-contract tests**

Create a complete fake runtime that returns physical packed tensors:

```python
float4_e2m1fn_x2 = 296
float8_e8m0fnu = 293

def npu_dynamic_mx_quant(
    source,
    *,
    dst_type,
    block_size,
    round_mode,
):
    packed = torch.empty(
        *source.shape[:-1],
        source.shape[-1] // 2,
        dtype=torch.uint8,
    )
    scale = torch.empty(
        *source.shape[:-1],
        source.shape[-1] // 64,
        2,
        dtype=torch.uint8,
    )
    return packed, scale
```

Assert for `M=16,N=32,K=64`:

- activation storage is `[16,32] uint8`;
- transposed weight storage is `[32,32] uint8`;
- activation scale is `[16,1,2] uint8`;
- weight scale is `[1,32,2] uint8`;
- quantization uses `dst_type=296`, `block_size=32`, `round_mode="round"`;
- the timed call is exactly one `npu_quant_matmul`;
- `x1_dtype=x2_dtype=296`;
- `scale_dtype=pertoken_scale_dtype=293`;
- `group_sizes=[1,1,32]`;
- output dtype is BF16 and bias is absent;
- provider is formal only when the device name starts with `Ascend950PR`.

- [ ] **Step 2: Run the provider tests and verify RED**

```bash
pytest -q \
  tests/ut/ops/test_linear_fp8_pr_providers.py \
  tests/ut/ops/test_operator_curve_providers.py
```

Expected: import or precision-token failure because the MXFP4 provider does not exist.

- [ ] **Step 3: Implement the minimal provider**

Implement:

```python
class LinearMxFp4NpuOperatorTest(LinearFp8NpuBaseOperatorTest):
    NPU_IMPLEMENTATION = (
        "npu_quant_matmul_mxfp4_e2m1_e8m0_group32_bf16"
    )
    PRECISION = PrecisionType.MXFP4
    GROUP_SIZE = 32
    K_ALIGNMENT = 64
```

Preparation must retain only packed activation, packed transposed weight,
pair-packed scales, dtype codes, and the cached callable. Execution must issue
one `npu_quant_matmul` with the exact interface asserted in Step 1.

- [ ] **Step 4: Integrate the independent precision and artifacts**

Add `MXFP4 = "mxfp4"` to `PrecisionType`. Add `mxfp4` to Linear CLI/formal
dispatch, quantization/output semantics, byte estimation, provider selection,
CSV stem, and PNG stem. Do not alias `mxfp4` to `fp8` or `mxfp8`.

- [ ] **Step 5: Run GREEN and Linear regressions**

```bash
pytest -q \
  tests/ut/ops/test_linear_fp8_pr_providers.py \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_dispatcher.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add \
  tests/ops/linear/linear_mxfp4_npu_operator.py \
  tests/ops/operator_test_framework.py \
  tests/ops/tests/test_linear.py \
  tests/ops/run_tests.sh \
  tests/ut/ops/test_linear_fp8_pr_providers.py \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_dispatcher.py
git commit -m "test(ops): add Ascend MXFP4 Linear provider"
```

---

### Task 4: Add the Ascend 950PR Pure MXFP4 GroupGemm Provider

**Files:**
- Create: `tests/ops/groupgemm/groupgemm_mxfp4_npu.py`
- Modify: `tests/ops/groupgemm/groupgemm_fp8_npu.py`
- Modify: `tests/ops/tests/test_groupgemm.py`
- Modify: `tests/ops/run_tests.sh`
- Modify: `tests/ut/ops/test_groupgemm_fp8_npu_providers.py`
- Modify: `tests/ut/ops/test_operator_curve_entries.py`
- Modify: `tests/ut/ops/test_operator_curve_providers.py`
- Modify: `tests/ut/ops/test_operator_dispatcher.py`

**Interfaces:**
- Consumes: `_BaseGroupGemmFp8NpuOperatorTest`, `npu_dynamic_mx_quant`, and `npu_grouped_matmul`.
- Produces: `GroupGemmMxFp4NpuOperatorTest` and provider `npu_grouped_matmul_mxfp4_e2m1_e8m0_group32_bf16`.

- [ ] **Step 1: Write failing packed-layout and one-call tests**

For `E=2,M=4,N=32,K=64`, assert:

- activation storage is `[4,32] uint8`;
- weight storage is `[2,32,32] uint8`, representing logical `[E,K,N]`;
- activation scale is `[4,1,2] uint8`;
- weight scale is `[2,1,32,2] uint8`;
- one `npu_grouped_matmul` call receives `split_item=2`,
  `group_type=0`, `group_list_type=1`, BF16 output,
  `x_dtype=weight_dtype=296`, and both scale dtype codes `293`;
- no fused SwiGLU symbol is called;
- no expert loop occurs;
- the provider gate requires `Ascend950PR`.

- [ ] **Step 2: Run and verify RED**

```bash
pytest -q \
  tests/ut/ops/test_groupgemm_fp8_npu_providers.py \
  tests/ut/ops/test_operator_curve_providers.py
```

Expected: import or provider-dispatch failure because MXFP4 GroupGemm does not exist.

- [ ] **Step 3: Add reusable packed-data hooks to the base**

Keep FP8/MXFP8 behavior unchanged while introducing hooks equivalent to:

```python
def _quantization_kwargs(self) -> Dict[str, Any]:
    return {"dst_type": torch.float8_e4m3fn}

def _expected_quantized_shapes(
    self,
    *,
    seq_len,
    num_experts,
    hidden_dim,
    out_channel,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return (
        (seq_len, hidden_dim),
        (num_experts, out_channel, hidden_dim),
    )
```

MXFP4 overrides the kwargs with dtype code `296`, block size `32`, round mode
`"round"`, expects half-width `uint8` storage, and adds logical dtype codes to
the grouped-matmul kwargs.

- [ ] **Step 4: Integrate MXFP4 GroupGemm**

Add the independent `mxfp4` precision to provider selection, CLI/formal
dispatch, byte estimates, quantization semantics, CSV/PNG stems, and coverage
tests. Preserve the existing GroupGemm shape matrix and protocol exactly.

- [ ] **Step 5: Run GREEN and all operator regressions**

```bash
pytest -q \
  tests/ut/ops/test_groupgemm_fp8_npu_providers.py \
  tests/ut/ops/test_linear_fp8_pr_providers.py \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_dispatcher.py
```

Expected: all tests pass and the existing FP8/MXFP8 provider tests remain
unchanged in behavior.

- [ ] **Step 6: Commit**

```bash
git add \
  tests/ops/groupgemm/groupgemm_fp8_npu.py \
  tests/ops/groupgemm/groupgemm_mxfp4_npu.py \
  tests/ops/tests/test_groupgemm.py \
  tests/ops/run_tests.sh \
  tests/ut/ops/test_groupgemm_fp8_npu_providers.py \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_dispatcher.py
git commit -m "test(ops): add Ascend MXFP4 GroupGemm provider"
```

---

### Task 5: Validate H20 Capability and Both Remote Platforms

**Files:**
- Create locally after collection:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/logs/`
- Create locally after collection:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/results/`
- Create locally after collection:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/profiles/`

**Interfaces:**
- Consumes: the tested local `tests/ops` implementation.
- Produces: remote smoke/correctness logs, formal CSVs, profile exports, and an evidence-backed H20 FP4 capability statement.

- [ ] **Step 1: Record H20 hardware and native FP4 gates**

Run on `ssh -p 7890 root@localhost`:

```bash
python3 -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.__version__, torch.version.cuda)'
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
```

Inspect the installed vLLM MXFP4 Linear selection. Success criterion: H20 is
SM90 and the true W4A4 path is gated to SM100+, while SM90 selects W4A16
Marlin/dequant fallback. Do not publish an H20 MXFP4 curve.

- [ ] **Step 2: Sync exact local code to H20**

Use rsync or scp from local to a fresh remote staging directory. Do not copy
remote source back:

```bash
rsync -avz -e 'ssh -p 7890' \
  tests/ops/ root@localhost:/root/codex_ops_extended_20260729/tests/ops/
```

- [ ] **Step 3: Run H20 smoke and the 34-point FP8 curve**

First run one small shape. Then run:

```bash
OPERATOR_TEST_STABILIZATION_REPEATS=2 \
PYTHONPATH=/root/vllm-lastest-rebase-test \
python3 tests/ops/tests/test_linear.py \
  --mode tflops --precision fp8 --device cuda:0 \
  --tflops-sizes \
  $(seq 256 128 4096) 8192 16384 32768 \
  --tflops-repeats 5 \
  --result-dir results/h20/linear_fp8
```

Success criterion: 34 successful rows, exact shape grid, `W10/I50`,
`W7/I32`, `W2/I7`, five measured repeat samples, independent inputs, retained
outputs, and no fresh-storage overflow.

- [ ] **Step 4: Sync exact local code to 950PR**

Sync local files to host staging on `218.28.9.108:50228`, then copy that
local-origin staging directory into container `664596e78816`. Do not edit code
inside the container.

- [ ] **Step 5: Run 950PR correctness gates**

Run small `K=64` Linear and `E=2,M=4,N=32,K=64` GroupGemm cases. Compare
against BF16 references and require matching BF16 shape, finite values,
cosine similarity `>=0.95`, and normalized RMSE `<0.25`.

- [ ] **Step 6: Run 950PR formal curves**

Run the 34-point Linear curve separately for `fp8`, `mxfp8`, and `mxfp4`
with `S2/R5`. Run the unchanged GroupGemm matrix for `mxfp4` only after its
single-call correctness gate passes.

- [ ] **Step 7: Profile native MXFP4 kernels**

Collect Level1 `PipeUtilization` profiles for representative Linear and
GroupGemm shapes. Accept the data only if the task names identify quantized
MXFP4 matmul and the timeline contains no dequantize-to-BF16 fallback matmul
sequence.

- [ ] **Step 8: Pull results only and audit**

Pull logs, CSVs, PNGs, and profiler exports into the local artifact directories.
Check row count, grid equality, status, provider, dtype/scale semantics,
protocol, storage verification, repeat spread, and profile task names.

---

### Task 6: Build and Verify the Final Figures

**Files:**
- Create:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/build_comparison.py`
- Create:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/extended_quantized_comparison.csv`
- Create:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/linear_fp8_h20_950pr_extended.png`
- Create:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/linear_950pr_fp8_mxfp8_mxfp4.png`
- Create conditionally:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/groupgemm_950pr_fp8_mxfp8_mxfp4.png`
- Create:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_extended_20260729/validation.json`

**Interfaces:**
- Consumes: audited formal CSVs from Task 5.
- Produces: final comparison CSV, PNG/PDF plots, and machine-readable validation.

- [ ] **Step 1: Write an audit-first plot builder**

The builder must reject:

```python
if len(linear_rows) != 34:
    raise ValueError("Linear curve must contain exactly 34 rows")
if linear_sizes != [*range(256, 4097, 128), 8192, 16384, 32768]:
    raise ValueError("Linear curve shape grid mismatch")
if any(row["status"] not in {"ok", "success"} for row in rows):
    raise ValueError("unsuccessful rows cannot be plotted")
```

It must also check provider names, precision tokens, Framework V2 protocol,
input non-reuse, repeat count, matching per-shape W/I across compared Linear
series, and expected large-shape W/I values.

- [ ] **Step 2: Generate separate and combined figures**

Use a linear x-axis through 4096 and a log2 x-axis or clearly labeled broken
axis for the three large square shapes. Use a linear throughput y-axis. Do not
use a nonuniform unlabeled y-axis. Label every provider and annotate peak
throughput plus the 32768 endpoint.

- [ ] **Step 3: Render and inspect**

Run the builder, open each PNG, and verify readable axes, no overlapping
legends, visible 8192/16384/32768 points, and accurate captions for protocol
changes.

- [ ] **Step 4: Run the final data audit**

Write `validation.json` with source paths, hashes, row counts, exact grids,
provider identities, repeat-spread maxima, failed rows, and profile task names.
The final handoff must link the new artifact directory and must explicitly say
that the old 4096-only plot was not overwritten.
