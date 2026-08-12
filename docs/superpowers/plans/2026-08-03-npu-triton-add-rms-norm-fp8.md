# 950PR Triton AddRmsNorm FP8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and optimize one 950PR Triton provider for the H20/vLLM consumer-equivalent static-FP8 fused AddRmsNorm contract, then compare it fairly with native CANN and H20.

**Architecture:** The provider accepts BF16 `x`, mutable BF16 `residual`, BF16 `weight`, scalar FP32 dequant `scale`, and a caller-preallocated E4M3 output. One Triton kernel rounds `x + residual` to BF16, writes it back to `residual`, computes row-wise RMS in FP32 from that rounded value, rounds normalized BF16 values before E4M3 conversion, and writes one FP8 output. Native CANN remains an independent allocating provider; benchmark rows record provider-specific physical ABI.

**Tech Stack:** Python 3.11, PyTorch/torch_npu 2.10 development image, triton-ascend 3.2.0, pytest, OperatorTestFramework V2, msprof/CANN profiler.

## Global Constraints

- Target only `NormQuantVariant.ADD_RMS_NORM_STATIC_FP8` with `PrecisionType.FP8` on device names beginning `Ascend950PR`.
- Match the H20 vLLM static-FP8 consumer contract: one preallocated `[T,H]` E4M3 output plus BF16 residual updated in place to `x + residual`.
- Use one Triton kernel per logical invocation; preparation, allocation, imports, compilation, correctness reference, and mutable-input restore stay outside the timed region.
- Use scalar `[1]` FP32 dequant scale and `quant = clamp(round_bf16(norm) / scale, -448, 448)`.
- Keep native `torch_npu.npu_add_rms_norm_quant` as a separate provider. Its placeholder second FP8 tensor is not counted as kernel traffic when `scales2` and `zero_points2` are absent.
- Preserve Framework V2 fresh addresses, graph/eager dispatch, Event timing, `W10`, five measured repeats, and correctness-before-performance.
- Do not modify remote source files directly; edit locally, sync local to a disposable remote copy, execute there, and pull results back.
- Do not touch unrelated untracked `tests/ops/vector_fma` files.

---

### Task 1: Implement and unit-test the preallocated Triton provider

**Files:**
- Create: `tests/ops/norm_quant/npu_triton_impl.py`
- Modify: `tests/ops/norm_quant/npu_impl.py`
- Modify: `tests/ops/tests/test_norm_quant.py`
- Modify: `tests/ut/ops/test_norm_quant_providers.py`
- Modify: `tests/ut/ops/test_operator_curve_entries.py`

**Interfaces:**
- Consumes: `NormQuantVariant.ADD_RMS_NORM_STATIC_FP8`, existing fake torch_npu helpers, Framework V2's preallocated-output contract.
- Produces: `run_triton_add_rms_norm_static_fp8_quant_out(output, x, residual, weight, scale, eps) -> torch.Tensor` and provider `npu_triton_fused_add_rms_norm_static_fp8_quant_out`.

- [ ] **Step 1: Write failing discovery, prepare, alias, and byte-accounting tests**

```python
operator = NpuNormQuantOperatorTest(
    NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
    PrecisionType.FP8,
)
assert operator.get_declared_implementations() == [
    operator.ADD_STATIC_FP8,
    operator.TRITON_STATIC_ADD,
]
prepared = operator._prepare_data_for_core_operator(
    data, "npu:0", PrecisionType.FP8, operator.TRITON_STATIC_ADD
)
assert prepared["static_scale"].shape == (1,)
assert prepared["static_scale"].dtype is torch.float32
result = operator._execute_core_operator(prepared, operator.TRITON_STATIC_ADD)
assert result is prepared["output"]
assert operator._declares_preallocated_output_contract(
    prepared, operator.TRITON_STATIC_ADD
)
assert operator.physical_bytes_for_implementation(
    data, operator.TRITON_STATIC_ADD
) == operator.logical_bytes(data)
```

- [ ] **Step 2: Run focused tests in the 950PR image and confirm the new expectations fail**

```bash
pytest -q tests/ut/ops/test_norm_quant_providers.py \
  tests/ut/ops/test_operator_curve_entries.py
```

Expected: failures name the missing Triton provider/interfaces; all pre-existing native-provider assertions remain green.

- [ ] **Step 3: Add the single-kernel Triton implementation**

```python
@triton.jit
def _add_rms_norm_static_fp8_kernel(
    output_ptr, x_ptr, residual_ptr, weight_ptr, scale_ptr,
    rows, cols: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
    for row in range(pid, rows, programs):
        row_offsets = row * cols + offsets
        summed = (
            tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        ).to(tl.bfloat16)
        tl.store(residual_ptr + row_offsets, summed, mask=mask)
        summed_f32 = summed.to(tl.float32)
        inv_rms = 1.0 / tl.sqrt(
            tl.sum(summed_f32 * summed_f32, axis=0) / cols + eps
        )
        normalized = (summed_f32 * inv_rms * weight).to(tl.bfloat16).to(tl.float32)
        quantized = tl.maximum(-448.0, tl.minimum(448.0, normalized * scale_inv))
        tl.store(
            output_ptr + row_offsets,
            quantized.to(output_ptr.dtype.element_ty),
            mask=mask,
        )
```

The wrapper validates contiguous 2-D tensors, `H <= 32768`, E4M3 output, scalar FP32 scale, and launches `min(T, num_vectorcore)` programs with `BLOCK_SIZE=triton.next_power_of_2(H)`.

- [ ] **Step 4: Integrate independent capability probing and preparation**

Expose native and Triton as declared providers. Probe Triton with `[1,64]`, validate output alias plus residual mutation, cache its error independently, and exclude only the failed provider. Prepare `static_scale`, FP8 `output`, callable, `x`, `residual`, `residual_seed`, and `weight` before timing. `_execute_core_operator` performs one cached call and returns the exact output object.

- [ ] **Step 5: Add provider-aware physical-byte provenance**

```python
def physical_bytes_for_implementation(self, data, implementation):
    if implementation == self.TRITON_STATIC_ADD:
        return super().physical_bytes(data)
    return self.physical_bytes(data)
```

Make the curve runner use this method, when present, both in the base row and `physical_bandwidth_gb_s`; retain the existing fallback for every other operator.

- [ ] **Step 6: Run focused tests until green**

```bash
pytest -q tests/ut/ops/test_norm_quant_providers.py \
  tests/ut/ops/test_operator_curve_entries.py
```

Expected: all selected tests pass and native provider behavior is unchanged.

- [ ] **Step 7: Commit the complete green task**

```bash
git add tests/ops/norm_quant/npu_triton_impl.py \
  tests/ops/norm_quant/npu_impl.py \
  tests/ops/tests/test_norm_quant.py \
  tests/ut/ops/test_norm_quant_providers.py \
  tests/ut/ops/test_operator_curve_entries.py
git commit -m "feat(ops): add 950pr triton fused add rms norm fp8"
```

### Task 2: Validate correctness and tune representative shapes

**Files:**
- Modify: `tests/ops/norm_quant/npu_triton_impl.py`
- Modify: `tests/ut/ops/test_norm_quant_providers.py`
- Create: `tests/ops/norm_quant/TRITON_ADD_RMS_NORM_FP8_NOTES.md`

**Interfaces:**
- Consumes: Task 1 Triton and native CANN providers.
- Produces: validated E4M3/residual behavior and one selected configuration for `T={1,128,4096}, H=7168`.

- [ ] **Step 1: Run device correctness for the target variant**

Before the production shapes, run the real `[1,64]` capability probe and
record whether Triton-Ascend compiles the E4M3 store.  The installed
`torch_npu/CANN` runtime rejects `npu_add_rms_norm_quant(div_mode=False)`
with an explicit "only support True" error, so record that attempted native
Mul control as unavailable rather than presenting an unmeasured native
optimization.

```bash
python tests/ops/tests/test_norm_quant.py \
  --mode accuracy --device npu:0 --precision fp8 \
  --variants add_rms_norm_quant \
  --tokens 1 128 4096 --hidden-sizes 7168 --quick
```

Require finite E4M3 output, correct saturation, at most one E4M3 code ULP where permitted, exact BF16 residual equality, and graph capture/replay success in curve smoke.

- [ ] **Step 2: Benchmark representative shapes with identical protocol**

```bash
python tests/ops/tests/test_norm_quant.py \
  --mode curve --device npu:0 --precision fp8 \
  --variants add_rms_norm_quant \
  --tokens 1 128 4096 --hidden-sizes 7168 \
  --warmup 10 --iterations 30 --repeats 5 \
  --dispatch-mode both --no-plot
```

Record five Event samples for Triton and native CANN; verify one target kernel per invocation.

- [ ] **Step 3: Tune only finite, evidence-backed variants**

Compare cyclic versus contiguous rows/core, `min(T,V)` versus `min(T,2V)`
grid, weight load once/program versus once/row, next-power-of-two versus a
two-pass chunked-1024 reduction, and `multibuffer` disabled versus enabled.
The chunked baseline must round and store BF16 residual during pass 1, then
re-read it during pass 2 for gamma, BF16-round, reciprocal-scale multiply,
clamp, and FP8 store.  Start with `BLOCK_M=1`; test `BLOCK_M={2,4}` only after
the one-row variant compiles and passes correctness.  Select a shape-family
branch only if the five-repeat median improves by at least 3% with unchanged
correctness.

- [ ] **Step 4: Profile native and winning Triton at `T=4096,H=7168`**

```bash
python tests/ops/tests/test_norm_quant.py \
  --mode profile --device npu:0 --precision fp8 \
  --variants add_rms_norm_quant --tokens 4096 \
  --hidden-sizes 7168 --warmup 10 --iterations 30
```

Export kernel duration, AIV Vec, Scalar, MTE2, and MTE3. Require one kernel name and `AI_VECTOR_CORE`; treat pipeline ratios as overlapping.

- [ ] **Step 5: Record selected and rejected configurations**

Write exact commands, package/device versions, correctness outcome, five latency samples, selected launch configuration, rejected variants with measurements, and profile summary to `TRITON_ADD_RMS_NORM_FP8_NOTES.md`.

- [ ] **Step 6: Re-run focused unit/device tests and commit**

```bash
pytest -q tests/ut/ops/test_norm_quant_providers.py \
  tests/ut/ops/test_operator_curve_entries.py
git add tests/ops/norm_quant/npu_triton_impl.py \
  tests/ut/ops/test_norm_quant_providers.py \
  tests/ops/norm_quant/TRITON_ADD_RMS_NORM_FP8_NOTES.md
git commit -m "perf(ops): tune 950pr triton add rms norm fp8"
```

### Task 3: Produce formal curves, audit, and comparison report

**Files:**
- Modify: `tests/ops/tests/test_norm_quant.py` only for a tested artifact-metadata correction.
- Create outside git: `fresh_ops_comparison_20260723/norm_quant_triton_20260803/950pr/...`

**Interfaces:**
- Consumes: reviewed Task 2 provider.
- Produces: raw CSV, graph/eager curves, 30-run profile summary, audit JSON, and H20/native-CANN/Triton report.

- [ ] **Step 1: Run the full formal static AddRMS matrix**

```bash
python tests/ops/tests/test_norm_quant.py \
  --mode curve --device npu:0 --precision fp8 \
  --variants add_rms_norm_quant --warmup 10 --repeats 5 \
  --dispatch-mode both
```

Every formal shape/provider/dispatch row must be terminal and successful; preserve checkpoints and raw logs.

- [ ] **Step 2: Build a strict audit**

Validate provider identity, full shape coverage, `W10`, five repeats, fresh addresses, Triton output alias verification, graph status, one logical kernel call, and finite positive latency.

- [ ] **Step 3: Plot continuous comparison curves**

Create latency and logical-bandwidth panels for H20 best static FP8 AddRMS, 950PR native CANN, and 950PR Triton. Reuse existing token/hidden axes, label native and Triton separately, and do not hide them in an unexplained envelope.

- [ ] **Step 4: Write the result report**

Report exact large-shape latency ratios, logical bytes, provider physical ABI, graph/profile agreement, and whether Triton closes the Vector bottleneck. State explicitly if Triton is experimental or loses on any shape family.

- [ ] **Step 5: Run final focused regression tests**

```bash
pytest -q tests/ut/ops/test_norm_quant_providers.py \
  tests/ut/ops/test_operator_curve_entries.py
```

- [ ] **Step 6: Commit only source, tests, and tracked notes**

```bash
git add tests/ops/norm_quant tests/ops/tests/test_norm_quant.py \
  tests/ut/ops/test_norm_quant_providers.py \
  tests/ut/ops/test_operator_curve_entries.py
git commit -m "docs(ops): record 950pr triton add rms norm fp8 results"
```
