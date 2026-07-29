# H20 FP8 Linear and GroupGemm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible H20 FP8 W8A8 Linear and GroupGemm Framework V2 curves using vLLM CUTLASS providers with strictly preallocated outputs.

**Architecture:** Add a shared FP8 quantization/runtime helper and separate FP8 operator classes, then route the existing Linear and GroupGemm curve entry points to those classes for `--precision fp8`. Keep quantization and CUTLASS metadata preparation outside timing; the timed methods only invoke caller-output CUTLASS kernels.

**Tech Stack:** Python, PyTorch, vLLM CUDA custom ops, CUTLASS, pytest, matplotlib, SSH/rsync.

## Global Constraints

- H20/SM90 only in this phase; do not run or modify 950PR.
- FP8 means E4M3 W8A8, per-token activation scale, per-output-channel weight scale, BF16 output, no bias.
- Reuse OperatorTestFramework V2 unchanged for allocation, Device Event timing, repeat aggregation, and storage auditing.
- Preserve the existing uncommitted `tests/ops/flashattention/__init__.py` change.
- Use GPU 0 on H20; GPUs 4-7 are occupied.
- Do not edit source code directly on the remote host.

---

### Task 1: FP8 Precision and Quantization Helpers

**Files:**
- Create: `tests/ops/fp8_utils.py`
- Modify: `tests/ops/operator_test_framework.py`
- Test: `tests/ut/ops/test_operator_curve_providers.py`

**Interfaces:**
- Produces: `PrecisionType.FP8`.
- Produces: `quantize_fp8_per_row(tensor) -> tuple[Tensor, Tensor]`.
- Produces: `quantize_fp8_weight_per_channel(weight_nk) -> tuple[Tensor, Tensor]`.
- Produces: `resolve_vllm_cutlass_scaled_mm()`.
- Produces: `resolve_vllm_cutlass_grouped_mm()`.

- [ ] **Step 1: Write failing unit tests**

Add tests asserting that FP8 is a unique precision, zero rows receive finite
unit scales, nonzero tensors dequantize within FP8 tolerance, and missing vLLM
operators fail with explicit messages.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/ut/ops/test_operator_curve_providers.py -k fp8
```

Expected: failures because `PrecisionType.FP8` and `fp8_utils` do not exist.

- [ ] **Step 3: Implement the minimal helpers**

Use `torch.float8_e4m3fn`, `torch.finfo(...).max`, FP32 scale tensors, and
lazy vLLM imports. Do not allocate CUDA tensors in module import scope.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same pytest command and require exit code 0.

### Task 2: FP8 Linear Provider

**Files:**
- Create: `tests/ops/linear/linear_fp8_operator.py`
- Modify: `tests/ops/tests/test_linear.py`
- Test: `tests/ut/ops/test_operator_curve_providers.py`
- Test: `tests/ut/ops/test_operator_curve_entries.py`

**Interfaces:**
- Produces: `LinearFp8OperatorTest`.
- Consumes: FP8 helpers from Task 1.
- Provider: `cuda_vllm_cutlass_scaled_mm_fp8_bf16`.

- [ ] **Step 1: Write failing provider tests**

Assert data/scale/output shapes, column-major B layout, BF16 output, no bias,
the exact low-level out-first call, strict output declaration, FP8-specific
fresh-byte estimate, and CLI/curve dispatch for `--precision fp8`.

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_curve_entries.py -k 'linear and fp8'
```

- [ ] **Step 3: Implement the provider and curve routing**

Prepare independent FP8 A/B/scales/output for every payload. Execute only
`torch.ops._C.cutlass_scaled_mm(output, A, B, scale_a, scale_b, None)`.
Calculate unique bytes as A + B + BF16 C + FP32 row/channel scales.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same targeted pytest command and require exit code 0.

### Task 3: FP8 GroupGemm Provider

**Files:**
- Create: `tests/ops/groupgemm/groupgemm_fp8.py`
- Modify: `tests/ops/tests/test_groupgemm.py`
- Test: `tests/ut/ops/test_operator_curve_providers.py`
- Test: `tests/ut/ops/test_operator_curve_entries.py`

**Interfaces:**
- Produces: `GroupGemmFp8OperatorTest`.
- Formal provider: `cuda_vllm_cutlass_scaled_mm_fp8_bf16_expert_loop`.
- Diagnostic provider: `cuda_vllm_cutlass_grouped_gemm_fp8_bf16`.

The provider order reflects H20 runtime evidence: grouped CUTLASS fails at
the formal `M=64, E=8` point, while the expert loop covers the entire matrix
without padding or pointwise provider switching.

- [ ] **Step 1: Write failing provider tests**

Assert E4M3 input/weight, `[M,1]` and `[E,N]` FP32 scales, `[M,N]` BF16
output, expert offsets, `[E,3]` problem sizes, stride tensors, grouped-op
arguments, strict output declaration, output semantics, metric name, and CLI.

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_curve_entries.py -k 'groupgemm and fp8'
```

- [ ] **Step 3: Implement the provider and curve routing**

Subclass `BaseGroupGemmOperatorTest`, override CUDA preparation/execution and
formal provider selection, and retain the existing ten-point formal shape
matrix and W10/I30/R3 protocol.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same targeted pytest command and require exit code 0.

### Task 4: Formal Dispatcher and Documentation

**Files:**
- Modify: `tests/ops/run_tests.sh`
- Modify: `tests/ops/README.md`
- Test: `tests/ut/ops/test_operator_dispatcher.py`

**Interfaces:**
- Produces: opt-in H20 FP8 formal entries without forcing FP8 on NPU runs.

- [ ] **Step 1: Write failing dispatcher tests**

Assert that an explicit FP8 selection produces exactly the Linear FP8 and
GroupGemm FP8 commands, while the legacy default matrices remain unchanged.

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q tests/ut/ops/test_operator_dispatcher.py -k fp8
```

- [ ] **Step 3: Add opt-in dispatch and document semantics**

Document W8A8 format, scale granularity, timing exclusion for quantization,
strict output reuse, providers, and H20-only runtime requirement.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same pytest command and require exit code 0.

### Task 5: Local Regression Verification

**Files:**
- No production changes unless a failing test exposes a regression.

- [ ] **Step 1: Run the complete ops unit subset**

```bash
pytest -q \
  tests/ut/ops/test_operator_test_framework.py \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_dispatcher.py
```

- [ ] **Step 2: Inspect the diff**

```bash
git diff --check
git status --short
git diff -- tests/ops tests/ut/ops docs/superpowers
```

Confirm the existing FlashAttention change was not modified.

### Task 6: H20 Runtime Validation and Curves

**Files:**
- Remote code destination: a fresh `/root/codex-*` directory.
- Local result destination: `h20_fp8_ops_<timestamp>/`.

**Interfaces:**
- Consumes: locally verified implementation from Tasks 1-5.
- Produces: smoke logs, provider diagnostic, two CSVs, two PNGs, manifest.

- [ ] **Step 1: Verify the SSH target and GPU**

Use a temporary `known_hosts`, verify the expected host key, hostname, H20
name, SM90 capability, and GPU 0 availability. Stop on identity mismatch.

- [ ] **Step 2: Sync local code to a fresh remote directory**

Use rsync from local to remote. Never pull remote source back over local
source.

- [ ] **Step 3: Run capability and correctness smoke tests**

Source `/root/vllm-latest-env.sh`, set `CUDA_VISIBLE_DEVICES=0`, check both
CUTLASS capability probes, and validate small/representative Linear and
GroupGemm outputs against dequantized references.

- [ ] **Step 4: Compare GroupGemm providers**

Run grouped CUTLASS and the per-expert CUTLASS loop at representative total
token counts 128, 2048, and 32768 with identical prepared tensors and Event
timing. Record all results and keep one fixed formal provider.

- [ ] **Step 5: Run formal Linear FP8 curve**

Run all 31 formal points with Framework V2, checkpoint CSV writes, and PNG
generation.

- [ ] **Step 6: Run formal GroupGemm FP8 curve**

Run all ten formal points with Framework V2 using W10/I30/R3 and generate the
CSV/PNG.

- [ ] **Step 7: Pull results and verify artifacts**

Copy only logs/results from remote to local. Verify every formal point has
`status=ok/success`, coverage is complete, provider/format fields are correct,
metrics are finite, and PNGs open successfully.
