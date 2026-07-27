# Operator Curve Benchmark Consolidation Implementation Plan

> Design: `docs/superpowers/specs/2026-07-27-ops-framework-benchmark-consolidation-design.md`

**Goal:** Make `OperatorTestFramework` the only W/I/R preallocation and device
Event timing engine used by formal H20/A3/A5 operator curves, then remove the
duplicated runners and validate representative points on all three platforms.

**Architecture:** Keep the existing operator prepare/execute hooks. Add repeat
aggregation and strict storage/provenance behavior to Framework V2. Operator
modules own native providers; existing per-operator test entry points own shape
matrices and CSV checkpointing. Plotting remains local.

**Tech stack:** Python, PyTorch, torch-npu, CUDA, vLLM custom ops, FlashInfer,
flash-attn, pytest, shell/SSH/itask.

---

## Task 1: Specify Framework V2 behavior with CPU unit tests

**Files:**

- Create: `tests/ut/ops/test_operator_test_framework.py`
- Read: `tests/ops/operator_test_framework.py`

1. Add a tiny CPU `BaseOperatorTest` fixture whose prepare method creates fresh
   tensors and whose execute method records call/output lifetimes.
2. Add a test requiring `num_repeats=3` to prepare exactly
   `3 * (warmup + iterations)` payloads.
3. Stub `_measure_execution_time_v2` with repeat means `[3.0, 1.0, 2.0]` and
   assert median `avg_time_ms == 2.0` plus ordered `repeat_samples_ms`.
4. Add invalid count tests for negative warmup and non-positive
   iteration/repeat counts.
5. Add storage-reuse regression tests for prepared inputs and returned outputs.
6. Add provenance serialization assertions.
7. Run:

   ```bash
   pytest -q tests/ut/ops/test_operator_test_framework.py
   ```

   Confirm the new tests fail because repeat/provenance behavior is absent.

## Task 2: Implement the single Framework measurement protocol

**Files:**

- Modify: `tests/ops/operator_test_framework.py`
- Test: `tests/ut/ops/test_operator_test_framework.py`

1. Extend `PerformanceMetrics` with repeat, aggregation, protocol, pointer
   count, and timed-region fields.
2. Add strict W/I/R validation.
3. Extract one-repeat preparation/warmup/Event/verification into a private
   helper using the existing prepare/execute hooks.
4. Loop repeats in `run_core_operator_performance_test_v2`, aggregate with
   `statistics.median`, and calculate throughput from the median.
5. Clear payload/output references before device cache cleanup.
6. Add `performance_provenance(metrics)` returning a stable flat dictionary.
7. Preserve old callers with `num_repeats=1`.
8. Run the Task 1 pytest file until green.

## Task 3: Make formal provider storage semantics explicit

**Files:**

- Modify: `tests/ops/add/add_operator.py`
- Modify: `tests/ops/linear/linear_operator.py`
- Modify: `tests/ops/rmsnorm/rmsnorm_operator.py`
- Modify: `tests/ops/flashattention/base.py`
- Modify: `tests/ops/flashattention/impl.py`
- Modify: `tests/ops/groupgemm/base_groupgemm.py`
- Modify: `tests/ops/paged_attention/base.py`
- Modify: `tests/ops/paged_attention/cuda_impl.py`
- Modify: `tests/ops/recurrent_gated_delta_rule/base.py`
- Modify: `tests/ops/recurrent_gated_delta_rule/impl.py`
- Create: `tests/ut/ops/test_operator_curve_providers.py`

1. Add import-safe tests for deterministic formal provider resolution.
2. Add tests that formal PA resolves only FlashInfer FA2 on CUDA and fused
   infer-attention on NPU, with block size 128 enforced by the formal entry.
3. Add H20 `out=` providers previously embedded in the standalone runner:
   Add, Linear mm, and vLLM RMSNorm.
4. Keep both H20 Flash providers needed for pointwise-best reporting; force the
   Flash backend and forbid fallback.
5. Keep GroupGemm BF16 cuBLAS and INT8 CUTLASS providers, removing raw INT32
   fallback from formal selection.
6. Keep only the PA FlashInfer provider in the formal CUDA selection; mark
   flash-attn block-256 and SDPA gather paths as debug-only or remove them when
   unreferenced.
7. Remove Recurrent's operator-local storage audit; Framework supplies it.
8. Ensure every prepare path creates fresh target-device storage.
9. Run provider unit tests and Python compilation.

## Task 4: Replace per-operator repeat/Event code with Framework calls

**Files:**

- Modify: `tests/ops/tests/test_add.py`
- Modify: `tests/ops/tests/test_linear.py`
- Modify: `tests/ops/tests/test_rmsnorm.py`
- Modify: `tests/ops/tests/test_flash_attention.py`
- Modify: `tests/ops/tests/test_groupgemm.py`
- Modify: `tests/ops/tests/test_paged_attention.py`
- Modify: `tests/ops/recurrent_gated_delta_rule/benchmark.py`
- Modify: `tests/ops/tests/test_recurrent_gated_delta_rule.py`

1. Add a test that monkeypatches each formal curve entry and asserts it calls
   Framework V2 with the requested W/I/R and storage verification.
2. Remove FlashAttention `_benchmark_core_with_reused_inputs`.
3. Replace PA's nested `measure_one` repeat loop with one Framework call per
   point.
4. Replace Recurrent's repeat/audit loop with one Framework call per point.
5. Update Add, Linear, RMSNorm, and GroupGemm formal modes to request the same
   historical counts.
6. Use Framework provenance fields in CSV rows.
7. Keep incremental checkpoint writes and fail the process when any formal
   point fails.

## Task 5: Thin the dispatcher and remove obsolete runners

**Files:**

- Modify: `tests/ops/run_tests.sh`
- Delete: `tests/ops/benchmark_h20_paged_backends.py`
- Delete: `tests/ops/benchmark_h20_targeted_rerun.py`
- Delete: `tests/ops/benchmark_h20_preallocated_ops.py`
- Delete: `tests/ops/benchmark_ascend_910c_ops.py`

1. Add a non-interactive formal-curve dispatch mode while keeping the current
   interactive menu compatible.
2. Confirm all formal providers and shape matrices are reachable through the
   per-operator entry points.
3. Search for imports/references to each standalone runner.
4. Delete the four runners only after the search is empty.
5. Compare `git diff --stat` and require a material net line reduction.

## Task 6: Local verification

**Files:**

- Verify all modified files.

Run:

```bash
pytest -q \
  tests/ut/ops/test_operator_test_framework.py \
  tests/ut/ops/test_operator_curve_providers.py
PYTHONPYCACHEPREFIX=/tmp/vllm_ascend_ops_pycache \
  python3 -m compileall -q tests/ops
git diff --check
```

Also run one CPU fake benchmark and inspect its flat provenance.

## Task 7: Remote H20 validation

**Remote:** `ssh -p 7890 root@localhost`

1. Discover a staging path and create a separate remote test tree.
2. Sync local modified files to the staging tree using rsync/scp; do not edit
   remote source.
3. Run import/compile tests in the existing H20 environment.
4. Run W1/I2/R2 smoke points for Add, Linear, RMSNorm, FlashAttention,
   GroupGemm BF16/INT8, PagedAttention, and Recurrent when dependencies exist.
5. Run representative larger points with formal W/I/R counts.
6. Write logs/CSVs under the remote staging log directory.
7. Pull logs/CSVs into a new local validation directory and analyze failures.

## Task 8: Remote Ascend A3 validation

**Remote:** `zhj-dev22` through `itask`.

Repeat the remote-debug bridge workflow:

1. sync into a separate staging tree;
2. compile/import;
3. W1/I2/R2 smoke for every NPU formal provider;
4. representative small/large W/I/R points;
5. pull and analyze logs/CSVs.

## Task 9: Remote Ascend A5 validation

**Remote:** `root@218.28.9.108:50228`, container
`qwen35-122b-fp8-tp4-ep-20260625`.

1. Sync local files to a host staging directory, then copy the staging tree
   into a separate container path.
2. Do not overwrite the container's working checkout.
3. Compile/import and run W1/I2/R2 smoke points.
4. Run representative formal points supported by the A5 torch-npu/CANN stack.
5. Pull logs/CSVs and analyze them locally.

## Task 10: Plot-data and completion verification

**Files:**

- Modify if needed: `/Users/yucheng/Documents/2027/A5/plot_all_ops_h20_910c_950dt.py`
- Verify: formal CSV outputs and existing historical artifacts.

1. Normalize the 950PR input path into a workspace-relative path without
   changing historical data.
2. Feed the new smoke/formal CSV schema through the plot loader.
3. Generate a validation comparison plot with A3/A5/H20 rows and explicit
   missing-data gaps.
4. Check row counts, finite positive values, providers, block size, W/I/R,
   storage counts, and protocol version.
5. Run final unit/static checks again.
6. Request code review and address findings.

