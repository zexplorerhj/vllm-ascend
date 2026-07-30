# Smoother Extended Linear Curves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerun complete H20 FP8 and Ascend 950PR FP8/MXFP8/MXFP4 square
Linear curves on a 39-point grid whose eight post-4096 points make the
extended trend visually continuous.

**Architecture:** Keep `OperatorTestFramework V2`, the existing quantized
Linear providers, and the existing 40 GiB worst-case retained-storage planner
unchanged. Change only the formal size matrix and its contracts, then collect
four fresh source CSVs. Build a new local audited comparison artifact that
rejects any source whose grid, provider, protocol, repeat policy, fresh-storage
policy, or per-point schedule differs from the approved contract.

**Tech Stack:** Python 3, PyTorch, vLLM CUTLASS, torch_npu, pytest, Matplotlib,
Bash, SSH/rsync.

## Global constraints

- Square shapes only, with `M=N=K`.
- The grid is exactly `256..4096 step 128`, then `5120`, `6144`, `7168`,
  `8192`, `12288`, `16384`, `24576`, `32768`.
- Rerun every one of the 39 points for every provider; do not splice old CSVs.
- Use Framework V2 with `S2/R5`, independent prepared input storage, outputs
  retained to repeat end, and quantization/layout preparation outside Events.
- Use the provider-independent MXFP8 retained-byte estimate and 40 GiB hard
  planner for all four series.
- Synchronize source only local to remote; pull only results/logs remote to
  local. Do not edit either remote checkout in place.
- Preserve the earlier 34-point result directory and the original checkout's
  unrelated dirty `tests/ops/flashattention/__init__.py`.

---

### Task 1: Lock the 39-point source contract with TDD

**Files:**

- Modify: `tests/ut/ops/test_operator_curve_entries.py`
- Modify: `tests/ops/tests/test_linear.py`

- [ ] **Step 1: Write the failing tests**

Change the expected formal grid to all 39 exact values. Require full, quick,
and sharded selection provenance to report a 39-point source matrix. Require
the eight extended shapes to produce these plans:

| Shape | Estimated bytes | W | I |
| ---: | ---: | ---: | ---: |
| 5120 | 106496000 | 10 | 50 |
| 6144 | 153354240 | 10 | 50 |
| 7168 | 208732160 | 10 | 50 |
| 8192 | 272629760 | 10 | 50 |
| 12288 | 613416960 | 10 | 50 |
| 16384 | 1090519040 | 7 | 32 |
| 24576 | 2453667840 | 3 | 14 |
| 32768 | 4362076160 | 2 | 7 |

- [ ] **Step 2: Prove RED**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_operator_curve_entries.py \
  -k 'linear_quantized_formal_grid or linear_quantized_default or linear_quantized_quick or linear_quantized_shard or linear_quantized_large_shape'
```

Expected: assertions fail because production still exposes 34 points.

- [ ] **Step 3: Make the minimal production change**

Insert only `5120`, `6144`, `7168`, `12288`, and `24576` into
`LINEAR_QUANTIZED_FORMAL_SIZES`, preserving unique ascending order. Do not
change Framework V2, providers, estimators, or planner behavior.

- [ ] **Step 4: Prove GREEN and run the focused regression**

Run:

```bash
pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_operator_curve_entries.py \
  tests/ut/ops/test_operator_dispatcher.py \
  tests/ut/ops/test_linear_fp8_pr_providers.py \
  tests/ut/ops/test_operator_test_framework.py
bash -n tests/ops/run_tests.sh
git diff --check
```

Require exit code zero for every command.

---

### Task 2: Build a fail-closed 39-point plotting artifact

**Files (outside the source worktree):**

- Create:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_smoother_20260730/build_comparison.py`
- Create:
  `/Users/yucheng/Documents/2027/A5/fp8_mxfp4_smoother_20260730/test_build_comparison.py`

- [ ] **Step 1: Write failing acceptance tests**

Require the builder to reject a missing/duplicate/reordered point, incorrect
provider or precision, non-V2 protocol, any policy other than `S2/R5`
input-fresh/output-retained, and an incorrect W/I or retained-byte estimate.
Require exactly 156 exported source rows.

- [ ] **Step 2: Prove RED**

Run the new acceptance file before the builder exists and retain the failure
as the RED evidence.

- [ ] **Step 3: Implement the minimal audited builder**

Read four explicit new CSV paths only. Export one normalized 156-row CSV and
one validation JSON. Render:

- H20 FP8 vs 950PR FP8;
- 950PR FP8 vs MXFP8 vs MXFP4.

Use a linear, zero-based TFLOPS y-axis. The left panel contains the dense 31
points through 4096; the right panel contains all eight extension points.
Plot the five-repeat median and unfiltered repeat min/max band.

- [ ] **Step 4: Prove GREEN with synthetic sources**

Run:

```bash
pytest -q \
  /Users/yucheng/Documents/2027/A5/fp8_mxfp4_smoother_20260730/test_build_comparison.py
```

Require all builder acceptance tests to pass before supplying real CSVs.

---

### Task 3: Stage and verify the exact source on both remotes

**H20:** `ssh -p 7890 root@localhost`

**950PR:** `root@218.28.9.108:50228`, container `664596e78816`

- [ ] **Step 1: Create new dated staging directories**

Use `/root/codex_ops_smoother_20260730` on H20 and the PR host, with
`/tmp/codex_ops_smoother_20260730/repo` inside the PR container. Do not reuse
or overwrite the 20260729 staging trees.

- [ ] **Step 2: Sync only `tests/ops` from the isolated local worktree**

Verify SHA256 of `tests/test_linear.py` locally and in each runtime before
running.

- [ ] **Step 3: Run the focused PR unit tests**

Inside the PR container, require the focused curve/provider/Framework suite to
pass before benchmarking. On H20, run import/compile and dispatcher smoke
checks appropriate to its installed environment.

---

### Task 4: Collect four complete formal curves

- [ ] **Step 1: Run H20 FP8**

Run the Linear FP8 formal curve with the default quantized grid, `S2/R5`,
Framework V2, and a new result directory. Preserve stdout/stderr and exit
status.

- [ ] **Step 2: Run 950PR FP8, MXFP8, and MXFP4**

Run the three precisions sequentially in the PR container so they do not
contend for the same NPU. Use the same default grid and protocol as H20.
Preserve one log and result directory per precision.

- [ ] **Step 3: Audit remotely before transfer**

For every CSV require 39 rows, all `status=ok`, exact ascending sizes, exact
provider, `coverage_complete=True`, and the expected W/I schedule.

- [ ] **Step 4: Pull complete artifacts**

Copy CSVs and logs to
`/Users/yucheng/Documents/2027/A5/fp8_mxfp4_smoother_20260730/results`.
Record SHA256 for every pulled file.

---

### Task 5: Audit, plot, visually inspect, and hand off

- [ ] **Step 1: Run the builder against the four real source CSVs**

Require acceptance tests, builder validation, 156-row export, and SHA256
manifest to pass.

- [ ] **Step 2: Inspect both PNGs**

Confirm all eight right-panel x values and labels are legible, the gap after
4096 is no longer represented by a three-point line, axes are linear and
zero-based, bands are visible without obscuring lines, and no panel truncates
data.

- [ ] **Step 3: Final verification**

Run the focused source test suite again, `git diff --check`, inspect source
status, and obtain a separate code/data review. Report exact source paths,
hashes, endpoint values, failures if any, and explicitly state that the prior
34-point artifacts were not overwritten.
