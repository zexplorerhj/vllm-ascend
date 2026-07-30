# Norm / Activation Quantized Operator Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add correctness-validated, fresh-address eager Event, independent-
address graph-chain Event, and prepared-core profiler coverage for the
requested fused RMSNorm + quantization operators on NVIDIA H20-3e and Ascend
950PR, then collect comparable CSV, profile, plot, and Markdown artifacts.

**Architecture:** Extend `OperatorTestFramework` with two generic facilities:
captured prepared-payload chains and prepared-core profiling. Implement the
quantized Norm family in a new `tests/ops/norm_quant` package with a
platform-independent contract plus CUDA and NPU providers. A dedicated formal
entry owns capability gates, correctness, fresh-memory planning, checkpointed
CSV files, graph/eager plots, and profile dispatch.

**Tech Stack:** Python 3, PyTorch device Events and CUDA Graphs,
`torch_npu.npu.NPUGraph`, vLLM raw custom ops, FlashInfer CuTe/PDL,
Ascend native `torch_npu` ops, pytest, Bash, CSV, matplotlib.

## Global Constraints

- HiFP8 is excluded. Do not create a HiFP8 provider, CSV row, or legend.
- Do not report INT8, INT4, Marlin, dequantized, or eager fallback results as
  FP8, MXFP8, MXFP4, or graph results.
- H20 formal rows require the captured device name `NVIDIA H20-3e`; 950PR
  formal rows require a device name beginning with `Ascend950PR`.
- H20 calls raw preallocated-output vLLM ops. Do not time vLLM Python wrappers
  that allocate `torch.empty`.
- H20 static FP8 runs both vLLM and FlashInfer challengers; H20 dynamic
  per-token FP8 runs vLLM raw only.
- 950PR plain FP8 providers must pass a real native E4M3 capability probe.
  Failure emits `unsupported` with the original error and never falls back.
- 950PR MXFP8 uses dtype code `292`; MXFP4 uses integer dtype code `296`;
  logical MX scale grouping is 32 values with physical pair-packed E8M0
  scales.
- Inputs are BF16 `[tokens, hidden]`, epsilon is `1e-6`.
- Token sweep: hidden `7168`, tokens
  `1,2,4,8,16,32,64,128,256,512,1024,2048,4096`.
- Hidden sweep: tokens `128`, hidden `4096,7168,8192`.
- Formal aggregation uses 2 stabilization repeats, 5 measured repeats, and
  the median of repeat means. The fresh-memory planner targets a 20–50 ms
  Event window and reduces eager iterations and graph width symmetrically
  under the same hard memory bound.
- `captured_chain` captures `I` different input/output addresses into one
  graph and times one replay divided by `I`. Capturing one address and
  replaying it `I` times is forbidden.
- Graph capture, graph creation, JIT, capability probing, input restoration,
  H2D/D2H, and output verification are outside the Event window.
- Mutable Add residuals have immutable per-payload seeds and are restored
  after capture and before every measured replay, outside timing.
- Profiler output is diagnostic only and never supplies formal curve latency.
- Preserve the unrelated local modification
  `tests/ops/flashattention/__init__.py`.
- Local files are the code source of truth. Sync local code to remote
  machines; only pull logs, profiles, CSV, and images from remote machines.
- Every repository commit uses Conventional Commits and `git commit -s`.

---

### Task 1: Prepared-core profiler in OperatorTestFramework

**Files:**

- Modify: `tests/ops/operator_test_framework.py`
- Modify: `tests/ut/ops/test_operator_test_framework.py`

**Interfaces:**

- Consumes: existing `_prepare_data_for_core_operator`,
  `_execute_core_operator`, `_core_operator_benchmark_context`, profiler
  factory, and storage-audit helpers.
- Produces:

  ```python
  def run_core_operator_profile_test_v2(
      self,
      operator_test: BaseOperatorTest,
      data: Dict[str, Any],
      device: str,
      precision: PrecisionType,
      implementation: str = "default",
      *,
      num_warmup: int = 10,
      num_iterations: int = 20,
      trace_file_path: str,
      profile_level: str = "Level1",
      aic_metrics: str = "PipeUtilization",
      verify_independent_storage: bool = True,
  ) -> Dict[str, Any]
  ```

- The returned dict contains
  `profile_path`, `provider`, `device`, `precision`, `warmup_iterations`,
  `profiled_invocations`, `preallocated_invocations`,
  `input_storage_sets_verified`, `output_storage_sets_verified`,
  `dispatch_mode="eager_prepared_core"`,
  and `profiler_is_diagnostic=True`.

- [ ] **Step 1: Add a failing prepared-core profile test**

  Add a fake operator whose full API raises, while prepare and core execute
  append observable payload IDs:

  ```python
  class _PreparedProfileOperator(_FreshCpuOperator):
      def __init__(self):
          super().__init__("PreparedProfile")
          self.prepared_ids = []
          self.executed_ids = []

      def run_device_implementation(self, *args, **kwargs):
          raise AssertionError("full device API must not enter core profile")

      def _prepare_data_for_core_operator(
          self, data, device, precision, implementation="default"
      ):
          payload = super()._prepare_data_for_core_operator(
              data, device, precision, implementation
          )
          payload["payload_id"] = len(self.prepared_ids)
          self.prepared_ids.append(payload["payload_id"])
          return payload

      def _execute_core_operator(self, prepared, implementation="default"):
          self.executed_ids.append(prepared["payload_id"])
          return super()._execute_core_operator(prepared, implementation)
  ```

  Use a fake profiler recording `start`, `step`, and `stop`. Assert that
  `W+I` payloads are prepared once, only the final `I` IDs execute while the
  profiler is active, every active call receives one `step`, the full API is
  never called, and the result is diagnostic.

- [ ] **Step 2: Run the focused test and verify RED**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_test_framework.py \
    -k prepared_core_profile
  ```

  Expected: failure because
  `OperatorTestFramework.run_core_operator_profile_test_v2` does not exist.

- [ ] **Step 3: Implement the minimal prepared-core profile path**

  Validate `W >= 0`, `I > 0`, and a CUDA/NPU device. Prepare `W+I` independent
  payloads before creating the profiler, run the first `W` calls unprofiled,
  synchronize, then profile only the final `I` core calls:

  ```python
  with device_context, provider_context, torch.inference_mode():
      for payload in prepared_payloads[:num_warmup]:
          retained_outputs.append(execute(payload, implementation))
      synchronize()
      profiler.start()
      try:
          for payload in prepared_payloads[num_warmup:]:
              retained_outputs.append(execute(payload, implementation))
              profiler.step()
          synchronize()
      finally:
          profiler.stop()
  ```

  Reuse the existing storage-domain audit before dispatch. Do not call
  `run_device_implementation`, `.cpu()`, or any prepare function inside the
  active profiler window. Clear retained payloads only after profiler stop.

- [ ] **Step 4: Run focused and full framework tests**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_test_framework.py \
    -k 'prepared_core_profile or profile'
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_test_framework.py
  ```

  Expected: all selected tests pass and existing V2 eager behavior is
  unchanged.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/ops/operator_test_framework.py \
    tests/ut/ops/test_operator_test_framework.py
  git commit -s -m "feat(ops): profile prepared core operators"
  ```

### Task 2: Independent-address graph-chain dispatch

**Files:**

- Modify: `tests/ops/operator_test_framework.py`
- Modify: `tests/ut/ops/test_operator_test_framework.py`

**Interfaces:**

- Consumes: Task 1's prepared-payload and device-context behavior.
- Produces:

  ```python
  @dataclass
  class _CapturedPreparedChain:
      replay: Callable[[], None]
      retained_outputs: List[Any]
      logical_invocations: int

  def BaseOperatorTest._restore_mutable_graph_inputs(
      self,
      prepared_payloads: List[Any],
      implementation: str = "default",
  ) -> None
  ```

  and the backward-compatible method extension:

  ```python
  def run_core_operator_performance_test_v2(
      self,
      operator_test: BaseOperatorTest,
      data: Dict[str, Any],
      device: str,
      precision: PrecisionType,
      implementation: str = "default",
      num_warmup: int = 10,
      num_iterations: int = 20,
      num_repeats: int = 1,
      retain_outputs: bool = True,
      verify_independent_storage: bool = False,
      num_stabilization_repeats: Optional[int] = None,
      dispatch_mode: str = "eager_direct",
  ) -> PerformanceMetrics
  ```

- Accepted modes are exactly `eager_direct` and `captured_chain`.
- New `PerformanceMetrics`/CSV provenance fields:
  `dispatch_mode`, `graph_capture_width`, `graph_replays`,
  `capture_timed`, `mutable_inputs_restored`, and
  `profiler_is_diagnostic`.

- [ ] **Step 1: Add failing graph-chain tests**

  Add tests that replace `_capture_prepared_payload_chain_v2` with a fake
  captured object. Each prepared payload owns a different CPU storage pointer.
  Assert:

  ```python
  assert captured_payload_ids == list(range(num_iterations))
  assert replay_calls == 1
  assert restore_log == ["after_capture", "before_measured_replay"]
  assert metrics.avg_time_ms == fake_graph_elapsed_ms / num_iterations
  assert metrics.dispatch_mode == "captured_chain"
  assert metrics.graph_capture_width == num_iterations
  assert metrics.capture_timed is False
  ```

  Add separate tests for invalid dispatch mode, capture failure without eager
  fallback, mutable restore occurring outside the fake Event window, retained
  capture outputs surviving replay, and legacy eager provenance remaining
  byte-for-byte unchanged.

- [ ] **Step 2: Run graph tests and verify RED**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_test_framework.py \
    -k 'captured_chain or graph_dispatch or mutable_graph'
  ```

  Expected: failure because graph dispatch and provenance fields do not exist.

- [ ] **Step 3: Implement backend-neutral graph orchestration**

  Append the default no-op restore hook to `BaseOperatorTest`. Within each V2
  repeat:

  1. prepare `W+I` payloads;
  2. execute the first `W` independent payloads as untimed eager warmup;
  3. capture only the final `I` independent payloads;
  4. retain every functional output returned during capture;
  5. call the mutable restore hook and synchronize;
  6. call the hook once more immediately before the measured replay;
  7. record one device Event pair around one `replay()`;
  8. divide elapsed time by `I`;
  9. create fresh payloads and a fresh graph for the next repeat.

  CUDA capture:

  ```python
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
      self._dispatch_prepared_payloads(
          prepared_payloads, execute, implementation, retained_outputs
      )
  replay = graph.replay
  ```

  NPU capture:

  ```python
  graph = torch_npu.npu.NPUGraph()
  with torch_npu.npu.graph(graph):
      self._dispatch_prepared_payloads(
          prepared_payloads, execute, implementation, retained_outputs
      )
  replay = graph.replay
  ```

  Neither backend may catch capture errors and switch to eager. Propagate the
  original exception so the formal entry emits
  `unsupported_graph_capture`.

- [ ] **Step 4: Emit exact graph/eager provenance**

  Graph values:

  ```text
  timing_method=device_event_graph_replay
  timing_semantics=device elapsed time; capture and mutable restore excluded
  dispatch_loop_policy=captured_prepared_payload_chain_single_replay
  timed_region=one graph replay containing I independent core invocations
  graph_replays=1
  capture_timed=false
  profiler_is_diagnostic=false
  ```

  Eager retains its existing timing semantics and sets
  `dispatch_mode=eager_direct`, `graph_capture_width=0`,
  `graph_replays=0`, `capture_timed=false`,
  `mutable_inputs_restored=false`, `profiler_is_diagnostic=false`.

- [ ] **Step 5: Run framework regression tests**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_test_framework.py
  ```

  Expected: all tests pass; existing callers that omit `dispatch_mode`
  continue to report eager V2 behavior.

- [ ] **Step 6: Commit**

  ```bash
  git add tests/ops/operator_test_framework.py \
    tests/ut/ops/test_operator_test_framework.py
  git commit -s -m "feat(ops): add prepared graph-chain timing"
  ```

### Task 3: Common Norm/Quant contract and H20 providers

**Files:**

- Create: `tests/ops/norm_quant/__init__.py`
- Create: `tests/ops/norm_quant/base.py`
- Create: `tests/ops/norm_quant/cuda_impl.py`
- Create: `tests/ut/ops/test_norm_quant_providers.py`

**Interfaces:**

- Consumes: Framework V2 prepare/execute/out contract and graph restore hook.
- Produces:

  ```python
  class NormQuantOperatorTestBase(BaseOperatorTest):
      """Platform-independent data, reference, bytes, and validation."""

  class CudaNormQuantOperatorTest(NormQuantOperatorTestBase):
      """One H20 variant with its fixed list of formal providers."""

  class NormQuantVariant(str, Enum):
      RMS_NORM_STATIC_FP8 = "rms_norm_quant"
      ADD_RMS_NORM_STATIC_FP8 = "add_rms_norm_quant"
      ADD_RMS_NORM_DYNAMIC_FP8 = "add_rms_norm_dynamic_quant"
      RMS_NORM_DYNAMIC_MX = "rms_norm_dynamic_mx_quant"
      ADD_RMS_NORM_DYNAMIC_MX = "add_rms_norm_dynamic_mx_quant"

  def create_norm_quant_operator(
      device: str,
      variant: NormQuantVariant,
      precision: PrecisionType,
  ) -> BaseOperatorTest
  ```

  `operator_name` is
  `f"NormQuant_{variant.value}_{precision.name}"`. The CUDA class is
  parameterized by `variant` rather than copied into three near-identical
  public classes.

- H20 implementation names:

  ```text
  cuda_vllm_rms_norm_static_fp8_quant_out
  cuda_flashinfer_rmsnorm_quant_fp8_out_pdl
  cuda_vllm_fused_add_rms_norm_static_fp8_quant_out
  cuda_flashinfer_fused_add_rmsnorm_quant_fp8_out_pdl
  cuda_vllm_fused_add_rms_norm_dynamic_per_token_fp8_quant_out
  ```

- [ ] **Step 1: Add failing common-contract and H20 provider tests**

  Cover the exact support matrix, exact H20 name gate, fresh storage, output
  alias contract, raw argument ordering, and mutable residual:

  Add tests named
  `test_vllm_static_rmsnorm_execute_uses_raw_out_first`,
  `test_vllm_static_add_orders_residual_and_returns_only_output`,
  `test_vllm_dynamic_add_uses_preallocated_fp8_and_fp32_scale_outputs`,
  `test_flashinfer_static_passes_preallocated_scale_and_enable_pdl`,
  `test_add_variants_restore_only_residual_from_seed`,
  `test_h20_gate_rejects_other_sm90_device_names`, and
  `test_hifp8_has_no_variant_or_provider`.

  Fake raw ops must write the supplied output, write dynamic scales, mutate
  residual, and return no newly allocated tensor. Assertions compare object
  identity and argument positions, not mock call count alone.

- [ ] **Step 2: Run provider tests and verify RED**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_norm_quant_providers.py -k 'cuda or h20 or common'
  ```

  Expected: import failure because `norm_quant` does not exist.

- [ ] **Step 3: Implement common generation, reference, and byte accounting**

  Generate deterministic CPU BF16 sources:

  ```python
  {
      "x": x,
      "residual": residual_or_none,
      "weight": weight,
      "eps": 1e-6,
      "metadata": {
          "tokens": tokens,
          "hidden": hidden,
          "variant": variant.value,
          "precision": precision.name,
      },
  }
  ```

  Reference in FP32 accumulation:

  ```python
  normalized_input = (
      x.float()
      if residual is None
      else (x.to(source_dtype) + residual.to(source_dtype)).float()
  )
  variance = normalized_input.square().mean(dim=-1, keepdim=True)
  reference = normalized_input * torch.rsqrt(variance + eps) * weight.float()
  ```

  Implement observable logical/physical byte accounting for BF16 inputs,
  FP8 output, FP32 dynamic scale, BF16 `x_out`, E8M0 scale, and pair-packed
  MXFP4 output. Empty optional outputs count zero bytes.

- [ ] **Step 4: Implement H20 raw vLLM providers**

  Resolve ops during prepare after importing `vllm._custom_ops` only to
  register the extension:

  ```python
  torch.ops._C.rms_norm_static_fp8_quant(
      output, x, weight, static_scale, eps
  )
  torch.ops._C.fused_add_rms_norm_static_fp8_quant(
      output, x, residual, weight, static_scale, eps
  )
  torch.ops._C.rms_norm_dynamic_per_token_quant(
      output, x, weight, scales, eps, None, residual
  )
  ```

  Static payloads contain a contiguous CUDA FP32 scale with shape `(1,)`.
  Dynamic payloads contain FP8 `output` and FP32 `scales` with shape
  `(tokens, 1)` under `outputs=(output, scales)`. Add payloads keep
  `residual` as an input and `residual_seed` as an immutable input; do not
  list residual under `outputs`.

- [ ] **Step 5: Implement FlashInfer static challengers**

  Resolve and warm the FlashInfer callable before timing:

  ```python
  flashinfer.rmsnorm_quant(
      output, x, weight, static_scale, eps, enable_pdl=True
  )
  flashinfer.fused_add_rmsnorm_quant(
      output, x, residual, weight, static_scale, eps, enable_pdl=True
  )
  ```

  FlashInfer is present only for the two static variants. Both vLLM and
  FlashInfer declare the phase-invariant prepared-output contract. Dynamic
  provider selection contains vLLM only.

- [ ] **Step 6: Add auxiliary correctness validation**

  The untimed correctness method validates:

  - dequantized primary output versus FP32 reference;
  - static FP8 output code/finite range and exact static scale semantics;
  - dynamic FP32 scale shape `(tokens, 1)` and dequantized result;
  - Add residual after execution versus source-dtype `x + residual_seed`;
  - output shapes, dtypes, and storage aliases.

  A correctness failure raises before any performance call.

- [ ] **Step 7: Run provider and framework tests**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_norm_quant_providers.py
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_test_framework.py
  python -m compileall -q tests/ops/norm_quant
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add tests/ops/norm_quant tests/ut/ops/test_norm_quant_providers.py
  git commit -s -m "feat(ops): add H20 norm quant providers"
  ```

### Task 4: Ascend 950PR FP8, MXFP8, and MXFP4 providers

**Files:**

- Create: `tests/ops/norm_quant/npu_impl.py`
- Modify: `tests/ops/norm_quant/__init__.py`
- Modify: `tests/ut/ops/test_norm_quant_providers.py`

**Interfaces:**

- Consumes: Task 3's variants, common data/reference/byte accounting, and
  factory.
- Produces native 950PR implementations:

  ```python
  class NpuNormQuantOperatorTest(NormQuantOperatorTestBase):
      """One 950PR variant/precision and an instance-owned probe cache."""
  ```

  The formal suite constructs one instance for each `(variant, precision)` and
  reuses it across shapes, so
  `self._capability_cache: Dict[Tuple[str, str, str, int], CapabilityResult]`
  probes once without module-level mutable state.

  ```text
  npu_rms_norm_quant_fp8_e4m3_static
  npu_add_rms_norm_quant_fp8_e4m3_static
  npu_add_rms_norm_dynamic_quant_fp8_e4m3_per_token
  npu_rms_norm_dynamic_mx_quant_mxfp8_e4m3_e8m0_g32
  npu_rms_norm_dynamic_mx_quant_mxfp4_e2m1_e8m0_g32
  npu_add_rms_norm_dynamic_mx_quant_mxfp8_e4m3_e8m0_g32
  npu_add_rms_norm_dynamic_mx_quant_mxfp4_e2m1_e8m0_g32
  ```

- [ ] **Step 1: Add failing NPU support-matrix and ABI tests**

  Use a complete fake `torch_npu` structure and fake `torch.ops.npu` schemas.
  Test:

  - exact Ascend950PR device gate;
  - missing symbol, schema fragment, dtype, and return arity rejection;
  - tiny capability probe result caching by device/runtime/op/dtype;
  - probe failure preserves the error and never selects INT8;
  - exact keyword values `dst_type=292` and integer `dst_type=296`;
  - output tuple ordering, shapes, dtypes, and full-tuple retention;
  - MXFP4 pair-packed last dimension and E8M0 scale shape;
  - no NPU provider declares an `out=` contract.

- [ ] **Step 2: Run NPU tests and verify RED**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_norm_quant_providers.py -k 'npu or pr or mx'
  ```

  Expected: failures because NPU providers and capability gates do not exist.

- [ ] **Step 3: Implement lazy runtime and schema gates**

  Import `torch_npu` only inside resolver methods. Require device name
  `Ascend950PR`, the callable runtime symbol, and the required schema fields:

  ```text
  npu_rms_norm_quant: dst_dtype, one output
  npu_add_rms_norm_quant: dst_type, three outputs
  npu_add_rms_norm_dynamic_quant: y_dtype, five outputs
  npu_rms_norm_dynamic_mx_quant: scale_alg, round_mode, dst_type, three outputs
  npu_add_rms_norm_dynamic_mx_quant: scale_alg, round_mode, dst_type, four outputs
  ```

  Cache a real tiny native probe for each
  `(device, torch_npu_version, native_op, dtype_code)` and expose its failure
  as `capability_error`. Never register a failed provider as formal.

- [ ] **Step 4: Implement plain FP8 native calls**

  Keep quantization inside `_execute_core_operator`:

  ```python
  y = torch_npu.npu_rms_norm_quant(
      x, gamma, beta, scale, offset, eps,
      dst_dtype=torch.float8_e4m3fn,
  )
  y1, y2, x_out = torch_npu.npu_add_rms_norm_quant(
      x1, x2, gamma, beta, scales1, zero_points1, eps,
      dst_type=292,
  )
  y1, y2, x_out, scale1, scale2 = (
      torch_npu.npu_add_rms_norm_dynamic_quant(
          x1, x2, gamma, beta, smooth_scale1, smooth_scale2, eps,
          output_mask=[True, False],
          y_dtype=torch.float8_e4m3fn,
      )
  )
  ```

  Preserve the exact remote schema's positional order when the tiny probe
  reveals additional optional parameters; encode that verified order in the
  unit fixture before implementation. Return the complete native tuple.

- [ ] **Step 5: Implement native MX calls and validate physical layouts**

  Calls:

  ```python
  torch_npu.npu_rms_norm_dynamic_mx_quant(
      x, gamma, beta, eps,
      scale_alg=0, round_mode="rint", dst_type=292_or_296,
  )
  torch_npu.npu_add_rms_norm_dynamic_mx_quant(
      x1, x2, gamma, beta, eps,
      scale_alg=0, round_mode="rint", dst_type=292_or_296,
  )
  ```

  MXFP8 output is E4M3 `[tokens, hidden]`; MXFP4 output is uint8
  `[tokens, hidden // 2]`; physical E8M0 scale is uint8
  `[tokens, ceil(hidden / 64), 2]`. Require `hidden % 64 == 0`.
  Empty `rstd` is legal and counts zero bytes.

- [ ] **Step 6: Implement NPU auxiliary correctness**

  Compare native auxiliary outputs against untimed standalone/native
  quantization of the FP32-accumulated RMSNorm reference. Separately validate:

  - static/dynamic plain FP8 dequantized primary;
  - dynamic plain FP32 scale shape;
  - `x_out` versus BF16 `x1 + x2`;
  - MXFP8 E8M0 group-32 scale metadata;
  - both E2M1 values in every MXFP4 packed byte;
  - output/scale shapes, dtypes, contiguity, and disabled empty outputs.

- [ ] **Step 7: Run all provider tests**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_norm_quant_providers.py
  python -m compileall -q tests/ops/norm_quant
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add tests/ops/norm_quant tests/ut/ops/test_norm_quant_providers.py
  git commit -s -m "feat(ops): add 950PR norm quant providers"
  ```

### Task 5: Formal CLI, checkpoint CSV, and continuous plots

**Files:**

- Create: `tests/ops/tests/test_norm_quant.py`
- Modify: `tests/ops/tests/__init__.py`
- Modify: `tests/ops/run_tests.sh`
- Modify: `tests/ut/ops/test_operator_curve_entries.py`
- Modify: `tests/ut/ops/test_operator_dispatcher.py`

**Interfaces:**

- Consumes: provider factory, Framework V2 eager/graph/profile APIs,
  `build_fresh_iteration_plan`, `finalize_curve_coverage`, and complete
  performance provenance.
- Produces CLI:

  ```text
  --mode accuracy|curve|profile
  --precision fp8|mxfp8|mxfp4
  --dispatch-mode eager|graph|both
  --variants rms_norm_quant add_rms_norm_quant
             add_rms_norm_dynamic_quant rms_norm_dynamic_mx_quant
             add_rms_norm_dynamic_mx_quant
  --tokens 1 2 4 8 16 32 64 128 256 512 1024 2048 4096
  --hidden-sizes 4096 7168 8192
  --warmup W --iterations I --repeats R
  --device auto|cuda[:N]|npu[:N]
  --result-dir DIR
  --quick --shard-index N --num-shards M --no-plot
  ```

- [ ] **Step 1: Add failing entry and dispatcher tests**

  Fake the framework and providers. Assert that:

  - correctness runs before either timing mode;
  - eager and graph use identical W/I/S/R and shape values;
  - every mode/provider owns a distinct checkpoint CSV;
  - capture failure becomes `unsupported_graph_capture`;
  - capability failure becomes `unsupported` with the original error;
  - H20 best-envelope rows retain the winning provider per point;
  - no profiler latency field is accepted as formal latency;
  - `all --precision fp8` dispatches Linear, GroupGemm, and NormQuant;
  - MX precisions reject CUDA and dispatch NormQuant on NPU;
  - default `all` without quantized precision remains unchanged.

- [ ] **Step 2: Run entry/dispatcher tests and verify RED**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_curve_entries.py -k norm_quant
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_dispatcher.py -k 'norm_quant or quantized'
  ```

  Expected: failures because the entry and dispatcher family do not exist.

- [ ] **Step 3: Implement formal curve execution and checkpointing**

  For each selected variant/provider/shape:

  1. build deterministic source data;
  2. run full auxiliary correctness;
  3. build one memory-bounded W/I plan;
  4. run `eager_direct`;
  5. run `captured_chain`;
  6. append and rewrite that mode/provider's checkpoint CSV immediately;
  7. finalize coverage only when every selected point has a terminal row.

  CSV includes variant, native op, precision, device model, provider, tokens,
  hidden, logical/physical bytes, residual semantics, latency, effective
  bandwidth, status/error, complete Framework provenance, and capability/
  graph status.

- [ ] **Step 4: Implement profiler mode**

  Profile only:

  ```text
  tokens=1,128,4096
  hidden=7168
  ```

  Call `run_core_operator_profile_test_v2`; write a diagnostic JSON manifest
  containing logical invocation count, expected fused-kernel count, trace
  directory, and `profiler_is_diagnostic=true`. Do not write latency into
  eager or graph CSV.

- [ ] **Step 5: Implement plots**

  Produce one latency and one effective-bandwidth plot per variant. Plot
  graph as a solid line and eager as a dashed line; use a continuous numeric
  x-axis with every actual token/hidden point connected. Do not use a broken
  axis. Unsupported cells remain absent from the data line and appear in a
  nearby annotation/table, not as zero.

- [ ] **Step 6: Wire `run_tests.sh`**

  Add `norm_quant` to usage and dispatch:

  ```bash
  python3 tests/test_norm_quant.py \
    --mode curve \
    --precision "$formal_precision" \
    --dispatch-mode both
  ```

  Quantized `all` includes the family; the unqualified default matrix does
  not. Require `--warmup >= 2`, positive iterations/repeats, and preserve
  `TASK_QUEUE_ENABLE` provenance.

- [ ] **Step 7: Run local entry and dispatcher regressions**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_curve_entries.py \
    tests/ut/ops/test_operator_dispatcher.py
  bash -n tests/ops/run_tests.sh
  bash tests/ops/run_tests.sh --formal \
    --operator norm_quant --precision fp8 --device cuda:0 \
    --output-dir /tmp/norm-quant --quick --dry-run
  PYTHONPATH=tests/ops python tests/ops/tests/test_norm_quant.py --help
  ```

- [ ] **Step 8: Run the complete device-independent regression set**

  Run:

  ```bash
  python -m pytest -q --confcutdir=tests/ut/ops \
    tests/ut/ops/test_operator_test_framework.py \
    tests/ut/ops/test_norm_quant_providers.py \
    tests/ut/ops/test_operator_curve_entries.py \
    tests/ut/ops/test_operator_dispatcher.py
  git diff --check
  ```

- [ ] **Step 9: Commit**

  ```bash
  git add tests/ops/tests/test_norm_quant.py \
    tests/ops/tests/__init__.py tests/ops/run_tests.sh \
    tests/ut/ops/test_operator_curve_entries.py \
    tests/ut/ops/test_operator_dispatcher.py
  git commit -s -m "feat(ops): add formal norm quant curves"
  ```

### Task 6: H20 correctness, eager/graph curves, and diagnostic profiles

**Files:**

- Sync source from the implementation worktree to:
  `/root/vllm-ascend-norm-quant`
- Pull artifacts to:
  `/Users/yucheng/Documents/2027/A5/fresh_ops_comparison_20260723/norm_quant_20260730/h20`

**Interfaces:**

- Consumes: Task 5 formal CLI.
- Produces: provider-specific CSV/PNG, H20 best-envelope CSV/PNG, raw CUDA
  profiles, command log, hardware/version manifest.

- [ ] **Step 1: Verify the idle H20 device and runtime**

  Run read-only checks through `ssh -p 7890 root@localhost`. Record
  `nvidia-smi`, PyTorch/CUDA/vLLM/FlashInfer versions, the exact GPU selected,
  and require its name to be `NVIDIA H20-3e`.

- [ ] **Step 2: Sync local code without remote editing**

  Use `rsync` from the implementation worktree to
  `/root/vllm-ascend-norm-quant`, excluding `.git`, caches, results, and the
  unrelated dirty FlashAttention file.

- [ ] **Step 3: Run provider and graph smoke tests**

  Run the three raw vLLM ops and two FlashInfer static ops on a small
  `[8,4096]` payload, including CUDA Graph capture. Confirm expected
  output/scale dtypes and finite values.

- [ ] **Step 4: Run quick formal correctness and both timing modes**

  Run:

  ```bash
  tests/ops/run_tests.sh --formal \
    --operator norm_quant --device cuda:GPU_INDEX --precision fp8 \
    --output-dir /root/norm_quant_results/h20_quick \
    --quick --repeats 5 --stabilization-repeats 2
  ```

  Stop and fix code locally if correctness, capture, storage audit, or
  provenance fails.

- [ ] **Step 5: Run the full H20 FP8 matrix**

  Run both static challengers and the dynamic vLLM provider over the complete
  token and hidden sweeps. Keep provider-specific data and select the H20
  envelope per shape only after correctness.

- [ ] **Step 6: Collect H20 prepared-core profiles**

  Profile tokens `1,128,4096` at hidden `7168`. Confirm one fused kernel per
  logical invocation, no allocator/import/H2D/D2H in the active window, and
  capture CPU launch spacing/device gaps for diagnosis.

- [ ] **Step 7: Pull and checksum complete H20 artifacts**

  Pull CSV, PNG, JSON, logs, and complete profile directories. Save a sorted
  SHA256 manifest and verify every CSV has terminal status and complete
  protocol provenance.

### Task 7: 950PR capability, curves, and MTE/Vector profiles

**Files:**

- Sync source into the existing container
  `qwen35-122b-fp8-tp4-ep-20260625` under
  `/workspace/vllm-ascend-norm-quant`.
- Pull artifacts to:
  `/Users/yucheng/Documents/2027/A5/fresh_ops_comparison_20260723/norm_quant_20260730/950pr`

**Interfaces:**

- Consumes: Task 5 CLI and Task 4 capability gates.
- Produces: per-precision CSV/PNG, unsupported capability records, raw NPU
  profiles with MTE/Vector evidence, logs, and hardware/version manifest.

- [ ] **Step 1: Verify container, device, and free memory**

  Connect to `218.28.9.108:50228`, inspect the named container, record
  PyTorch/torch_npu/CANN versions, `npu-smi`, and require `Ascend950PR`.

- [ ] **Step 2: Sync local code into the container without remote editing**

  Transfer a local archive or rsync staging directory, then copy it into
  `/workspace/vllm-ascend-norm-quant`. Do not patch files with remote shell
  redirection.

- [ ] **Step 3: Run all real capability probes**

  Probe plain FP8, MXFP8, and MXFP4 native contracts. Preserve failures for
  `npu_add_rms_norm_quant` or
  `npu_add_rms_norm_dynamic_quant` as unsupported rows with the native error.

- [ ] **Step 4: Run device-side unit tests and quick formal curves**

  Run the provider/framework/dispatcher tests in the container, then quick
  FP8, MXFP8, and MXFP4 formal commands with both dispatch modes. Fix failures
  locally, resync, and repeat.

- [ ] **Step 5: Run complete supported 950PR matrices**

  Run all supported precision/variant combinations over the exact token and
  hidden sweeps with S2/R5 and the symmetric memory-bounded W/I plan.

- [ ] **Step 6: Collect prepared-core NPU profiles**

  Profile tokens `1,128,4096` at hidden `7168` using Level1
  `PipeUtilization`. Export the complete profiler tree needed to inspect
  device kernel, Vector, MTE, scalar, cycles, active bandwidth, frequency,
  power, and temperature.

- [ ] **Step 7: Pull and checksum complete 950PR artifacts**

  Pull every CSV, PNG, JSON, log, capability error, and profile directory.
  Generate a sorted SHA256 manifest and validate terminal statuses and
  provenance.

### Task 8: Cross-platform plots and findings report

**Files:**

- Create:
  `/Users/yucheng/Documents/2027/A5/fresh_ops_comparison_20260723/norm_quant_20260730/plot_norm_quant_comparison.py`
- Create:
  `/Users/yucheng/Documents/2027/A5/fresh_ops_comparison_20260723/norm_quant_20260730/norm_quant_h20_950pr_comparison.png`
- Create:
  `/Users/yucheng/Documents/2027/A5/fresh_ops_comparison_20260723/norm_quant_20260730/README.md`

**Interfaces:**

- Consumes: checksummed H20 and 950PR terminal CSV/profile artifacts.
- Produces: continuous comparison curves and an evidence-backed Markdown
  analysis.

- [ ] **Step 1: Add artifact validation before plotting**

  The plotting script rejects rows lacking correctness success, device model,
  provider, dispatch mode, W/I/S/R, repeat samples, or Event semantics. It
  rejects profiler-derived latency and mixed eager/graph series.

- [ ] **Step 2: Generate continuous comparison panels**

  Use continuous numeric token/hidden axes. For each variant, plot graph solid
  and eager dashed. H20 uses only the best successful provider per shape while
  retaining source provider labels. Plot 950PR FP8/MXFP8/MXFP4 as independent
  series; do not connect unsupported combinations through zero.

- [ ] **Step 3: Write the findings report**

  Document:

  - exact hardware/software and commands;
  - supported/unsupported matrix and original capability errors;
  - graph-versus-eager gap at small and large tokens;
  - winning H20 provider per shape;
  - kernel count and allocator/import evidence;
  - NPU Vector/MTE evidence and whether the bottleneck is launch, bandwidth,
    or vector work;
  - limitations and any profile evidence still missing.

- [ ] **Step 4: Verify deliverables**

  Run the plotting script from a clean process, open the PNG, check that
  axes/labels/legends are readable and lines are continuous, validate every
  Markdown link, and compare SHA256 manifests to the pulled artifacts.
