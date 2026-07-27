# Operator Curve Benchmark Consolidation

Date: 2026-07-27

## Goal

Make `tests/ops` able to regenerate the formal H20, Ascend A3, and Ascend A5
operator-curve CSVs with one timing protocol:

- allocate `warmup + iterations` independent invocation payloads before timing;
- measure only `_execute_core_operator` with device events;
- retain returned outputs until the repeat ends;
- verify device-storage independence;
- run multiple repeats and report the median of repeat means;
- keep provider, shape, protocol, and storage provenance in every result.

Plotting remains a local post-processing step. Remote benchmark entry points
must not require matplotlib.

## Scope

Formal curve operators:

- Add BF16
- Linear FP16/BF16
- RMSNorm BF16
- FlashAttention FP16/BF16
- GroupGemm BF16/INT8
- PagedAttention BF16
- RecurrentGatedDeltaRule BF16

Formal providers:

| Operator | H20 | Ascend A3/A5 |
|---|---|---|
| Add | `torch.add(..., out=...)` | `torch.add` |
| Linear | `torch.mm(..., out=...)` | `torch.nn.functional.linear` |
| RMSNorm | vLLM RMSNorm `out=` | `torch_npu.npu_rms_norm` |
| FlashAttention | forced PyTorch SDPA Flash and external flash-attn | `npu_fused_infer_attention_score` |
| GroupGemm BF16 | balanced `torch.bmm`/cuBLAS | `npu_grouped_matmul` |
| GroupGemm INT8 | vLLM CUTLASS scaled-mm | `npu_grouped_matmul` |
| PagedAttention | FlashInfer FA2, block size 128 | `npu_fused_infer_attention_score`, block size 128 |
| RecurrentGatedDeltaRule | vLLM FLA direct-out kernel | CANN builtin |

Provider fallbacks may remain available for correctness/debugging but must never
be selected silently for a formal curve.

## Framework Design

Keep the existing operator contract:

```python
_prepare_data_for_core_operator(data, device, precision, implementation)
_execute_core_operator(prepared_data, implementation)
```

Do not add a second prepared-invocation abstraction. Extend
`OperatorTestFramework.run_core_operator_performance_test_v2` with:

- `num_repeats` (default `1` for compatibility);
- median aggregation of repeat event means;
- `repeat_samples_ms` and a stable protocol identifier;
- input/returned-output storage counts and pointer counts;
- strict argument validation;
- explicit cleanup after references are released.

Each repeat independently:

1. prepares `W + I` payloads;
2. optionally verifies that no device storage is shared across payloads;
3. runs the first `W` payloads as warmup;
4. synchronizes the selected device;
5. records `start`, launches the remaining `I` payloads, records `end`, and
   synchronizes `end`;
6. retains all outputs until verification completes;
7. verifies returned-output independence;
8. releases payload/output references before cache cleanup.

The method remains responsible for throughput/TOPS/bandwidth calculation.
Shape matrices, provider selection, correctness policy, sharding, checkpoint
CSV paths, and plotting stay outside the Framework.

## Result and CSV Contract

`PerformanceMetrics` retains its existing public fields and adds:

- `protocol_version`
- `repeat_samples_ms`
- `aggregation`
- `repeats`
- `preallocated_invocations_per_repeat`
- `input_storage_sets_verified`
- `input_storage_ptr_count`
- `output_storage_sets_verified`
- `output_storage_ptr_count`
- `output_storage_policy`
- `timed_region`

One helper returns the flat provenance dictionary used by every curve CSV.
Legacy field names may be emitted as aliases while existing plot loaders are
migrated.

## Operator Responsibilities

An operator/provider must:

- allocate or copy all input/state/workspace/output storage in
  `_prepare_data_for_core_operator`;
- keep import, backend selection, layout conversion, and FlashInfer planning
  outside the timed method;
- perform only the intended native launch(es) in `_execute_core_operator`;
- identify its provider and output semantics;
- use explicit `copy=True`, `clone`, or fresh allocation when the source may
  already be on the target device.

Providers with an `out=` API return that preallocated output. Providers without
one may allocate during launch, but Framework retains and verifies every return
address. The CSV must distinguish those policies.

## Curve Entry Points

Reuse the existing per-operator `tests/test_*.py` entry points. Their formal
curve modes own only:

- argument parsing;
- shape generation;
- provider selection;
- calls to the Framework;
- correctness checks;
- checkpoint CSV writing.

`run_tests.sh` remains a thin dispatcher. It must expose a non-interactive
formal-curve path suitable for remote automation.

## Subtraction

After the provider code and shape matrices are represented by the operator
modules and test entry points, remove:

- `benchmark_h20_paged_backends.py`;
- `benchmark_h20_targeted_rerun.py`;
- `benchmark_h20_preallocated_ops.py`;
- `benchmark_ascend_910c_ops.py`;
- FlashAttention's private reused-input Event benchmark;
- PagedAttention's private repeat/median/storage aggregation;
- RecurrentGatedDeltaRule's private storage audit and repeat aggregation;
- duplicated per-curve provenance construction where the Framework helper
  supplies the same fields.

Historical CSVs, logs, environment manifests, and plots are retained as
immutable benchmark evidence.

## Error Handling

- Formal mode fails if the requested fused provider is unavailable.
- Formal mode fails on any failed curve point and retains its checkpoint CSV.
- Invalid W/I/R values fail before allocation.
- Storage reuse fails the repeat before its result is accepted.
- Memory preflight produces an actionable error for large strict-preallocation
  points; it does not silently switch to reused inputs.
- Old V2 callers keep `num_repeats=1`; formal callers explicitly request the
  required repeat count and verification.

## Test Strategy

Local CPU unit tests:

- W/I/R call counts and fresh payload creation;
- median-of-repeat-means aggregation;
- invalid argument rejection;
- input-storage reuse detection;
- returned-output reuse detection;
- output retention through verification;
- flat provenance serialization;
- compatibility with a one-repeat legacy call.

Static tests:

- all `tests/ops` Python files compile;
- formal provider names resolve deterministically;
- formal PA rejects fallback providers and uses block size 128;
- curve CSV schemas contain required protocol fields.

Remote validation:

1. sync only local modified files to a separate remote staging tree;
2. run a one-shape W1/I2/R2 smoke test on H20, A3, and A5;
3. run representative small and large points for each available formal
   provider;
4. pull logs and CSVs to a new local validation directory;
5. verify status, protocol fields, storage counts, provider identity, positive
   finite latency, and expected row counts;
6. only then run the bounded formal curve matrices requested for final plots.

