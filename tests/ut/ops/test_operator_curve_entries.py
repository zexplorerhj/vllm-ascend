#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import csv
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from tests.test_add import AddTestSuite  # noqa: E402
from tests.test_flash_attention import FlashAttentionTestSuite  # noqa: E402
from tests.test_groupgemm import GroupGemmTestSuite  # noqa: E402
from tests.test_linear import LinearTestSuite  # noqa: E402
from tests.test_paged_attention import PagedAttentionTestSuite  # noqa: E402
from tests.test_rmsnorm import (  # noqa: E402
    RMSNormTestSuite,
    estimate_rmsnorm_fresh_bytes,
)
import recurrent_gated_delta_rule.benchmark as recurrent_benchmark  # noqa: E402
import tests.test_recurrent_gated_delta_rule  # noqa: E402,F401


PROVENANCE = {
    "framework_api": (
        "OperatorTestFramework.run_core_operator_performance_test_v2"
    ),
    "protocol_version": "operator-test-framework-v2-fresh-v5",
    "warmup": 1,
    "iterations": 2,
    "repeats": 3,
    "stabilization_repeats": 0,
    "stabilization_repeat_samples_ms": "[]",
    "repeat_samples_ms": "[1.5, 1.0, 2.0]",
    "event_window_samples_ms": "[3.0, 2.0, 4.0]",
    "event_window_min_ms": 2.0,
    "event_window_median_ms": 3.0,
    "event_window_max_ms": 4.0,
    "repeat_min_ms": 1.0,
    "repeat_median_ms": 1.5,
    "repeat_max_ms": 2.0,
    "repeat_p25_ms": 1.25,
    "repeat_p75_ms": 1.75,
    "repeat_iqr_pct": 100.0 / 3.0,
    "repeat_spread_pct": 100.0,
    "aggregation": "median_of_repeat_means",
    "preallocated_invocations_per_repeat": 3,
    "input_reuse_within_repeat": False,
    "input_storage_sets_verified": 3,
    "input_storage_ptr_count": 3,
    "input_output_storage_disjoint": True,
    "output_storage_sets_verified": 3,
    "output_storage_ptr_count": 3,
    "output_tensor_count": 3,
    "output_unique_storages_per_set": 1,
    "preallocated_output_aliases_verified": 1,
    "preallocated_output_sets_verified": 1,
    "output_tensors_per_set": 1,
    "output_allocation_mode": (
        "preallocated_output_contract_with_warmup_alias_probe"
    ),
    "output_allocation_policy": (
        "preallocated_output_contract_with_warmup_alias_probe"
    ),
    "output_storage_policy": "retained_until_repeat_end",
    "timing_method": "device_event",
    "timing_semantics": (
        "device elapsed time; includes stream-idle gaps between start/end "
        "events caused by host dispatch"
    ),
    "timed_output_capture_policy": (
        "preallocated_output_contract_no_timed_return_capture"
    ),
    "preallocated_output_contract": "declared_phase_invariant_out",
    "output_alias_verification_scope": "warmup_returns_only",
    "preallocated_output_contract_invocations_per_repeat": 3,
    "output_verification_replay_invocations_per_repeat": 0,
    "total_operator_calls_per_repeat": 3,
    "workspace_allocation_policy": "not_audited",
    "dispatch_loop_policy": "python_direct_prepared_payload_loop",
    "device_stabilization_policy": "none",
    "device_stabilization_timed": False,
    "stabilization_operator_calls": 0,
    "task_queue_enable": "not_applicable",
    "timed_region": (
        "Python direct prepared-payload loop of _execute_core_operator; "
        "prepare excluded; timed Python returns discarded under declared "
        "out contract"
    ),
}


class _FakeFramework:

    def __init__(self, result_dir, fail=False):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.fail = fail
        self.calls = []

    def register_operator(self, operator):
        del operator

    def run_core_operator_performance_test_v2(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("synthetic point failure")
        return SimpleNamespace(
            avg_time_ms=1.5,
            throughput=2500.0,
            bandwidth_gb_s=123.0,
        )

    def performance_provenance(self, metrics):
        del metrics
        return dict(PROVENANCE)


class _FakeOperator:
    CUDA_BF16_IMPLEMENTATION = "cuda_bmm_balanced_grouped_mm_jagged_bf16"
    CUDA_INT8_IMPLEMENTATION = "cuda_vllm_cutlass_scaled_mm_bf16"
    CUDA_INT8_FALLBACK_IMPLEMENTATION = "cuda_raw_int32_debug_fallback"

    def __init__(self, providers):
        self.providers = list(providers)
        self.operator_name = "fake"

    def get_formal_implementations(self, device):
        del device
        return list(self.providers)

    def generate_test_data(self, **kwargs):
        metadata = {
            "seed": 17,
            "num_blocks": kwargs.get("num_blocks", 10000),
            "page_pool_policy": "fixed",
        }
        return {
            "metadata": metadata,
            "group_list": torch.tensor(
                [kwargs.get("seq_len", 1)], dtype=torch.int64
            ),
            **kwargs,
        }

    def generate_latency_test_data(self, **kwargs):
        return self.generate_test_data(**kwargs)

    def calculate_tflops(self, data, time_ms, mode):
        del data, time_ms, mode
        return 42.0


def _assert_formal_call(call, warmup, iterations, repeats, implementation):
    assert call["implementation"] == implementation
    assert call["num_warmup"] == warmup
    assert call["num_iterations"] == iterations
    assert call["num_repeats"] == repeats
    assert call["retain_outputs"] is True
    assert call["verify_independent_storage"] is True


def _assert_success_rows(rows):
    assert rows
    for row in rows:
        assert row["status"] in ("ok", "success")
        for key, value in PROVENANCE.items():
            assert row[key] == value


def _adaptive_provenance(framework):
    call = framework.calls[-1]
    warmup = call["num_warmup"]
    iterations = call["num_iterations"]
    repeats = call["num_repeats"]
    repeat_samples = [1.5, 1.0, 2.0][:repeats]
    provenance = dict(PROVENANCE)
    provenance.update(
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
        repeat_samples_ms=str(repeat_samples),
        event_window_samples_ms=str([
            sample * iterations for sample in repeat_samples
        ]),
        event_window_min_ms=min(repeat_samples) * iterations,
        event_window_median_ms=sorted(repeat_samples)[
            len(repeat_samples) // 2
        ] * iterations,
        event_window_max_ms=max(repeat_samples) * iterations,
        repeat_min_ms=min(repeat_samples),
        repeat_median_ms=sorted(repeat_samples)[
            len(repeat_samples) // 2
        ],
        repeat_max_ms=max(repeat_samples),
        preallocated_invocations_per_repeat=warmup + iterations,
        input_storage_sets_verified=warmup + iterations,
        output_storage_sets_verified=warmup + iterations,
        output_tensor_count=warmup + iterations,
        preallocated_output_contract_invocations_per_repeat=(
            warmup + iterations
        ),
        total_operator_calls_per_repeat=warmup + iterations,
    )
    return provenance


def _assert_coverage(
    rows,
    *,
    mode,
    total,
    requested,
    selected,
    complete,
    source="custom",
    selection_complete=False,
):
    assert rows
    for row in rows:
        assert row["selection_mode"] == mode
        assert row["shape_matrix_source"] == source
        assert row["coverage_total_formal_points"] == total
        assert row["coverage_total_requested_points"] == requested
        assert row["coverage_selected_points"] == selected
        assert (
            row["selection_covers_full_formal_matrix"]
            is selection_complete
        )
        assert row["coverage_complete"] is complete


def test_add_formal_point_uses_one_v2_call_and_provenance(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = AddTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_torch_add_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = suite.run_bandwidth_test(
        sizes=[8],
        device="cuda:0",
        num_warmup=5,
        num_iterations=20,
        num_repeats=3,
        plot_results=False,
    )

    assert len(framework.calls) == 1
    _assert_formal_call(
        framework.calls[0], 5, 20, 3, "cuda_torch_add_out"
    )
    _assert_success_rows(result["rows"])
    _assert_coverage(
        result["rows"],
        mode="custom_shape_matrix",
        total=16,
        requested=1,
        selected=1,
        complete=False,
    )


def test_linear_formal_point_is_bias_free_and_uses_one_v2_call(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = LinearTestSuite(precision="bf16")
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_torch_mm_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = suite.run_tflops_test(
        sizes=[16],
        device="cuda:0",
        num_warmup=10,
        num_iterations=50,
        num_repeats=3,
        plot_results=False,
    )

    assert len(framework.calls) == 1
    _assert_formal_call(
        framework.calls[0], 10, 50, 3, "cuda_torch_mm_out"
    )
    assert framework.calls[0]["data"]["bias"] is False
    _assert_success_rows(result["rows"])
    _assert_coverage(
        result["rows"],
        mode="custom_shape_matrix",
        total=31,
        requested=1,
        selected=1,
        complete=False,
    )


def test_add_auto_iterations_match_effective_csv_count(
    monkeypatch,
    tmp_path,
):
    framework = _FakeFramework(tmp_path)
    suite = AddTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_torch_add_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        framework,
        "performance_provenance",
        lambda metrics: _adaptive_provenance(framework),
    )

    result = suite.run_bandwidth_test(
        sizes=[4096],
        device="cuda:0",
        num_warmup=5,
        num_iterations=None,
        num_repeats=3,
        plot_results=False,
    )

    row = result["rows"][0]
    assert framework.calls[0]["num_iterations"] == 8192
    assert row["iteration_selection_policy"] == (
        "adaptive_unique_storage_soft_target"
    )
    assert row["iterations"] == row["effective_iterations"] == 8192
    assert row["estimated_unique_bytes_per_invocation"] == 6 * 4096


def test_linear_auto_iterations_preserve_large_shape_base(
    monkeypatch,
    tmp_path,
):
    framework = _FakeFramework(tmp_path)
    suite = LinearTestSuite(precision="bf16")
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_torch_mm_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        framework,
        "performance_provenance",
        lambda metrics: _adaptive_provenance(framework),
    )

    result = suite.run_tflops_test(
        sizes=[4096],
        device="cuda:0",
        num_warmup=10,
        num_iterations=None,
        num_repeats=3,
        plot_results=False,
    )

    row = result["rows"][0]
    assert framework.calls[0]["num_iterations"] == 50
    assert row["iterations"] == row["effective_iterations"] == 50
    assert row["fresh_storage_soft_target_overflow"] is True
    assert row["estimated_unique_bytes_per_invocation"] == 6 * 4096**2


def test_rmsnorm_each_formal_point_uses_one_v2_call(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = RMSNormTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_vllm_rms_norm_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = suite.run_bandwidth_test(
        sizes=[4096],
        hidden_sizes=[1024],
        target_total_elements=4096,
        device="cuda:0",
        num_warmup=10,
        num_iterations=50,
        num_repeats=3,
        plot_results=False,
    )

    assert len(framework.calls) == 2
    for call in framework.calls:
        _assert_formal_call(
            call, 10, 50, 3, "cuda_vllm_rms_norm_out"
        )
    rows = result["size_rows"] + result["hidden_rows"]
    _assert_success_rows(rows)
    _assert_coverage(
        rows,
        mode="custom_shape_matrix",
        total=16,
        requested=1,
        selected=1,
        complete=False,
    )


def test_rmsnorm_auto_iterations_use_cross_provider_byte_formula(
    monkeypatch,
    tmp_path,
):
    framework = _FakeFramework(tmp_path)
    suite = RMSNormTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_vllm_rms_norm_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        framework,
        "performance_provenance",
        lambda metrics: _adaptive_provenance(framework),
    )

    result = suite.run_bandwidth_test(
        sizes=[4096],
        hidden_sizes=[1024],
        target_total_elements=4096,
        device="cuda:0",
        num_warmup=10,
        num_iterations=None,
        num_repeats=3,
        plot_results=False,
        shard_index=0,
        num_shards=2,
    )

    assert len(framework.calls) == 1
    assert framework.calls[0]["num_iterations"] == 2048
    row = result["size_rows"][0]
    assert row["iterations"] == row["effective_iterations"] == 2048
    assert row["estimated_unique_bytes_per_invocation"] == (
        estimate_rmsnorm_fresh_bytes(4096, 4096)
    )
    assert row["estimated_unique_bytes_per_invocation"] == 24580


def test_flash_cuda_runs_both_formal_providers_once_per_point(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = FlashAttentionTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator([
        "cuda_sdpa_flash_attention",
        "cuda_flash_attn_func",
    ])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake")
    monkeypatch.setattr(
        suite,
        "_cuda_benchmark_data",
        lambda **kwargs: {
            "metadata": kwargs,
            "sparse_mode": 3 if kwargs["causal"] else 0,
        },
    )

    result = suite.run_tflops_test(
        precision="fp16",
        num_warmup=5,
        num_iterations=10,
        num_repeats=3,
        n_ctx_values=[32],
        head_dims=[64],
        causal_values=[True],
        device="cuda:0",
        plot_results=False,
    )

    assert len(framework.calls) == 2
    assert [call["implementation"] for call in framework.calls] == [
        "cuda_sdpa_flash_attention",
        "cuda_flash_attn_func",
    ]
    for call in framework.calls:
        _assert_formal_call(
            call, 5, 10, 3, call["implementation"]
        )
    _assert_success_rows(result["rows"])


@pytest.mark.parametrize(
    ("precision", "provider"),
    [
        ("bf16", "cuda_bmm_balanced_grouped_mm_jagged_bf16"),
        ("int8", "cuda_vllm_cutlass_scaled_mm_bf16"),
    ],
)
def test_groupgemm_formal_point_uses_i30_and_native_provider(
    monkeypatch, tmp_path, precision, provider
):
    framework = _FakeFramework(tmp_path)
    suite = GroupGemmTestSuite(
        precision=precision,
        num_experts=1,
        hidden_dim=16,
        out_channel=16,
    )
    suite.framework = framework
    suite.operator_test = _FakeOperator([provider])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = suite.run_tflops_test(
        seq_lens=[8],
        num_experts=1,
        hidden_dim=16,
        out_channel=16,
        device="cuda:0",
        num_warmup=10,
        num_iterations=30,
        num_repeats=3,
        plot_results=False,
    )

    assert len(framework.calls) == 1
    _assert_formal_call(framework.calls[0], 10, 30, 3, provider)
    _assert_success_rows(result["results"])
    _assert_coverage(
        result["results"],
        mode="custom_shape_matrix",
        total=10,
        requested=1,
        selected=1,
        complete=False,
    )


def test_paged_attention_two_matrices_make_one_v2_call_per_point(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = PagedAttentionTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_flashinfer_fa2"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake")

    result = suite.run_latency_plot_test(
        device="cuda:0",
        seqlen_start=128,
        seqlen_end=128,
        seqlen_step=128,
        seqlen_batch_size=1,
        batch_min=1,
        batch_max=1,
        batch_step=1,
        batch_curve_seq_lens=[128],
        num_warmup=5,
        repeats=3,
        num_blocks=10000,
        block_size=128,
        plot_results=False,
    )

    assert len(framework.calls) == 2
    for call in framework.calls:
        _assert_formal_call(call, 5, 30, 3, "cuda_flashinfer_fa2")
        assert call["data"]["block_size"] == 128
        assert call["data"]["num_heads"] == 8
        assert call["data"]["num_kv_heads"] == 1
        assert call["data"]["num_blocks"] == 10000
    _assert_success_rows(
        result["seqlen_rows"] + result["batch_rows"]
    )


def test_paged_attention_rejects_nonformal_block_size(tmp_path):
    suite = PagedAttentionTestSuite()
    suite.framework = _FakeFramework(tmp_path)
    suite.operator_test = _FakeOperator(["cuda_flashinfer_fa2"])

    with pytest.raises(ValueError, match="block_size=128"):
        suite.run_latency_plot_test(
            device="cuda:0",
            block_size=256,
            plot_results=False,
        )

    assert suite.framework.calls == []


def test_recurrent_point_uses_one_v2_call_and_framework_provenance(
    monkeypatch, tmp_path
):
    output = tmp_path / "recurrent.csv"
    framework = _FakeFramework(tmp_path)
    operator = _FakeOperator(["cuda_vllm_fla_direct_out"])
    monkeypatch.setattr(
        recurrent_benchmark,
        "resolve_device",
        lambda requested: ("cuda:0", "fake"),
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "RecurrentGatedDeltaRuleOperatorTest",
        lambda: operator,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "OperatorTestFramework",
        lambda result_dir: framework,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "environment",
        lambda *args: {},
    )

    exit_code = recurrent_benchmark.main([
        "--device", "cuda:0",
        "--modes", "decode",
        "--batches", "1",
        "--warmup", "5",
        "--iterations", "20",
        "--repeats", "3",
        "--output", str(output),
        "--skip-correctness",
    ])

    assert exit_code == 0
    assert len(framework.calls) == 1
    _assert_formal_call(
        framework.calls[0], 5, 20, 3, "cuda_vllm_fla_direct_out"
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["protocol_version"] == PROVENANCE["protocol_version"]
    assert rows[0]["preallocated_output_aliases_verified"] == "1"
    assert rows[0]["preallocated_output_sets_verified"] == "1"
    assert rows[0]["output_tensors_per_set"] == "1"
    assert rows[0]["output_allocation_mode"] == (
        "preallocated_output_contract_with_warmup_alias_probe"
    )
    assert rows[0]["output_allocation_policy"] == (
        "preallocated_output_contract_with_warmup_alias_probe"
    )
    assert rows[0]["repeat_min_ms"] == "1.0"
    assert rows[0]["repeat_median_ms"] == "1.5"
    assert rows[0]["repeat_max_ms"] == "2.0"
    assert rows[0]["repeat_spread_pct"] == "100.0"
    assert rows[0]["selection_mode"] == "custom_shape_matrix"
    assert rows[0]["shape_matrix_source"] == "custom"
    assert rows[0]["coverage_total_formal_points"] == "7"
    assert rows[0]["coverage_total_requested_points"] == "1"
    assert rows[0]["coverage_selected_points"] == "1"
    assert rows[0]["selection_covers_full_formal_matrix"] == "False"
    assert rows[0]["coverage_complete"] == "False"
    assert rows[0]["status"] == "ok"


def test_recurrent_canonical_bytes_use_cross_provider_worst_case():
    assert (
        recurrent_benchmark.canonical_recurrent_bytes_per_invocation(
            "decode",
            1,
        )
        == 1_058_924
    )
    assert (
        recurrent_benchmark.canonical_recurrent_bytes_per_invocation(
            "mtp3",
            128,
        )
        == 274_254_852
    )


def test_recurrent_auto_iterations_match_effective_csv_count(
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "recurrent-auto.csv"
    framework = _FakeFramework(tmp_path)
    operator = _FakeOperator(["cuda_vllm_fla_direct_out"])
    monkeypatch.setattr(
        recurrent_benchmark,
        "resolve_device",
        lambda requested: ("cuda:0", "fake"),
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "RecurrentGatedDeltaRuleOperatorTest",
        lambda: operator,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "OperatorTestFramework",
        lambda result_dir: framework,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "environment",
        lambda *args: {},
    )
    monkeypatch.setattr(
        framework,
        "performance_provenance",
        lambda metrics: _adaptive_provenance(framework),
    )

    exit_code = recurrent_benchmark.main([
        "--device", "cuda:0",
        "--modes", "decode",
        "--batches", "1",
        "--warmup", "5",
        "--repeats", "3",
        "--output", str(output),
        "--skip-correctness",
    ])

    assert exit_code == 0
    assert framework.calls[0]["num_iterations"] == 2048
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["iterations"] == row["effective_iterations"] == "2048"
    assert row["estimated_unique_bytes_per_invocation"] == "1058924"


@pytest.mark.parametrize("failure_kind", ["returned_failure", "exception"])
def test_recurrent_correctness_error_checkpoints_mode_and_continues(
    monkeypatch, tmp_path, failure_kind
):
    output = tmp_path / f"recurrent-{failure_kind}.csv"
    framework = _FakeFramework(tmp_path)

    class CorrectnessOperator(_FakeOperator):

        def correctness(self, mode, device, provider):
            del device, provider
            if mode == "decode":
                if failure_kind == "exception":
                    raise RuntimeError("synthetic correctness exception")
                return {
                    "cosine": 0.1,
                    "max_abs": 3.0,
                    "mean_abs": 2.0,
                    "state_cosine": 0.2,
                    "state_max_abs": 4.0,
                    "state_mean_abs": 2.5,
                    "passed": False,
                }
            return {
                "cosine": 1.0,
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "state_cosine": 1.0,
                "state_max_abs": 0.0,
                "state_mean_abs": 0.0,
                "passed": True,
            }

    operator = CorrectnessOperator(["cuda_vllm_fla_direct_out"])
    monkeypatch.setattr(
        recurrent_benchmark,
        "resolve_device",
        lambda requested: ("cuda:0", "fake"),
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "RecurrentGatedDeltaRuleOperatorTest",
        lambda: operator,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "OperatorTestFramework",
        lambda result_dir: framework,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "environment",
        lambda *args: {},
    )
    monkeypatch.setattr(recurrent_benchmark, "cleanup", lambda device: None)
    checkpoint_snapshots = []
    original_write_csv = recurrent_benchmark.write_csv

    def record_checkpoint(path, rows):
        checkpoint_snapshots.append([dict(row) for row in rows])
        original_write_csv(path, rows)

    monkeypatch.setattr(
        recurrent_benchmark,
        "write_csv",
        record_checkpoint,
    )

    exit_code = recurrent_benchmark.main([
        "--device", "cuda:0",
        "--modes", "decode,mtp3",
        "--batches", "1",
        "--output", str(output),
    ])

    assert exit_code == 1
    assert len(framework.calls) == 1
    assert framework.calls[0]["data"]["mode"] == "mtp3"
    assert [len(snapshot) for snapshot in checkpoint_snapshots] == [1, 2]
    assert checkpoint_snapshots[0][0]["mode"] == "decode"
    assert checkpoint_snapshots[0][0]["status"] == "error"
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in rows] == ["decode", "mtp3"]
    assert rows[0]["status"] == "error"
    assert "correctness" in rows[0]["error"]
    assert rows[1]["status"] == "ok"
    assert rows[1]["protocol_version"] == PROVENANCE["protocol_version"]


def test_failed_point_is_checkpointed_and_curve_fails(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path, fail=True)
    suite = AddTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_torch_add_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="formal Add point"):
        suite.run_bandwidth_test(
            sizes=[8],
            device="cuda:0",
            plot_results=False,
        )

    csv_paths = list(tmp_path.glob("add_bandwidth_*.csv"))
    assert len(csv_paths) == 1
    with csv_paths[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "error"
    assert "synthetic point failure" in rows[0]["error"]


def test_add_quick_keeps_first_formal_point_identity(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = AddTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_torch_add_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = suite.run_bandwidth_test(
        sizes=[8, 16, 32],
        device="cuda:0",
        quick=True,
        shard_index=0,
        num_shards=1,
        plot_results=False,
    )

    assert [call["data"]["shape"] for call in framework.calls] == [(8,)]
    assert [row["point_index"] for row in result["rows"]] == [0]
    _assert_coverage(
        result["rows"],
        mode="quick_shape_subset",
        total=16,
        requested=3,
        selected=1,
        complete=False,
    )


def test_linear_shard_keeps_global_formal_point_identity(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = LinearTestSuite(precision="bf16")
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_torch_mm_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = suite.run_tflops_test(
        sizes=[8, 16, 32],
        device="cuda:0",
        shard_index=1,
        num_shards=2,
        plot_results=False,
    )

    assert [call["data"]["batch_size"] for call in framework.calls] == [16]
    assert [row["point_index"] for row in result["rows"]] == [1]
    assert result["rows"][0]["shard_index"] == 1
    assert result["rows"][0]["num_shards"] == 2
    _assert_coverage(
        result["rows"],
        mode="custom_shape_matrix",
        total=31,
        requested=3,
        selected=1,
        complete=False,
    )


def test_rmsnorm_quick_keeps_one_point_per_formal_matrix(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = RMSNormTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_vllm_rms_norm_out"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = suite.run_bandwidth_test(
        sizes=[4096, 8192],
        hidden_sizes=[1024, 2048],
        target_total_elements=8192,
        device="cuda:0",
        quick=True,
        plot_results=False,
    )

    assert len(framework.calls) == 2
    rows = result["size_rows"] + result["hidden_rows"]
    assert [row["point_index"] for row in rows] == [0, 2]
    _assert_coverage(
        rows,
        mode="quick_shape_subset",
        total=16,
        requested=2,
        selected=1,
        complete=False,
    )


def test_flash_quick_keeps_one_shape_for_each_formal_provider(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = FlashAttentionTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator([
        "cuda_sdpa_flash_attention",
        "cuda_flash_attn_func",
    ])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake")
    monkeypatch.setattr(
        suite,
        "_cuda_benchmark_data",
        lambda **kwargs: {
            "metadata": kwargs,
            "sparse_mode": 3 if kwargs["causal"] else 0,
        },
    )

    result = suite.run_tflops_test(
        precision="fp16",
        n_ctx_values=[32, 64],
        head_dims=[64, 128],
        causal_values=[True, False],
        device="cuda:0",
        quick=True,
        plot_results=False,
    )

    assert [call["implementation"] for call in framework.calls] == [
        "cuda_sdpa_flash_attention",
        "cuda_sdpa_flash_attention",
        "cuda_sdpa_flash_attention",
        "cuda_sdpa_flash_attention",
        "cuda_flash_attn_func",
        "cuda_flash_attn_func",
        "cuda_flash_attn_func",
        "cuda_flash_attn_func",
    ]
    assert {
        (
            call["data"]["metadata"]["head_dim"],
            call["data"]["metadata"]["causal"],
        )
        for call in framework.calls
    } == {
        (64, True),
        (64, False),
        (128, True),
        (128, False),
    }
    assert [row["point_index"] for row in result["rows"]] == [
        0, 2, 4, 6, 8, 10, 12, 14,
    ]


def test_groupgemm_shard_keeps_global_formal_point_identity(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = GroupGemmTestSuite(
        precision="bf16",
        num_experts=1,
        hidden_dim=16,
        out_channel=16,
    )
    suite.framework = framework
    provider = "cuda_bmm_balanced_grouped_mm_jagged_bf16"
    suite.operator_test = _FakeOperator([provider])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = suite.run_tflops_test(
        seq_lens=[8, 16, 32],
        num_experts=1,
        hidden_dim=16,
        out_channel=16,
        device="cuda:0",
        shard_index=1,
        num_shards=2,
        plot_results=False,
    )

    assert [call["data"]["seq_len"] for call in framework.calls] == [16]
    assert [row["point_index"] for row in result["results"]] == [1]
    _assert_coverage(
        result["results"],
        mode="custom_shape_matrix",
        total=10,
        requested=3,
        selected=1,
        complete=False,
    )


def test_paged_attention_quick_keeps_one_point_per_formal_matrix(
    monkeypatch, tmp_path
):
    framework = _FakeFramework(tmp_path)
    suite = PagedAttentionTestSuite()
    suite.framework = framework
    suite.operator_test = _FakeOperator(["cuda_flashinfer_fa2"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake")

    result = suite.run_latency_plot_test(
        device="cuda:0",
        seqlen_start=128,
        seqlen_end=256,
        seqlen_step=128,
        seqlen_batch_size=1,
        batch_min=1,
        batch_max=2,
        batch_step=1,
        batch_curve_seq_lens=[128, 256],
        num_blocks=10000,
        block_size=128,
        quick=True,
        plot_results=False,
    )

    assert len(framework.calls) == 3
    rows = result["seqlen_rows"] + result["batch_rows"]
    assert [row["point_index"] for row in rows] == [0, 2, 4]
    assert [
        call["data"]["max_seq_len"] for call in framework.calls[1:]
    ] == [128, 256]


def test_recurrent_quick_keeps_first_batch_of_each_mode(
    monkeypatch, tmp_path
):
    output = tmp_path / "recurrent-quick.csv"
    framework = _FakeFramework(tmp_path)
    operator = _FakeOperator(["cuda_vllm_fla_direct_out"])
    monkeypatch.setattr(
        recurrent_benchmark,
        "resolve_device",
        lambda requested: ("cuda:0", "fake"),
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "RecurrentGatedDeltaRuleOperatorTest",
        lambda: operator,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "OperatorTestFramework",
        lambda result_dir: framework,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "environment",
        lambda *args: {},
    )

    exit_code = recurrent_benchmark.main([
        "--device", "cuda:0",
        "--modes", "decode,mtp3",
        "--batches", "1,4",
        "--output", str(output),
        "--skip-correctness",
        "--quick",
    ])

    assert exit_code == 0
    assert [call["data"]["mode"] for call in framework.calls] == [
        "decode",
        "mtp3",
    ]
    assert [call["data"]["batch_size"] for call in framework.calls] == [1, 1]
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["point_index"]) for row in rows] == [0, 2]
    assert {
        (
            row["selection_mode"],
            row["coverage_total_formal_points"],
            row["coverage_selected_points"],
            row["coverage_complete"],
        )
        for row in rows
    } == {("quick_shape_subset", "7", "1", "False")}


def test_default_formal_shape_matrices_remain_reachable(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "fake")

    add_framework = _FakeFramework(tmp_path / "add")
    add_suite = AddTestSuite()
    add_suite.framework = add_framework
    add_suite.operator_test = _FakeOperator(["cuda_torch_add_out"])
    add_result = add_suite.run_bandwidth_test(
        device="cuda:0", plot_results=False
    )
    assert [
        call["data"]["shape"][0] for call in add_framework.calls
    ] == [2**power for power in range(12, 28)]
    _assert_coverage(
        add_result["rows"],
        mode="full_formal_shape_matrix",
        source="default_formal",
        total=16,
        requested=16,
        selected=16,
        selection_complete=True,
        complete=True,
    )

    linear_framework = _FakeFramework(tmp_path / "linear")
    linear_suite = LinearTestSuite(precision="fp16")
    linear_suite.framework = linear_framework
    linear_suite.operator_test = _FakeOperator(["cuda_torch_mm_out"])
    linear_result = linear_suite.run_tflops_test(
        device="cuda:0", plot_results=False
    )
    assert [
        call["data"]["batch_size"] for call in linear_framework.calls
    ] == list(range(256, 4097, 128))
    _assert_coverage(
        linear_result["rows"],
        mode="full_formal_shape_matrix",
        source="default_formal",
        total=31,
        requested=31,
        selected=31,
        selection_complete=True,
        complete=True,
    )

    rms_framework = _FakeFramework(tmp_path / "rmsnorm")
    rms_suite = RMSNormTestSuite()
    rms_suite.framework = rms_framework
    rms_suite.operator_test = _FakeOperator(["cuda_vllm_rms_norm_out"])
    rms_result = rms_suite.run_bandwidth_test(
        device="cuda:0", plot_results=False
    )
    assert len(rms_framework.calls) == 32
    assert [
        call["data"]["shape"][1] for call in rms_framework.calls[16:]
    ] == [1024 * index for index in range(1, 17)]
    for curve_rows in (
        rms_result["size_rows"],
        rms_result["hidden_rows"],
    ):
        _assert_coverage(
            curve_rows,
            mode="full_formal_shape_matrix",
            source="default_formal",
            total=16,
            requested=16,
            selected=16,
            selection_complete=True,
            complete=True,
        )

    flash_framework = _FakeFramework(tmp_path / "flash")
    flash_suite = FlashAttentionTestSuite()
    flash_suite.framework = flash_framework
    flash_suite.operator_test = _FakeOperator([
        "cuda_sdpa_flash_attention",
        "cuda_flash_attn_func",
    ])
    monkeypatch.setattr(
        flash_suite,
        "_cuda_benchmark_data",
        lambda **kwargs: {
            "metadata": kwargs,
            "sparse_mode": 3 if kwargs["causal"] else 0,
        },
    )
    flash_result = flash_suite.run_tflops_test(
        precision="bf16",
        device="cuda:0",
        plot_results=False,
    )
    assert len(flash_framework.calls) == 40
    assert {
        call["data"]["metadata"]["seq_len"]
        for call in flash_framework.calls
    } == {1024, 2048, 4096, 8192, 16384}
    assert {
        call["data"]["metadata"]["head_dim"]
        for call in flash_framework.calls
    } == {64, 128}
    assert {
        call["data"]["metadata"]["causal"]
        for call in flash_framework.calls
    } == {True, False}
    _assert_coverage(
        flash_result["rows"],
        mode="full_formal_shape_matrix",
        source="default_formal",
        total=40,
        requested=40,
        selected=40,
        selection_complete=True,
        complete=True,
    )

    group_framework = _FakeFramework(tmp_path / "group")
    group_suite = GroupGemmTestSuite(
        precision="int8",
        num_experts=8,
        hidden_dim=7168,
        out_channel=4096,
    )
    group_suite.framework = group_framework
    group_suite.operator_test = _FakeOperator([
        "cuda_vllm_cutlass_scaled_mm_bf16"
    ])
    group_result = group_suite.run_tflops_test(
        device="cuda:0", plot_results=False
    )
    assert [
        call["data"]["seq_len"] for call in group_framework.calls
    ] == [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    _assert_coverage(
        group_result["results"],
        mode="full_formal_shape_matrix",
        source="default_formal",
        total=10,
        requested=10,
        selected=10,
        selection_complete=True,
        complete=True,
    )

    paged_framework = _FakeFramework(tmp_path / "paged")
    paged_suite = PagedAttentionTestSuite()
    paged_suite.framework = paged_framework
    paged_suite.operator_test = _FakeOperator(["cuda_flashinfer_fa2"])
    paged_result = paged_suite.run_latency_plot_test(
        device="cuda:0",
        plot_results=False,
    )
    assert len(paged_framework.calls) == 32 + 2 * 128
    assert {
        call["data"]["max_seq_len"]
        for call in paged_framework.calls[:32]
    } == set(range(1024, 32769, 1024))
    assert {
        call["data"]["batch_size"]
        for call in paged_framework.calls[32:]
    } == set(range(1, 129))
    assert {
        call["data"]["max_seq_len"]
        for call in paged_framework.calls[32:]
    } == {10000, 30000}
    _assert_coverage(
        paged_result["seqlen_rows"],
        mode="full_formal_shape_matrix",
        source="default_formal",
        total=32,
        requested=32,
        selected=32,
        selection_complete=True,
        complete=True,
    )
    _assert_coverage(
        paged_result["batch_rows"],
        mode="full_formal_shape_matrix",
        source="default_formal",
        total=256,
        requested=256,
        selected=256,
        selection_complete=True,
        complete=True,
    )

    recurrent_framework = _FakeFramework(tmp_path / "recurrent")
    recurrent_operator = _FakeOperator(["cuda_vllm_fla_direct_out"])
    monkeypatch.setattr(
        recurrent_benchmark,
        "resolve_device",
        lambda requested: ("cuda:0", "fake"),
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "RecurrentGatedDeltaRuleOperatorTest",
        lambda: recurrent_operator,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "OperatorTestFramework",
        lambda result_dir: recurrent_framework,
    )
    monkeypatch.setattr(
        recurrent_benchmark,
        "environment",
        lambda *args: {},
    )
    output = tmp_path / "recurrent" / "curve.csv"
    assert recurrent_benchmark.main([
        "--device", "cuda:0",
        "--output", str(output),
        "--skip-correctness",
    ]) == 0
    assert [
        (call["data"]["mode"], call["data"]["batch_size"])
        for call in recurrent_framework.calls
    ] == [
        (mode, batch)
        for mode in ("decode", "mtp3")
        for batch in (1, 4, 8, 16, 32, 64, 128)
    ]
    with output.open(newline="", encoding="utf-8") as handle:
        recurrent_rows = list(csv.DictReader(handle))
    assert {
        (
            row["selection_mode"],
            row["shape_matrix_source"],
            row["coverage_total_formal_points"],
            row["coverage_total_requested_points"],
            row["coverage_selected_points"],
            row["selection_covers_full_formal_matrix"],
            row["coverage_complete"],
        )
        for row in recurrent_rows
    } == {
        (
            "full_formal_shape_matrix",
            "default_formal",
            "7",
            "7",
            "7",
            "True",
            "True",
        )
    }
