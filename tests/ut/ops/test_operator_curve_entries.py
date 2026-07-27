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
from tests.test_rmsnorm import RMSNormTestSuite  # noqa: E402
import recurrent_gated_delta_rule.benchmark as recurrent_benchmark  # noqa: E402
import tests.test_recurrent_gated_delta_rule  # noqa: E402,F401


PROVENANCE = {
    "framework_api": (
        "OperatorTestFramework.run_core_operator_performance_test_v2"
    ),
    "protocol_version": "operator-test-framework-v2-fresh-v1",
    "warmup": 1,
    "iterations": 2,
    "repeats": 3,
    "repeat_samples_ms": "[1.5, 1.0, 2.0]",
    "aggregation": "median_of_repeat_means",
    "preallocated_invocations_per_repeat": 3,
    "input_reuse_within_repeat": False,
    "input_storage_sets_verified": 3,
    "input_storage_ptr_count": 6,
    "output_storage_sets_verified": 3,
    "output_storage_ptr_count": 3,
    "output_storage_policy": "retained_until_repeat_end",
    "timed_region": "_execute_core_operator only; prepare excluded",
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
    _assert_success_rows(result["size_rows"] + result["hidden_rows"])


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
        num_iterations=20,
        repeats=3,
        num_blocks=10000,
        block_size=128,
        plot_results=False,
    )

    assert len(framework.calls) == 2
    for call in framework.calls:
        _assert_formal_call(call, 5, 20, 3, "cuda_flashinfer_fa2")
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
    assert rows[0]["status"] == "ok"


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
