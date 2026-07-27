#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from contextlib import contextmanager
import gc
from pathlib import Path
import sys
import weakref

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import (  # noqa: E402
    BaseOperatorTest,
    OperatorTestFramework,
    PrecisionType,
    build_curve_selection_provenance,
    finalize_curve_coverage,
)


class _FreshCpuOperator(BaseOperatorTest):

    def __init__(self):
        super().__init__("fresh_cpu")
        self.prepare_calls = 0
        self.execute_calls = 0

    def generate_test_data(self, **kwargs):
        return {"x": torch.arange(8, dtype=torch.float32)}

    def run_cpu_reference(self, data):
        return data["x"] + 1

    def run_device_implementation(
        self,
        data,
        device,
        precision,
        implementation="default",
    ):
        return data["x"] + 1

    def get_available_implementations(self, device):
        return ["default"]

    def _prepare_data_for_core_operator(
        self,
        data,
        device,
        precision,
        implementation="default",
    ):
        self.prepare_calls += 1
        return {"x": data["x"].to(device=device, dtype=precision.value).clone()}

    def _execute_core_operator(self, prepared_data, implementation="default"):
        self.execute_calls += 1
        return prepared_data["x"] + 1

    def calculate_throughput(self, data, time_ms):
        return data["x"].numel() * 1000.0 / time_ms


class _ReusedInputOperator(_FreshCpuOperator):

    def _prepare_data_for_core_operator(
        self,
        data,
        device,
        precision,
        implementation="default",
    ):
        self.prepare_calls += 1
        return {"x": data["x"]}


class _ReusedOutputOperator(_FreshCpuOperator):

    def __init__(self):
        super().__init__()
        self.shared_output = torch.empty(8)

    def _execute_core_operator(self, prepared_data, implementation="default"):
        self.execute_calls += 1
        torch.add(prepared_data["x"], 1, out=self.shared_output)
        return self.shared_output


class _PreallocatedOutputOperator(_FreshCpuOperator):

    def _prepare_data_for_core_operator(
        self,
        data,
        device,
        precision,
        implementation="default",
    ):
        self.prepare_calls += 1
        x = data["x"].to(
            device=device,
            dtype=precision.value,
        ).clone()
        return {"x": x, "output": torch.empty_like(x)}

    def _execute_core_operator(self, prepared_data, implementation="default"):
        self.execute_calls += 1
        torch.add(
            prepared_data["x"],
            1,
            out=prepared_data["output"],
        )
        return prepared_data["output"]


class _InPlaceInputAliasOperator(_FreshCpuOperator):

    def _execute_core_operator(self, prepared_data, implementation="default"):
        self.execute_calls += 1
        prepared_data["x"].add_(1)
        return prepared_data["x"]


class _ContextOperator(_FreshCpuOperator):

    def __init__(self):
        super().__init__()
        self.context_entries = 0
        self.context_active = False

    @contextmanager
    def _core_operator_benchmark_context(
        self,
        device,
        precision,
        implementation,
    ):
        self.context_entries += 1
        self.context_active = True
        try:
            yield
        finally:
            self.context_active = False

    def _execute_core_operator(self, prepared_data, implementation="default"):
        assert self.context_active
        return super()._execute_core_operator(prepared_data, implementation)


class _RepeatLifetimeOperator(_FreshCpuOperator):

    def __init__(self, invocations_per_repeat):
        super().__init__()
        self.invocations_per_repeat = invocations_per_repeat
        self.current_input_refs = []
        self.current_output_refs = []
        self.alive_at_repeat_start = []

    def _prepare_data_for_core_operator(
        self,
        data,
        device,
        precision,
        implementation="default",
    ):
        invocation_index = self.prepare_calls % self.invocations_per_repeat
        if invocation_index == 0 and self.prepare_calls:
            gc.collect()
            self.alive_at_repeat_start.append((
                sum(ref() is not None for ref in self.current_input_refs),
                sum(ref() is not None for ref in self.current_output_refs),
            ))
            self.current_input_refs = []
            self.current_output_refs = []
        prepared = super()._prepare_data_for_core_operator(
            data,
            device,
            precision,
            implementation,
        )
        self.current_input_refs.append(weakref.ref(prepared["x"]))
        return prepared

    def _execute_core_operator(self, prepared_data, implementation="default"):
        output = super()._execute_core_operator(
            prepared_data,
            implementation,
        )
        self.current_output_refs.append(weakref.ref(output))
        return output


def _framework(tmp_path):
    return OperatorTestFramework(result_dir=str(tmp_path / "results"))


def test_v2_preallocates_each_repeat_and_uses_median(monkeypatch, tmp_path):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()
    repeat_means = iter([3.0, 1.0, 2.0])

    def measure(functions, device):
        for function in functions:
            function()
        return next(repeat_means)

    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        measure,
    )

    metrics = framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cpu",
        precision=PrecisionType.FP32,
        num_warmup=1,
        num_iterations=2,
        num_repeats=3,
        retain_outputs=True,
        verify_independent_storage=True,
    )

    assert operator.prepare_calls == 3 * (1 + 2)
    assert operator.execute_calls == 3 * (1 + 2)
    assert metrics.avg_time_ms == 2.0
    assert metrics.repeat_samples_ms == [3.0, 1.0, 2.0]
    assert metrics.repeats == 3
    assert metrics.aggregation == "median_of_repeat_means"
    assert metrics.preallocated_invocations_per_repeat == 3
    assert metrics.input_storage_sets_verified == 3
    assert metrics.output_storage_sets_verified == 3


@pytest.mark.parametrize(
    ("num_warmup", "num_iterations", "num_repeats"),
    [
        (-1, 1, 1),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, -1),
    ],
)
def test_v2_rejects_invalid_counts(
    tmp_path,
    num_warmup,
    num_iterations,
    num_repeats,
):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()

    with pytest.raises(ValueError, match="warmup|iterations|repeats"):
        framework.run_core_operator_performance_test_v2(
            operator_test=operator,
            data=operator.generate_test_data(),
            device="cpu",
            precision=PrecisionType.FP32,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            num_repeats=num_repeats,
        )

    assert operator.prepare_calls == 0


@pytest.mark.parametrize(
    ("count_name", "invalid_value"),
    [
        ("num_warmup", True),
        ("num_warmup", 1.0),
        ("num_warmup", "1"),
        ("num_iterations", True),
        ("num_iterations", 1.0),
        ("num_iterations", "1"),
        ("num_repeats", True),
        ("num_repeats", 1.0),
        ("num_repeats", "1"),
    ],
)
def test_v2_rejects_non_integer_counts_before_preparation(
    tmp_path,
    count_name,
    invalid_value,
):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()
    counts = {
        "num_warmup": 0,
        "num_iterations": 1,
        "num_repeats": 1,
    }
    counts[count_name] = invalid_value

    with pytest.raises(ValueError, match="must be a non-bool int"):
        framework.run_core_operator_performance_test_v2(
            operator_test=operator,
            data=operator.generate_test_data(),
            device="cpu",
            precision=PrecisionType.FP32,
            **counts,
        )

    assert operator.prepare_calls == 0


def test_v2_rejects_reused_prepared_storage(tmp_path):
    framework = _framework(tmp_path)
    operator = _ReusedInputOperator()

    with pytest.raises(RuntimeError, match="reuse device storage"):
        framework.run_core_operator_performance_test_v2(
            operator_test=operator,
            data=operator.generate_test_data(),
            device="cpu",
            precision=PrecisionType.FP32,
            num_warmup=1,
            num_iterations=2,
            verify_independent_storage=True,
        )


def test_v2_rejects_reused_returned_output_storage(tmp_path):
    framework = _framework(tmp_path)
    operator = _ReusedOutputOperator()

    with pytest.raises(RuntimeError, match="reuse device storage"):
        framework.run_core_operator_performance_test_v2(
            operator_test=operator,
            data=operator.generate_test_data(),
            device="cpu",
            precision=PrecisionType.FP32,
            num_warmup=1,
            num_iterations=2,
            retain_outputs=True,
            verify_independent_storage=True,
        )


def test_v2_emits_flat_protocol_provenance(monkeypatch, tmp_path):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()

    def measure(functions, device):
        for function in functions:
            function()
        return 0.25

    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        measure,
    )

    metrics = framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cpu",
        precision=PrecisionType.FP32,
        num_warmup=1,
        num_iterations=2,
        num_repeats=2,
        retain_outputs=True,
        verify_independent_storage=True,
    )
    provenance = framework.performance_provenance(metrics)

    assert provenance == {
        "framework_api": (
            "OperatorTestFramework.run_core_operator_performance_test_v2"
        ),
        "protocol_version": "operator-test-framework-v2-fresh-v2",
        "warmup": 1,
        "iterations": 2,
        "repeats": 2,
        "repeat_samples_ms": "[0.25, 0.25]",
        "repeat_min_ms": 0.25,
        "repeat_median_ms": 0.25,
        "repeat_max_ms": 0.25,
        "repeat_spread_pct": 0.0,
        "aggregation": "median_of_repeat_means",
        "preallocated_invocations_per_repeat": 3,
        "input_reuse_within_repeat": False,
        "input_storage_sets_verified": 3,
        "input_storage_ptr_count": 3,
        "output_storage_sets_verified": 3,
        "output_storage_ptr_count": 3,
        "output_tensor_count": 3,
        "output_unique_storages_per_set": 1,
        "preallocated_output_aliases_verified": 0,
        "preallocated_output_sets_verified": 0,
        "output_tensors_per_set": 1,
        "output_allocation_mode": (
            "no_preallocated_output_buffer_verified"
        ),
        "output_allocation_policy": (
            "no_preallocated_output_buffer_verified"
        ),
        "output_storage_policy": "retained_until_repeat_end",
        "timing_method": "host_perf_counter",
        "timing_semantics": "host wall-clock elapsed time",
        "workspace_allocation_policy": "not_audited",
        "timed_region": (
            "_execute_core_operator calls only; prepare excluded"
        ),
    }


def test_v2_proves_preallocated_output_aliases(monkeypatch, tmp_path):
    framework = _framework(tmp_path)
    operator = _PreallocatedOutputOperator()

    def measure(functions, device):
        for function in functions:
            function()
        return 0.25

    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        measure,
    )

    metrics = framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cpu",
        precision=PrecisionType.FP32,
        num_warmup=1,
        num_iterations=2,
        retain_outputs=True,
        verify_independent_storage=True,
    )

    assert metrics.preallocated_output_aliases_verified == 3
    assert metrics.output_allocation_mode == (
        "preallocated_output_buffers_verified"
    )
    assert framework.performance_provenance(metrics)[
        "preallocated_output_aliases_verified"
    ] == 3


def test_v2_does_not_mislabel_in_place_input_as_preallocated_output(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _InPlaceInputAliasOperator()

    def measure(functions, device):
        for function in functions:
            function()
        return 0.25

    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        measure,
    )
    metrics = framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cpu",
        precision=PrecisionType.FP32,
        num_warmup=1,
        num_iterations=2,
        retain_outputs=True,
        verify_independent_storage=True,
    )

    assert metrics.preallocated_output_aliases_verified == 0
    assert metrics.output_allocation_mode == (
        "no_preallocated_output_buffer_verified"
    )


def test_curve_selection_never_labels_custom_or_quick_as_full():
    custom = build_curve_selection_provenance(
        quick=False,
        num_shards=1,
        total_formal_points=16,
        total_requested_points=1,
        selected_points=1,
        uses_formal_shape_matrix=False,
    )
    assert custom["selection_mode"] == "custom_shape_matrix"
    assert custom["selection_covers_full_formal_matrix"] is False
    assert custom["coverage_complete"] is False

    quick = build_curve_selection_provenance(
        quick=True,
        num_shards=1,
        total_formal_points=16,
        total_requested_points=16,
        selected_points=1,
        uses_formal_shape_matrix=True,
    )
    assert quick["selection_mode"] == "quick_shape_subset"
    assert quick["shape_matrix_source"] == "default_formal"
    assert quick["selection_covers_full_formal_matrix"] is False

    shard = build_curve_selection_provenance(
        quick=False,
        num_shards=2,
        total_formal_points=16,
        total_requested_points=16,
        selected_points=8,
        uses_formal_shape_matrix=True,
    )
    assert shard["selection_mode"] == "formal_shape_shard"
    assert shard["coverage_mode"] == "sharded"


def test_curve_coverage_becomes_complete_only_after_all_success():
    selection = build_curve_selection_provenance(
        quick=False,
        num_shards=1,
        total_formal_points=2,
        total_requested_points=2,
        selected_points=2,
        uses_formal_shape_matrix=True,
    )
    rows = [
        {**selection, "point_index": 0, "status": "ok"},
        {**selection, "point_index": 1, "status": "ok"},
    ]
    assert finalize_curve_coverage(rows) is True
    assert all(row["coverage_complete"] is True for row in rows)

    failed_rows = [
        {**selection, "point_index": 0, "status": "ok"},
        {**selection, "point_index": 1, "status": "error"},
    ]
    assert finalize_curve_coverage(failed_rows) is False
    assert all(row["coverage_complete"] is False for row in failed_rows)

    duplicate_rows = [
        {**selection, "point_index": 0, "status": "ok"},
        {**selection, "point_index": 0, "status": "ok"},
    ]
    assert finalize_curve_coverage(duplicate_rows) is False


def test_v2_keeps_single_repeat_compatibility(monkeypatch, tmp_path):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()

    def measure(functions, device):
        for function in functions:
            function()
        return 0.5

    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        measure,
    )

    metrics = framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cpu",
        precision=PrecisionType.FP32,
        num_warmup=1,
        num_iterations=1,
    )

    assert metrics.avg_time_ms == 0.5
    assert metrics.repeat_samples_ms == [0.5]
    assert metrics.repeats == 1


def test_v2_enters_provider_context_once_per_repeat(tmp_path):
    framework = _framework(tmp_path)
    operator = _ContextOperator()

    metrics = framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cpu",
        precision=PrecisionType.FP32,
        num_warmup=1,
        num_iterations=2,
        num_repeats=3,
    )

    assert metrics.avg_time_ms > 0
    assert operator.context_entries == 3


def test_v2_releases_repeat_storage_before_next_repeat(tmp_path):
    framework = _framework(tmp_path)
    operator = _RepeatLifetimeOperator(invocations_per_repeat=2)

    framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cpu",
        precision=PrecisionType.FP32,
        num_warmup=1,
        num_iterations=1,
        num_repeats=3,
        retain_outputs=True,
    )

    assert operator.alive_at_repeat_start == [(0, 0), (0, 0)]
