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
from dataclasses import fields
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
    PerformanceMetrics,
    PrecisionType,
    build_curve_selection_provenance,
    build_fresh_iteration_plan,
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

    def _declares_preallocated_output_contract(
        self,
        prepared_data,
        implementation="default",
    ):
        del prepared_data, implementation
        return True


class _InPlaceInputAliasOperator(_FreshCpuOperator):

    def _execute_core_operator(self, prepared_data, implementation="default"):
        self.execute_calls += 1
        prepared_data["x"].add_(1)
        return prepared_data["x"]


class _CrossInvocationInputAliasOperator(_FreshCpuOperator):

    def __init__(self):
        super().__init__()
        self.prepared_inputs = []

    def _prepare_data_for_core_operator(
        self,
        data,
        device,
        precision,
        implementation="default",
    ):
        prepared = super()._prepare_data_for_core_operator(
            data,
            device,
            precision,
            implementation,
        )
        self.prepared_inputs.append(prepared["x"])
        return prepared

    def _execute_core_operator(
        self,
        prepared_data,
        implementation="default",
    ):
        del prepared_data, implementation
        output_index = (self.execute_calls + 1) % len(
            self.prepared_inputs
        )
        self.execute_calls += 1
        return self.prepared_inputs[output_index]


class _IgnoredPreallocatedOutputOperator(_PreallocatedOutputOperator):

    def _execute_core_operator(self, prepared_data, implementation="default"):
        self.execute_calls += 1
        return prepared_data["x"] + 1


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


class _DirectRepeatLifetimeOperator(_PreallocatedOutputOperator):

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
        self.current_output_refs.append(weakref.ref(prepared["output"]))
        return prepared


def _framework(tmp_path):
    return OperatorTestFramework(result_dir=str(tmp_path / "results"))


def test_performance_metrics_keeps_v4_positional_field_order():
    legacy_fields = (
        "avg_time_ms",
        "throughput",
        "precision_type",
        "device_type",
        "operator_name",
        "iterations",
        "throughput_ops_per_sec",
        "tops",
        "bandwidth_gb_s",
        "framework_api",
        "warmup_iterations",
        "preallocated_input_sets",
        "independent_storage_sets_verified",
        "independent_output_storage_sets_verified",
        "preallocated_output_aliases_verified",
        "output_allocation_mode",
        "output_storage_policy",
        "protocol_version",
        "repeats",
        "repeat_samples_ms",
        "aggregation",
        "preallocated_invocations_per_repeat",
        "input_storage_sets_verified",
        "input_storage_ptr_count",
        "output_storage_sets_verified",
        "output_storage_ptr_count",
        "output_tensor_count",
        "input_reuse_within_repeat",
        "timing_method",
        "timing_semantics",
        "timed_output_capture_policy",
        "preallocated_output_contract",
        "output_alias_verification_scope",
        "preallocated_output_contract_invocations_per_repeat",
        "output_verification_replay_invocations_per_repeat",
        "total_operator_calls_per_repeat",
        "workspace_allocation_policy",
        "dispatch_loop_policy",
        "device_stabilization_policy",
        "device_stabilization_timed",
        "task_queue_enable",
        "timed_region",
    )
    actual_fields = tuple(field.name for field in fields(PerformanceMetrics))

    assert actual_fields[:len(legacy_fields)] == legacy_fields
    assert actual_fields[len(legacy_fields):] == (
        "stabilization_repeats",
        "stabilization_repeat_samples_ms",
        "input_output_storage_disjoint",
        "stabilization_operator_calls",
    )


def _execute_measurement_payloads(
    payloads,
    execute_core_operator,
    implementation,
    device,
    retained_outputs=None,
):
    del device
    OperatorTestFramework._dispatch_prepared_payloads(
        payloads,
        execute_core_operator,
        implementation,
        retained_outputs,
    )


def test_fresh_iteration_plan_uses_soft_target_without_lowering_base():
    small = build_fresh_iteration_plan(
        num_warmup=5,
        requested_iterations=None,
        base_iterations=20,
        estimated_unique_bytes_per_invocation=6 * 4096,
    )
    assert small["effective_iterations"] == 2048
    assert small["fresh_storage_soft_target_overflow"] is False

    large = build_fresh_iteration_plan(
        num_warmup=5,
        requested_iterations=None,
        base_iterations=20,
        estimated_unique_bytes_per_invocation=6 * 2**27,
    )
    assert large["adaptive_capacity_iterations"] == 0
    assert large["effective_iterations"] == 20
    assert large["fresh_storage_soft_target_overflow"] is True
    assert large["estimated_fresh_storage_bytes_per_repeat"] == (
        25 * 6 * 2**27
    )


def test_fresh_iteration_plan_honors_explicit_count():
    plan = build_fresh_iteration_plan(
        num_warmup=5,
        requested_iterations=7,
        base_iterations=20,
        estimated_unique_bytes_per_invocation=1024,
    )
    assert plan["iteration_selection_policy"] == "explicit_fixed"
    assert plan["requested_iterations"] == 7
    assert plan["effective_iterations"] == 7


def test_nested_output_collection_matches_input_exclusion():
    input_tensor = torch.empty(8)
    output_tensor = torch.empty(8)
    prepared = {
        "x": input_tensor,
        "providers": [
            {
                "metadata": "nested",
                "output": output_tensor,
            }
        ],
    }

    outputs = OperatorTestFramework._prepared_output_values(prepared)
    assert len(outputs) == 1
    assert outputs[0] is output_tensor

    inputs = OperatorTestFramework._prepared_input_values(prepared)
    input_ptrs = OperatorTestFramework._device_storage_ptrs(
        inputs,
        "cpu",
    )
    output_ptrs = OperatorTestFramework._device_storage_ptrs(
        outputs,
        "cpu",
    )
    assert input_tensor.untyped_storage().data_ptr() in input_ptrs
    assert output_tensor.untyped_storage().data_ptr() not in input_ptrs
    assert output_tensor.untyped_storage().data_ptr() in output_ptrs


def test_dispatch_writes_preallocated_output_slots_without_growth():
    retained_outputs = ["warmup", None, None]

    OperatorTestFramework._dispatch_prepared_payloads(
        [2, 3],
        lambda value, implementation: value * 10,
        "default",
        retained_outputs,
    )

    assert retained_outputs == ["warmup", 20, 30]


def test_v2_preallocates_each_repeat_and_uses_median(monkeypatch, tmp_path):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()
    repeat_means = iter([3.0, 1.0, 2.0])

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
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


def test_v2_excludes_full_window_stabilization_repeats(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()
    repeat_means = iter([9.0, 8.0, 3.0, 1.0, 2.0])

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
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
        num_stabilization_repeats=2,
        retain_outputs=True,
        verify_independent_storage=True,
    )

    assert operator.prepare_calls == 5 * (1 + 2)
    assert operator.execute_calls == 5 * (1 + 2)
    assert metrics.stabilization_repeat_samples_ms == [9.0, 8.0]
    assert metrics.repeat_samples_ms == [3.0, 1.0, 2.0]
    assert metrics.avg_time_ms == 2.0
    assert metrics.aggregation == (
        "median_of_post_stabilization_repeat_means"
    )
    assert metrics.device_stabilization_policy == (
        "fresh_storage_full_window_priming_repeats"
    )
    assert metrics.device_stabilization_timed is True
    assert metrics.stabilization_operator_calls == 2 * (1 + 2)

    provenance = framework.performance_provenance(metrics)
    assert provenance["stabilization_repeats"] == 2
    assert provenance["stabilization_repeat_samples_ms"] == "[9.0, 8.0]"
    assert provenance["repeat_p25_ms"] == 1.5
    assert provenance["repeat_p75_ms"] == 2.5
    assert provenance["repeat_iqr_pct"] == 50.0


def test_v2_rejects_noncanonical_stabilization_environment(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()
    monkeypatch.setenv("OPERATOR_TEST_STABILIZATION_REPEATS", "02")

    with pytest.raises(
        ValueError,
        match="canonical base-10 integer syntax",
    ):
        framework.run_core_operator_performance_test_v2(
            operator_test=operator,
            data=operator.generate_test_data(),
            device="cpu",
            precision=PrecisionType.FP32,
            num_warmup=1,
            num_iterations=2,
            num_repeats=1,
        )


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

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
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
        "protocol_version": "operator-test-framework-v2-fresh-v5",
        "warmup": 1,
        "iterations": 2,
        "repeats": 2,
        "stabilization_repeats": 0,
        "stabilization_repeat_samples_ms": "[]",
        "repeat_samples_ms": "[0.25, 0.25]",
        "event_window_samples_ms": "[0.5, 0.5]",
        "event_window_min_ms": 0.5,
        "event_window_median_ms": 0.5,
        "event_window_max_ms": 0.5,
        "repeat_min_ms": 0.25,
        "repeat_median_ms": 0.25,
        "repeat_max_ms": 0.25,
        "repeat_p25_ms": 0.25,
        "repeat_p75_ms": 0.25,
        "repeat_iqr_pct": 0.0,
        "repeat_spread_pct": 0.0,
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
        "timed_output_capture_policy": (
            "retained_return_inside_timed_region"
        ),
        "preallocated_output_contract": "not_declared",
        "output_alias_verification_scope": (
            "warmup_and_measured_returns"
        ),
        "preallocated_output_contract_invocations_per_repeat": 0,
        "output_verification_replay_invocations_per_repeat": 0,
        "total_operator_calls_per_repeat": 3,
        "workspace_allocation_policy": "not_audited",
        "dispatch_loop_policy": "python_direct_prepared_payload_loop",
        "device_stabilization_policy": "none",
        "device_stabilization_timed": False,
        "stabilization_operator_calls": 0,
        "task_queue_enable": "not_applicable",
        "timed_region": (
            "Python direct prepared-payload loop of "
            "_execute_core_operator plus return slot assignment; "
            "prepare excluded"
        ),
    }


def test_v2_proves_preallocated_output_aliases(monkeypatch, tmp_path):
    framework = _framework(tmp_path)
    operator = _PreallocatedOutputOperator()

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        assert operator.execute_calls == 1
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
        assert operator.execute_calls == 3
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

    assert metrics.preallocated_output_aliases_verified == 1
    assert operator.execute_calls == 3
    assert metrics.input_storage_sets_verified == 3
    assert metrics.input_storage_ptr_count == 3
    assert metrics.input_output_storage_disjoint is True
    assert metrics.output_storage_sets_verified == 3
    assert metrics.output_storage_ptr_count == 3
    assert metrics.output_allocation_mode == (
        "preallocated_output_contract_with_warmup_alias_probe"
    )
    provenance = framework.performance_provenance(metrics)
    assert provenance["preallocated_output_aliases_verified"] == 1
    assert provenance["timed_output_capture_policy"] == (
        "preallocated_output_contract_no_timed_return_capture"
    )
    assert provenance["preallocated_output_contract"] == (
        "declared_phase_invariant_out"
    )
    assert provenance["output_alias_verification_scope"] == (
        "warmup_returns_only"
    )
    assert provenance[
        "preallocated_output_contract_invocations_per_repeat"
    ] == 3
    assert provenance[
        "output_verification_replay_invocations_per_repeat"
    ] == 0
    assert provenance["total_operator_calls_per_repeat"] == 3
    assert provenance["timed_region"] == (
        "Python direct prepared-payload loop of _execute_core_operator; "
        "prepare excluded; timed Python returns discarded under declared "
        "out contract"
    )


def test_v2_rejects_input_output_storage_alias(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _InPlaceInputAliasOperator()

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
        return 0.25

    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        measure,
    )
    with pytest.raises(
        RuntimeError,
        match="input/workspace and output storage domains overlap",
    ):
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


def test_v2_rejects_cross_invocation_input_output_alias(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _CrossInvocationInputAliasOperator()

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
        return 0.25

    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        measure,
    )
    with pytest.raises(
        RuntimeError,
        match="input/workspace and output storage domains overlap",
    ):
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


def test_v2_rejects_direct_path_when_prepared_output_is_ignored(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _IgnoredPreallocatedOutputOperator()

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
        return 0.25

    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        measure,
    )
    with pytest.raises(
        RuntimeError,
        match="direct preallocated timing contract failed its warmup",
    ):
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


def test_v2_reports_declared_contract_when_zero_warmup_disables_direct_path(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _PreallocatedOutputOperator()

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
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
        num_warmup=0,
        num_iterations=2,
        retain_outputs=True,
        verify_independent_storage=True,
    )

    provenance = framework.performance_provenance(metrics)
    assert provenance["preallocated_output_contract"] == (
        "declared_phase_invariant_out"
    )
    assert provenance["timed_output_capture_policy"] == (
        "retained_return_inside_timed_region"
    )
    assert provenance["output_alias_verification_scope"] == (
        "warmup_and_measured_returns"
    )
    assert provenance[
        "preallocated_output_contract_invocations_per_repeat"
    ] == 2
    assert provenance["preallocated_output_aliases_verified"] == 2
    assert provenance["total_operator_calls_per_repeat"] == 2


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

    def measure(
        payloads,
        execute_core_operator,
        implementation,
        device,
        retained_outputs=None,
    ):
        _execute_measurement_payloads(
            payloads,
            execute_core_operator,
            implementation,
            device,
            retained_outputs,
        )
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


def test_v2_releases_direct_output_storage_before_next_repeat(tmp_path):
    framework = _framework(tmp_path)
    operator = _DirectRepeatLifetimeOperator(invocations_per_repeat=2)

    framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cpu",
        precision=PrecisionType.FP32,
        num_warmup=1,
        num_iterations=1,
        num_repeats=3,
        retain_outputs=True,
        verify_independent_storage=True,
    )

    assert operator.alive_at_repeat_start == [(0, 0), (0, 0)]
