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
        "protocol_version": "operator-test-framework-v2-fresh-v1",
        "warmup": 1,
        "iterations": 2,
        "repeats": 2,
        "repeat_samples_ms": "[0.25, 0.25]",
        "aggregation": "median_of_repeat_means",
        "preallocated_invocations_per_repeat": 3,
        "input_reuse_within_repeat": False,
        "input_storage_sets_verified": 3,
        "input_storage_ptr_count": 3,
        "output_storage_sets_verified": 3,
        "output_storage_ptr_count": 3,
        "output_storage_policy": "retained_until_repeat_end",
        "timed_region": "_execute_core_operator only; prepare excluded",
    }


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
