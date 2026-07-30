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

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, fields
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import (  # noqa: E402
    BaseOperatorTest,
    OperatorTestFramework,
    PerformanceMetrics,
    ProfilerBackend,
    ProfilerFactory,
    PrecisionType,
    build_curve_selection_provenance,
    build_fresh_iteration_plan,
    finalize_curve_coverage,
)


class _FreshCpuOperator(BaseOperatorTest):

    def __init__(self, operator_name="fresh_cpu"):
        super().__init__(operator_name)
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


class _MutableGraphOperator(_PreparedProfileOperator):

    def __init__(self):
        super().__init__()
        self.restore_log = []
        self.event_window_active = None

    def _restore_mutable_graph_inputs(
        self,
        prepared_payloads,
        implementation="default",
    ):
        del prepared_payloads, implementation
        assert self.event_window_active is not None
        assert self.event_window_active[0] is False
        self.restore_log.append(
            "after_capture"
            if not self.restore_log
            else "before_measured_replay"
        )


class _FakeProfiler:

    def __init__(self, operator):
        self.operator = operator
        self.events = []
        self.executed_at_start = []
        self.executed_after_steps = []
        self.executed_at_stop = []
        self.prepared_at_start = []
        self.prepared_after_steps = []
        self.prepared_at_stop = []

    def start(self):
        self.events.append("start")
        self.executed_at_start = list(self.operator.executed_ids)
        self.prepared_at_start = list(self.operator.prepared_ids)

    def step(self):
        self.events.append("step")
        self.executed_after_steps.append(list(self.operator.executed_ids))
        self.prepared_after_steps.append(list(self.operator.prepared_ids))

    def stop(self):
        self.events.append("stop")
        self.executed_at_stop = list(self.operator.executed_ids)
        self.prepared_at_stop = list(self.operator.prepared_ids)


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


def _install_cpu_backed_cuda(monkeypatch, elapsed_ms=12.0):
    event_window_active = [False]
    original_storage_ptrs = OperatorTestFramework._device_storage_ptrs

    def prepare_on_cpu(
        self,
        data,
        device,
        precision,
        implementation="default",
    ):
        del device, implementation
        self.prepare_calls += 1
        return {
            "x": data["x"].to(dtype=precision.value).clone(),
        }

    def cpu_backed_cuda_storage_ptrs(value, device_type):
        if device_type == "cuda":
            device_type = "cpu"
        return original_storage_ptrs(value, device_type)

    class FakeEvent:

        created = 0

        def __init__(self, enable_timing):
            assert enable_timing is True
            self.index = FakeEvent.created
            FakeEvent.created += 1

        def record(self):
            event_window_active[0] = self.index % 2 == 0

        def synchronize(self):
            assert self.index % 2 == 1
            event_window_active[0] = False

        def elapsed_time(self, end_event):
            assert self.index % 2 == 0
            assert end_event.index == self.index + 1
            return elapsed_ms

    monkeypatch.setattr(
        OperatorTestFramework,
        "_device_storage_ptrs",
        staticmethod(cpu_backed_cuda_storage_ptrs),
    )
    monkeypatch.setattr(
        _FreshCpuOperator,
        "_prepare_data_for_core_operator",
        prepare_on_cpu,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    return event_window_active


def _run_captured_chain(
    framework,
    operator,
    *,
    num_warmup=0,
    num_iterations=2,
    num_repeats=1,
    retain_outputs=True,
    verify_independent_storage=False,
):
    return framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cuda:0",
        precision=PrecisionType.FP32,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        num_repeats=num_repeats,
        dispatch_mode="captured_chain",
        retain_outputs=retain_outputs,
        verify_independent_storage=verify_independent_storage,
    )


def test_captured_chain_times_one_replay_of_independent_payloads(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _MutableGraphOperator()
    event_window_active = _install_cpu_backed_cuda(
        monkeypatch,
        elapsed_ms=12.0,
    )
    operator.event_window_active = event_window_active
    captured_payload_ids = []
    captured_storage_ptrs = []
    replay_calls = 0

    def capture(
        prepared_payloads,
        execute_core_operator,
        implementation,
        device,
    ):
        del device
        captured_payload_ids.extend(
            payload["payload_id"] for payload in prepared_payloads
        )
        captured_storage_ptrs.extend(
            payload["x"].untyped_storage().data_ptr()
            for payload in prepared_payloads
        )
        retained_outputs = [
            execute_core_operator(payload, implementation)
            for payload in prepared_payloads
        ]

        def replay():
            nonlocal replay_calls
            assert event_window_active[0] is True
            assert all(output is not None for output in retained_outputs)
            replay_calls += 1

        return SimpleNamespace(
            replay=replay,
            retained_outputs=retained_outputs,
            logical_invocations=len(prepared_payloads),
        )

    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        capture,
    )

    num_iterations = 3
    metrics = _run_captured_chain(
        framework,
        operator,
        num_iterations=num_iterations,
        retain_outputs=True,
        verify_independent_storage=False,
    )

    assert captured_payload_ids == list(range(num_iterations))
    assert len(set(captured_storage_ptrs)) == num_iterations
    assert replay_calls == 1
    assert operator.restore_log == [
        "after_capture",
        "before_measured_replay",
    ]
    assert metrics.avg_time_ms == 12.0 / num_iterations
    assert metrics.dispatch_mode == "captured_chain"
    assert metrics.graph_capture_width == num_iterations
    assert metrics.capture_timed is False
    assert metrics.mutable_inputs_restored is True
    assert metrics.graph_replays == 1
    assert metrics.profiler_is_diagnostic is False
    assert metrics.input_storage_sets_verified == num_iterations
    assert metrics.output_storage_sets_verified == num_iterations
    assert metrics.input_output_storage_disjoint is True
    provenance = framework.performance_provenance(metrics)
    assert provenance["timing_method"] == "device_event_graph_replay"
    assert provenance["timing_semantics"] == (
        "device elapsed time; capture and mutable restore excluded"
    )
    assert provenance["dispatch_loop_policy"] == (
        "captured_prepared_payload_chain_single_replay"
    )
    assert provenance["timed_region"] == (
        "one graph replay containing I independent core invocations"
    )
    assert provenance["timed_output_capture_policy"] == (
        "capture_returns_retained_outside_timed_replay"
    )
    assert provenance["output_storage_policy"] == (
        "capture_outputs_retained_until_repeat_end"
    )


def test_graph_dispatch_rejects_invalid_mode_before_preparation(tmp_path):
    framework = _framework(tmp_path)
    operator = _FreshCpuOperator()

    with pytest.raises(ValueError, match="dispatch_mode"):
        framework.run_core_operator_performance_test_v2(
            operator_test=operator,
            data=operator.generate_test_data(),
            device="cpu",
            precision=PrecisionType.FP32,
            num_warmup=0,
            num_iterations=1,
            dispatch_mode="eager",
        )

    assert operator.prepare_calls == 0


def test_captured_chain_propagates_capture_failure_without_eager_fallback(
    monkeypatch,
    tmp_path,
):
    from operator_test_framework import GraphCaptureUnsupportedError

    framework = _framework(tmp_path)
    operator = _MutableGraphOperator()
    event_window_active = _install_cpu_backed_cuda(monkeypatch)
    operator.event_window_active = event_window_active

    class CaptureFailure(RuntimeError):
        pass

    expected = CaptureFailure("capture refused")

    def fail_capture(*args, **kwargs):
        raise expected

    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        fail_capture,
    )
    monkeypatch.setattr(
        framework,
        "_measure_execution_time_v2",
        lambda *args, **kwargs: pytest.fail("must not fall back to eager"),
    )

    with pytest.raises(
        GraphCaptureUnsupportedError,
        match="capture refused",
    ) as exc_info:
        _run_captured_chain(
            framework,
            operator,
            num_warmup=1,
            num_iterations=2,
        )

    assert exc_info.value.__cause__ is expected


def test_captured_chain_replay_failure_is_not_capture_unsupported(
    monkeypatch,
    tmp_path,
):
    from operator_test_framework import GraphCaptureUnsupportedError

    framework = _framework(tmp_path)
    operator = _MutableGraphOperator()
    event_window_active = _install_cpu_backed_cuda(monkeypatch)
    operator.event_window_active = event_window_active

    def capture(
        prepared_payloads,
        execute_core_operator,
        implementation,
        device,
    ):
        del device
        outputs = [
            execute_core_operator(payload, implementation)
            for payload in prepared_payloads
        ]

        def replay():
            raise RuntimeError("graph replay failed after capture")

        return SimpleNamespace(
            replay=replay,
            retained_outputs=outputs,
            logical_invocations=len(prepared_payloads),
        )

    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        capture,
    )

    with pytest.raises(
        RuntimeError,
        match="graph replay failed after capture",
    ) as exc_info:
        _run_captured_chain(
            framework,
            operator,
            num_iterations=2,
        )

    assert not isinstance(exc_info.value, GraphCaptureUnsupportedError)


def test_captured_chain_capture_oom_is_not_capture_unsupported(
    monkeypatch,
    tmp_path,
):
    from operator_test_framework import GraphCaptureUnsupportedError

    framework = _framework(tmp_path)
    operator = _MutableGraphOperator()
    event_window_active = _install_cpu_backed_cuda(monkeypatch)
    operator.event_window_active = event_window_active

    def oom_during_capture(*args, **kwargs):
        raise torch.OutOfMemoryError("capture allocation exhausted")

    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        oom_during_capture,
    )

    with pytest.raises(
        torch.OutOfMemoryError,
        match="capture allocation exhausted",
    ) as exc_info:
        _run_captured_chain(
            framework,
            operator,
            num_iterations=2,
        )

    assert not isinstance(exc_info.value, GraphCaptureUnsupportedError)


def test_mutable_graph_restore_runs_outside_event_window(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _MutableGraphOperator()
    event_window_active = _install_cpu_backed_cuda(monkeypatch)
    operator.event_window_active = event_window_active
    timeline = []

    def capture(
        prepared_payloads,
        execute_core_operator,
        implementation,
        device,
    ):
        del device
        outputs = [
            execute_core_operator(payload, implementation)
            for payload in prepared_payloads
        ]

        def replay():
            assert event_window_active[0] is True
            timeline.append("replay")

        return SimpleNamespace(
            replay=replay,
            retained_outputs=outputs,
            logical_invocations=len(prepared_payloads),
        )

    original_restore = operator._restore_mutable_graph_inputs

    def restore(prepared_payloads, implementation="default"):
        original_restore(prepared_payloads, implementation)
        timeline.append("restore")

    monkeypatch.setattr(
        operator,
        "_restore_mutable_graph_inputs",
        restore,
    )
    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        capture,
    )

    _run_captured_chain(
        framework,
        operator,
        num_iterations=2,
    )

    assert timeline == ["restore", "restore", "replay"]


def test_captured_chain_retains_capture_outputs_through_replay(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _MutableGraphOperator()
    event_window_active = _install_cpu_backed_cuda(monkeypatch)
    operator.event_window_active = event_window_active
    capture_output_refs = []

    def capture(
        prepared_payloads,
        execute_core_operator,
        implementation,
        device,
    ):
        del device
        outputs = [
            execute_core_operator(payload, implementation)
            for payload in prepared_payloads
        ]
        capture_output_refs.extend(weakref.ref(output) for output in outputs)

        def replay():
            gc.collect()
            assert all(ref() is not None for ref in capture_output_refs)

        return SimpleNamespace(
            replay=replay,
            retained_outputs=outputs,
            logical_invocations=len(prepared_payloads),
        )

    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        capture,
    )

    _run_captured_chain(
        framework,
        operator,
        num_iterations=2,
        retain_outputs=False,
    )

    assert len(capture_output_refs) == 2


def test_captured_chain_rejects_reused_measured_inputs_without_opt_in(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _ReusedInputOperator()
    _install_cpu_backed_cuda(monkeypatch)
    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        lambda *args, **kwargs: pytest.fail(
            "invalid measured inputs must fail before capture"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="reuse device storage",
    ):
        _run_captured_chain(
            framework,
            operator,
            num_iterations=2,
            retain_outputs=False,
            verify_independent_storage=False,
        )


def test_captured_chain_rejects_reused_capture_outputs_before_replay(
    monkeypatch,
    tmp_path,
):
    from operator_test_framework import GraphCaptureUnsupportedError

    framework = _framework(tmp_path)
    operator = _ReusedOutputOperator()
    _install_cpu_backed_cuda(monkeypatch)
    replay_calls = 0
    restore_calls = 0

    def capture(
        prepared_payloads,
        execute_core_operator,
        implementation,
        device,
    ):
        del device
        outputs = [
            execute_core_operator(payload, implementation)
            for payload in prepared_payloads
        ]

        def replay():
            nonlocal replay_calls
            replay_calls += 1

        return SimpleNamespace(
            replay=replay,
            retained_outputs=outputs,
            logical_invocations=len(prepared_payloads),
        )

    def restore(*args, **kwargs):
        nonlocal restore_calls
        restore_calls += 1

    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        capture,
    )
    monkeypatch.setattr(
        operator,
        "_restore_mutable_graph_inputs",
        restore,
    )

    with pytest.raises(
        RuntimeError,
        match="reuse device storage",
    ) as exc_info:
        _run_captured_chain(
            framework,
            operator,
            num_iterations=2,
            retain_outputs=False,
            verify_independent_storage=False,
        )

    assert not isinstance(exc_info.value, GraphCaptureUnsupportedError)
    assert restore_calls == 0
    assert replay_calls == 0


def test_captured_chain_rejects_capture_output_input_alias_before_replay(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _InPlaceInputAliasOperator()
    _install_cpu_backed_cuda(monkeypatch)
    replay_calls = 0

    def capture(
        prepared_payloads,
        execute_core_operator,
        implementation,
        device,
    ):
        del device
        outputs = [
            execute_core_operator(payload, implementation)
            for payload in prepared_payloads
        ]

        def replay():
            nonlocal replay_calls
            replay_calls += 1

        return SimpleNamespace(
            replay=replay,
            retained_outputs=outputs,
            logical_invocations=len(prepared_payloads),
        )

    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        capture,
    )

    with pytest.raises(
        RuntimeError,
        match="input/workspace and output storage domains overlap",
    ):
        _run_captured_chain(
            framework,
            operator,
            num_iterations=2,
            retain_outputs=False,
            verify_independent_storage=False,
        )

    assert replay_calls == 0


def test_captured_chain_builds_fresh_payloads_and_graph_each_repeat(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _MutableGraphOperator()
    event_window_active = _install_cpu_backed_cuda(
        monkeypatch,
        elapsed_ms=6.0,
    )
    operator.event_window_active = event_window_active
    captured_id_groups = []
    captured_pointer_groups = []
    payload_refs = []
    prior_payloads_alive_at_capture = []
    graph_refs = []
    chain_refs = []
    graph_ids = []
    chain_ids = []
    prior_capture_objects_alive = []

    class GraphSentinel:

        def __init__(self, identity):
            self.identity = identity

        def replay(self):
            return None

    class ChainSentinel:

        def __init__(
            self,
            identity,
            graph,
            outputs,
            logical_invocations,
        ):
            self.identity = identity
            self.replay = graph.replay
            self.retained_outputs = outputs
            self.logical_invocations = logical_invocations

    def capture(
        prepared_payloads,
        execute_core_operator,
        implementation,
        device,
    ):
        del device
        gc.collect()
        prior_payloads_alive_at_capture.append(
            sum(ref() is not None for ref in payload_refs)
        )
        prior_capture_objects_alive.append((
            sum(ref() is not None for ref in graph_refs),
            sum(ref() is not None for ref in chain_refs),
        ))
        captured_id_groups.append([
            payload["payload_id"] for payload in prepared_payloads
        ])
        captured_pointer_groups.append([
            payload["x"].untyped_storage().data_ptr()
            for payload in prepared_payloads
        ])
        payload_refs.extend(
            weakref.ref(payload["x"]) for payload in prepared_payloads
        )
        outputs = [
            execute_core_operator(payload, implementation)
            for payload in prepared_payloads
        ]
        capture_identity = len(graph_ids)
        graph = GraphSentinel(capture_identity)
        chain = ChainSentinel(
            capture_identity,
            graph,
            outputs,
            len(prepared_payloads),
        )
        graph_ids.append(graph.identity)
        chain_ids.append(chain.identity)
        graph_refs.append(weakref.ref(graph))
        chain_refs.append(weakref.ref(chain))
        return chain

    monkeypatch.setattr(
        framework,
        "_capture_prepared_payload_chain_v2",
        capture,
    )

    metrics = _run_captured_chain(
        framework,
        operator,
        num_warmup=1,
        num_iterations=2,
        num_repeats=2,
        verify_independent_storage=True,
    )

    assert captured_id_groups == [[1, 2], [4, 5]]
    assert graph_ids == [0, 1]
    assert chain_ids == [0, 1]
    assert all(
        len(set(pointer_group)) == 2
        for pointer_group in captured_pointer_groups
    )
    assert prior_payloads_alive_at_capture == [0, 0]
    assert prior_capture_objects_alive == [(0, 0), (0, 0)]
    gc.collect()
    assert all(ref() is None for ref in graph_refs)
    assert all(ref() is None for ref in chain_refs)
    assert metrics.repeat_samples_ms == [3.0, 3.0]


def test_graph_dispatch_default_eager_provenance_is_byte_identical(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)

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

    def run(dispatch_mode=None):
        kwargs = {}
        if dispatch_mode is not None:
            kwargs["dispatch_mode"] = dispatch_mode
        operator = _FreshCpuOperator()
        metrics = framework.run_core_operator_performance_test_v2(
            operator_test=operator,
            data=operator.generate_test_data(),
            device="cpu",
            precision=PrecisionType.FP32,
            num_warmup=1,
            num_iterations=2,
            **kwargs,
        )
        return json.dumps(
            {
                "metrics": asdict(metrics),
                "provenance": framework.performance_provenance(metrics),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    default_bytes = run()
    explicit_eager_bytes = run("eager_direct")

    assert default_bytes == explicit_eager_bytes
    eager = json.loads(default_bytes)
    assert eager["metrics"]["dispatch_mode"] == "eager_direct"
    assert eager["metrics"]["graph_capture_width"] == 0
    assert eager["metrics"]["graph_replays"] == 0
    assert eager["metrics"]["capture_timed"] is False
    assert eager["metrics"]["mutable_inputs_restored"] is False
    assert eager["metrics"]["profiler_is_diagnostic"] is False


def test_cuda_graph_dispatch_helper_retains_every_capture_output(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    replay_calls = []
    graph_entries = []

    class FakeCudaGraph:

        def replay(self):
            replay_calls.append("replay")

    @contextmanager
    def fake_graph_context(graph):
        graph_entries.append(graph)
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeCudaGraph)
    monkeypatch.setattr(torch.cuda, "graph", fake_graph_context)

    payloads = [{"x": torch.tensor([index])} for index in range(3)]
    chain = framework._capture_prepared_payload_chain_v2(
        payloads,
        lambda payload, implementation: payload["x"] + 1,
        "default",
        "cuda:0",
    )

    assert len(graph_entries) == 1
    assert chain.logical_invocations == 3
    assert [output.item() for output in chain.retained_outputs] == [1, 2, 3]
    chain.replay()
    assert replay_calls == ["replay"]


def test_prepared_core_profile_dispatches_only_prepared_core_payloads(
    monkeypatch,
    tmp_path,
):
    framework = _framework(tmp_path)
    operator = _PreparedProfileOperator()
    profiler = _FakeProfiler(operator)
    profiler_configs = []
    prepared_ids_at_profiler_creation = []

    def prepare_on_cpu(
        self,
        data,
        device,
        precision,
        implementation="default",
    ):
        del device, precision, implementation
        self.prepare_calls += 1
        return {"x": data["x"].clone()}

    original_storage_ptrs = OperatorTestFramework._device_storage_ptrs

    def cpu_backed_cuda_storage_ptrs(value, device_type):
        if device_type == "cuda":
            device_type = "cpu"
        return original_storage_ptrs(value, device_type)

    def create_profiler(config):
        profiler_configs.append(config)
        prepared_ids_at_profiler_creation.append(list(operator.prepared_ids))
        return profiler

    monkeypatch.setattr(
        _FreshCpuOperator,
        "_prepare_data_for_core_operator",
        prepare_on_cpu,
    )
    monkeypatch.setattr(
        OperatorTestFramework,
        "_device_storage_ptrs",
        staticmethod(cpu_backed_cuda_storage_ptrs),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(
        ProfilerFactory,
        "create_profiler",
        staticmethod(create_profiler),
    )

    result = framework.run_core_operator_profile_test_v2(
        operator_test=operator,
        data=operator.generate_test_data(),
        device="cuda:0",
        precision=PrecisionType.FP32,
        num_warmup=2,
        num_iterations=3,
        trace_file_path=str(tmp_path / "prepared-core-profile"),
    )

    assert operator.prepared_ids == [0, 1, 2, 3, 4]
    assert prepared_ids_at_profiler_creation == [[0, 1, 2, 3, 4]]
    assert profiler.prepared_at_start == [0, 1, 2, 3, 4]
    assert profiler.prepared_after_steps == [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]
    assert profiler.prepared_at_stop == [0, 1, 2, 3, 4]
    assert profiler.executed_at_start == [0, 1]
    assert profiler.executed_after_steps == [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
    assert profiler.executed_at_stop == [0, 1, 2, 3, 4]
    assert profiler.events == ["start", "step", "step", "step", "stop"]
    assert len(profiler_configs) == 1
    assert profiler_configs[0].backend is ProfilerBackend.CUDA
    assert profiler_configs[0].schedule_active == 3
    assert result == {
        "profile_path": str(tmp_path / "prepared-core-profile"),
        "provider": "default",
        "device": "cuda:0",
        "precision": "FP32",
        "warmup_iterations": 2,
        "profiled_invocations": 3,
        "preallocated_invocations": 5,
        "input_storage_sets_verified": 5,
        "output_storage_sets_verified": 5,
        "dispatch_mode": "eager_prepared_core",
        "profiler_is_diagnostic": True,
    }


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
        "dispatch_mode",
        "graph_capture_width",
        "graph_replays",
        "capture_timed",
        "mutable_inputs_restored",
        "profiler_is_diagnostic",
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
        "protocol_version": "operator-test-framework-v2-fresh-v6",
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
        "dispatch_mode": "eager_direct",
        "graph_capture_width": 0,
        "graph_replays": 0,
        "capture_timed": False,
        "mutable_inputs_restored": False,
        "profiler_is_diagnostic": False,
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
