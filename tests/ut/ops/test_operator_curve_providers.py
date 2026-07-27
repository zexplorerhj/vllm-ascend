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

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from add.add_operator import AddOperatorTest  # noqa: E402
from flashattention.base import FlashAttentionOperatorTest  # noqa: E402
import flashattention.impl as flashattention_impl  # noqa: E402
from groupgemm.groupgemm_bf16 import GroupGemmBF16OperatorTest  # noqa: E402
from groupgemm.groupgemm_int8 import GroupGemmOperatorTest  # noqa: E402
from linear.linear_operator import LinearOperatorTest  # noqa: E402
from operator_test_framework import OperatorTestFramework  # noqa: E402
from paged_attention.base import PagedAttentionOperatorTest  # noqa: E402
from paged_attention.cuda_impl import FlashInferPagedKVImpl  # noqa: E402
from recurrent_gated_delta_rule.base import (  # noqa: E402
    RecurrentGatedDeltaRuleOperatorTest,
)
from rmsnorm.rmsnorm_operator import RMSNormOperatorTest  # noqa: E402
import rmsnorm.rmsnorm_operator as rmsnorm_module  # noqa: E402


@pytest.mark.parametrize(
    ("factory", "cuda_names", "npu_names"),
    [
        (
            AddOperatorTest,
            ["cuda_torch_add_out"],
            ["npu_torch_add_out"],
        ),
        (
            LinearOperatorTest,
            ["cuda_torch_mm_out"],
            ["npu_torch_linear"],
        ),
        (
            RMSNormOperatorTest,
            ["cuda_vllm_rms_norm_out"],
            ["npu_torch_npu_rms_norm"],
        ),
        (
            FlashAttentionOperatorTest,
            [
                "cuda_sdpa_flash_attention",
                "cuda_flash_attn_func",
            ],
            ["npu_flash_attention"],
        ),
        (
            GroupGemmBF16OperatorTest,
            ["cuda_bmm_balanced_grouped_mm_jagged_bf16"],
            ["npu_grouped_matmul"],
        ),
        (
            GroupGemmOperatorTest,
            ["cuda_vllm_cutlass_scaled_mm_bf16"],
            ["npu_grouped_matmul"],
        ),
        (
            PagedAttentionOperatorTest,
            ["cuda_flashinfer_fa2"],
            ["npu_fused_infer_attention_score"],
        ),
        (
            RecurrentGatedDeltaRuleOperatorTest,
            ["cuda_vllm_fla_direct_out"],
            ["npu_cann_builtin"],
        ),
    ],
)
def test_formal_provider_names_are_deterministic_without_accelerator_imports(
    factory,
    cuda_names,
    npu_names,
):
    operator = factory()

    assert operator.get_formal_implementations("cuda:7") == cuda_names
    assert operator.get_formal_implementations("npu:3") == npu_names


def test_paged_attention_formal_selection_never_returns_debug_fallbacks():
    operator = PagedAttentionOperatorTest()

    assert operator.get_preferred_implementation("cuda:0") == (
        "cuda_flashinfer_fa2"
    )
    assert operator.get_preferred_implementation("npu:0") == (
        "npu_fused_infer_attention_score"
    )
    assert operator.get_formal_implementations("cuda:0") == [
        "cuda_flashinfer_fa2"
    ]
    assert operator.get_formal_implementations("npu:0") == [
        "npu_fused_infer_attention_score"
    ]


def test_paged_attention_formal_prepare_rejects_block_size_256_before_backend():
    operator = PagedAttentionOperatorTest()

    with pytest.raises(ValueError, match="block_size=128"):
        operator._prepare_data_for_core_operator(
            {"block_size": 256},
            "cuda:0",
            object(),
            "cuda_flashinfer_fa2",
        )


def test_groupgemm_int8_formal_selection_never_uses_raw_int32_fallback(
    monkeypatch,
):
    operator = GroupGemmOperatorTest()
    monkeypatch.setattr(
        operator,
        "_cuda_cutlass_scaled_mm_callable",
        lambda: None,
    )
    monkeypatch.setattr(torch, "_int_mm", object(), raising=False)

    assert operator.get_formal_implementations("cuda:0") == [
        operator.CUDA_INT8_IMPLEMENTATION
    ]
    assert operator.CUDA_INT8_FALLBACK_IMPLEMENTATION not in (
        operator.get_formal_implementations("cuda:0")
    )


def test_groupgemm_bf16_formal_provider_uses_preallocated_bmm_output():
    operator = GroupGemmBF16OperatorTest(
        num_experts=2,
        hidden_dim=8,
        out_channel=8,
    )
    data = operator.generate_test_data(
        seq_len=4,
        num_experts=2,
        hidden_dim=8,
        out_channel=8,
    )
    prepared = operator._prepare_cuda_core_data(
        data,
        "cpu",
        operator.CUDA_BF16_IMPLEMENTATION,
    )

    result = operator._execute_core_operator(
        prepared,
        operator.CUDA_BF16_IMPLEMENTATION,
    )

    assert result.untyped_storage().data_ptr() == (
        prepared["output"].untyped_storage().data_ptr()
    )


def test_groupgemm_int8_formal_provider_calls_low_level_out_kernel(monkeypatch):
    operator = GroupGemmOperatorTest(
        num_experts=2,
        hidden_dim=16,
        out_channel=16,
    )
    data = operator.generate_test_data(
        seq_len=4,
        num_experts=2,
        hidden_dim=16,
        out_channel=16,
    )
    output_arguments = []

    def fake_cutlass_scaled_mm(
        output,
        expert_x,
        expert_weight,
        token_scale,
        weight_scale,
        bias,
    ):
        del expert_x, expert_weight, token_scale, weight_scale, bias
        output_arguments.append(output)
        output.zero_()

    monkeypatch.setattr(
        operator,
        "_cuda_cutlass_scaled_mm_callable",
        lambda: fake_cutlass_scaled_mm,
    )
    prepared = operator._prepare_cuda_core_data(
        data,
        "cpu",
        operator.CUDA_INT8_IMPLEMENTATION,
    )

    result = operator._execute_core_operator(
        prepared,
        operator.CUDA_INT8_IMPLEMENTATION,
    )

    assert result is prepared["expert_outputs"]
    assert output_arguments == list(prepared["expert_outputs"])


def _fake_accelerator_tensor_to(monkeypatch, expected_device):
    original_to = torch.Tensor.to
    requested_devices = []

    def fake_to(tensor, *args, **kwargs):
        device = kwargs.pop("device", None)
        if device is None and args and isinstance(args[0], str):
            device, args = args[0], args[1:]
        if device is not None:
            requested_devices.append(device)
            assert device == expected_device
        kwargs.pop("copy", None)
        return original_to(tensor, *args, **kwargs).clone()

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    return requested_devices


@pytest.mark.parametrize(
    ("factory", "device", "implementation"),
    [
        (AddOperatorTest, "npu:0", AddOperatorTest.CUDA_IMPLEMENTATION),
        (AddOperatorTest, "cpu", AddOperatorTest.CUDA_IMPLEMENTATION),
        (AddOperatorTest, "cuda:0", AddOperatorTest.NPU_IMPLEMENTATION),
        (AddOperatorTest, "cpu", AddOperatorTest.NPU_IMPLEMENTATION),
        (LinearOperatorTest, "npu:0", LinearOperatorTest.CUDA_IMPLEMENTATION),
        (LinearOperatorTest, "cpu", LinearOperatorTest.CUDA_IMPLEMENTATION),
        (LinearOperatorTest, "cuda:0", LinearOperatorTest.NPU_IMPLEMENTATION),
        (LinearOperatorTest, "cpu", LinearOperatorTest.NPU_IMPLEMENTATION),
        (RMSNormOperatorTest, "npu:0", RMSNormOperatorTest.CUDA_IMPLEMENTATION),
        (RMSNormOperatorTest, "cpu", RMSNormOperatorTest.CUDA_IMPLEMENTATION),
        (RMSNormOperatorTest, "cuda:0", RMSNormOperatorTest.NPU_IMPLEMENTATION),
        (RMSNormOperatorTest, "cpu", RMSNormOperatorTest.NPU_IMPLEMENTATION),
    ],
)
def test_explicit_formal_provider_rejects_cross_device_request(
    factory,
    device,
    implementation,
):
    operator = factory()

    with pytest.raises(ValueError, match="不适用于|not formal"):
        operator._prepare_data_for_core_operator(
            {},
            device,
            SimpleNamespace(value=torch.float32),
            implementation,
        )


@pytest.mark.parametrize(
    "factory",
    [AddOperatorTest, LinearOperatorTest, RMSNormOperatorTest],
)
def test_default_formal_provider_rejects_unsupported_device(factory):
    operator = factory()

    with pytest.raises(ValueError, match="不支持设备|unsupported device"):
        operator._prepare_data_for_core_operator(
            {},
            "cpu",
            SimpleNamespace(value=torch.float32),
            "default",
        )


@pytest.mark.parametrize(
    ("factory", "device", "expected"),
    [
        (AddOperatorTest, "cuda:7", AddOperatorTest.CUDA_IMPLEMENTATION),
        (AddOperatorTest, "npu:3", AddOperatorTest.NPU_IMPLEMENTATION),
        (LinearOperatorTest, "cuda:7", LinearOperatorTest.CUDA_IMPLEMENTATION),
        (LinearOperatorTest, "npu:3", LinearOperatorTest.NPU_IMPLEMENTATION),
        (RMSNormOperatorTest, "cuda:7", RMSNormOperatorTest.CUDA_IMPLEMENTATION),
        (RMSNormOperatorTest, "npu:3", RMSNormOperatorTest.NPU_IMPLEMENTATION),
    ],
)
def test_default_formal_provider_resolves_for_requested_device(
    factory,
    device,
    expected,
):
    assert factory()._resolve_implementation(device, "default") == expected


def test_rmsnorm_timed_npu_result_keeps_auxiliary_output(monkeypatch):
    operator = RMSNormOperatorTest()
    primary = torch.tensor([1.0])
    auxiliary = torch.tensor([2.0])
    native_result = (primary, auxiliary)
    requested_devices = _fake_accelerator_tensor_to(monkeypatch, "npu:0")
    monkeypatch.setattr(
        rmsnorm_module,
        "torch_npu",
        SimpleNamespace(npu_rms_norm=lambda *args, **kwargs: native_result),
    )

    prepared = operator._prepare_data_for_core_operator(
        {
            "x": torch.tensor([3.0]),
            "gamma": torch.tensor([4.0]),
            "eps": 1.0e-6,
        },
        "npu:0",
        SimpleNamespace(value=torch.float32),
        "default",
    )
    timed_result = operator._execute_core_operator(
        prepared,
        "default",
    )
    correctness_result = operator._primary_output(timed_result)

    assert requested_devices == ["npu:0", "npu:0"]
    assert timed_result is native_result
    assert correctness_result is primary


def test_paged_attention_timed_result_keeps_auxiliary_output():
    implementation = "npu_fused_infer_attention_score"
    operator = PagedAttentionOperatorTest()
    primary = torch.tensor([1.0])
    auxiliary = torch.tensor([2.0])
    native_result = (primary, auxiliary)
    operator.implementations[implementation] = SimpleNamespace(
        execute_core_operator=lambda prepared: native_result
    )

    timed_result = operator._execute_core_operator(
        {"_implementation": implementation},
        implementation,
    )
    correctness_result = operator._primary_output(timed_result)

    assert timed_result is native_result
    assert correctness_result is primary


@pytest.mark.parametrize(
    "execute_implementation",
    ["default", "npu_flash_attention"],
)
def test_flashattention_npu_framework_metadata_never_reaches_provider(
    monkeypatch,
    execute_implementation,
):
    operator = FlashAttentionOperatorTest()
    primary = torch.tensor([1.0])
    auxiliary = torch.tensor([2.0])
    log_sum_exp = torch.tensor([3.0])
    native_result = (primary, auxiliary, log_sum_exp)
    provider_calls = []
    requested_devices = _fake_accelerator_tensor_to(monkeypatch, "npu:0")

    def fake_fused_infer_attention_score(**kwargs):
        provider_calls.append(kwargs)
        assert "_implementation" not in kwargs
        return native_result

    monkeypatch.setattr(
        flashattention_impl,
        "torch_npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=(
                fake_fused_infer_attention_score
            )
        ),
    )
    prepared = operator._prepare_data_for_core_operator(
        {
            "query": torch.ones(1, 2, 4, 8),
            "key": torch.ones(1, 2, 4, 8),
            "value": torch.ones(1, 2, 4, 8),
            "num_heads": 2,
            "num_kv_heads": 2,
            "input_layout": "BNSD",
        },
        "npu:0",
        SimpleNamespace(value=torch.float32),
        "default",
    )

    timed_result = operator._execute_core_operator(
        prepared,
        execute_implementation,
    )

    assert requested_devices == ["npu:0"] * 3
    assert set(prepared) == {"_implementation", "provider_data"}
    assert prepared["_implementation"] == "npu_flash_attention"
    assert provider_calls[0].keys() == prepared["provider_data"].keys()
    assert timed_result is native_result
    assert operator._primary_output(timed_result) is primary


@pytest.mark.parametrize(
    ("implementation", "execute_implementation", "method_name"),
    [
        (
            "cuda_sdpa_flash_attention",
            "default",
            "execute_core_operator_in_active_context",
        ),
        (
            "cuda_sdpa_flash_attention",
            "cuda_sdpa_flash_attention",
            "execute_core_operator_in_active_context",
        ),
        (
            "cuda_flash_attn_func",
            "cuda_flash_attn_func",
            "execute_core_operator",
        ),
    ],
)
def test_flashattention_cuda_execute_receives_only_nested_provider_data(
    implementation,
    execute_implementation,
    method_name,
):
    operator = FlashAttentionOperatorTest()
    provider_data = {"query": torch.tensor([1.0])}
    received = []

    def execute(prepared):
        received.append(("execute_core_operator", prepared))
        return prepared["query"]

    def execute_in_active_context(prepared):
        received.append(
            ("execute_core_operator_in_active_context", prepared)
        )
        return prepared["query"]

    provider = SimpleNamespace(
        prepare_data=lambda data, device, precision: provider_data,
        execute_core_operator=execute,
        execute_core_operator_in_active_context=execute_in_active_context,
    )
    operator.implementations[implementation] = provider
    prepared = operator._prepare_data_for_core_operator(
        {},
        "cuda:0",
        SimpleNamespace(value=torch.float32),
        implementation if execute_implementation != "default" else "default",
    )

    result = operator._execute_core_operator(
        prepared,
        execute_implementation,
    )

    assert set(prepared) == {"_implementation", "provider_data"}
    assert received == [(method_name, provider_data)]
    assert result is provider_data["query"]


def test_flashattention_execute_rejects_provider_mismatch_before_dispatch():
    operator = FlashAttentionOperatorTest()
    provider_data = {
        "query": torch.tensor([1.0]),
        "key": torch.tensor([2.0]),
        "value": torch.tensor([3.0]),
        "is_causal": False,
        "scale": None,
    }
    provider_calls = []
    operator.implementations["cuda_sdpa_flash_attention"] = SimpleNamespace(
        prepare_data=lambda data, device, precision: provider_data,
        execute_core_operator_in_active_context=lambda prepared: (
            provider_calls.append(("cuda_sdpa_flash_attention", prepared))
        ),
    )
    operator.implementations["cuda_flash_attn_func"] = SimpleNamespace(
        execute_core_operator=lambda prepared: provider_calls.append(
            ("cuda_flash_attn_func", prepared)
        ),
    )
    prepared = operator._prepare_data_for_core_operator(
        {},
        "cuda:0",
        SimpleNamespace(value=torch.float16),
        "cuda_sdpa_flash_attention",
    )

    with pytest.raises(
        ValueError,
        match=(
            "cuda_sdpa_flash_attention.*"
            "cuda_flash_attn_func"
        ),
    ):
        operator._execute_core_operator(
            prepared,
            "cuda_flash_attn_func",
        )

    assert prepared == {
        "_implementation": "cuda_sdpa_flash_attention",
        "provider_data": provider_data,
    }
    assert provider_calls == []


def test_flashattention_nested_provider_data_remains_visible_to_storage_audit():
    prepared_sets = [
        {
            "_implementation": "npu_flash_attention",
            "provider_data": {"query": torch.ones(2)},
        },
        {
            "_implementation": "npu_flash_attention",
            "provider_data": {"query": torch.ones(2)},
        },
    ]

    verified_sets, pointer_count = (
        OperatorTestFramework._verify_independent_storage_sets(
            prepared_sets,
            "cpu",
            "nested FlashAttention provider data",
        )
    )

    assert verified_sets == 2
    assert pointer_count == 2


def test_add_formal_prepare_owns_fresh_inputs_and_output(monkeypatch):
    operator = AddOperatorTest()
    data = {
        "tensor_a": torch.arange(8, dtype=torch.float32),
        "tensor_b": torch.arange(8, dtype=torch.float32),
    }
    requested_devices = _fake_accelerator_tensor_to(monkeypatch, "cuda:0")

    first = operator._prepare_data_for_core_operator(
        data,
        "cuda:0",
        SimpleNamespace(value=torch.float32),
        "default",
    )
    second = operator._prepare_data_for_core_operator(
        data,
        "cuda:0",
        SimpleNamespace(value=torch.float32),
        "default",
    )

    assert requested_devices == ["cuda:0"] * 4
    for key in ("tensor_a", "tensor_b", "output"):
        assert first[key].untyped_storage().data_ptr() != (
            second[key].untyped_storage().data_ptr()
        )
    assert operator._execute_core_operator(
        first, operator.CUDA_IMPLEMENTATION
    ) is first["output"]


def test_linear_formal_correctness_uses_bias_free_mm_semantics(monkeypatch):
    operator = LinearOperatorTest()
    data = {
        "input": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "bias": torch.full((3,), 1000.0),
    }
    requested_devices = _fake_accelerator_tensor_to(monkeypatch, "cuda:0")
    mm_calls = []
    original_empty = torch.empty

    def fake_mm(input_tensor, weight_t, *, out):
        mm_calls.append((input_tensor, weight_t, out))
        torch.matmul(input_tensor, weight_t, out=out)
        return out

    def fake_empty(*args, **kwargs):
        if kwargs.get("device") == "cuda:0":
            kwargs = {**kwargs, "device": "cpu"}
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "mm", fake_mm)
    monkeypatch.setattr(torch, "empty", fake_empty)
    prepared = operator._prepare_data_for_core_operator(
        data,
        "cuda:0",
        SimpleNamespace(value=torch.float32),
        "default",
    )
    result = operator._execute_core_operator(prepared, "default")

    assert requested_devices == ["cuda:0", "cuda:0"]
    assert len(mm_calls) == 1
    assert mm_calls[0][0] is prepared["input"]
    assert mm_calls[0][1] is prepared["weight_t"]
    assert mm_calls[0][2] is prepared["output"]
    assert result is prepared["output"]
    assert torch.equal(result, torch.matmul(data["input"], data["weight"].t()))


def test_flashinfer_prepare_owns_fresh_zeroed_workspace(monkeypatch):
    implementation = FlashInferPagedKVImpl(workspace_bytes=32)
    original_empty = torch.empty
    original_zeros = torch.zeros
    zeros_calls = []

    def fake_empty(*args, **kwargs):
        if kwargs.get("device") == "cuda:0":
            kwargs = {**kwargs, "device": "cpu"}
        return original_empty(*args, **kwargs)

    def fake_zeros(*args, **kwargs):
        zeros_calls.append((args, kwargs.copy()))
        if kwargs.get("device") == "cuda:0":
            kwargs = {**kwargs, "device": "cpu"}
        return original_zeros(*args, **kwargs)

    class FakeWrapper:

        def __init__(
            self,
            workspace,
            kv_layout,
            use_tensor_cores,
            backend,
        ):
            self.workspace = workspace
            self.kv_layout = kv_layout
            self.use_tensor_cores = use_tensor_cores
            self.backend = backend

        def plan(self, *args, **kwargs):
            self.plan_args = args
            self.plan_kwargs = kwargs

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch, "zeros", fake_zeros)
    monkeypatch.setattr(
        implementation,
        "_copy_tensor",
        lambda tensor, *, device, dtype: tensor.to(dtype=dtype).clone(),
    )
    monkeypatch.setattr(implementation, "_wrapper_class", lambda: FakeWrapper)
    data = {
        "block_size": 128,
        "num_heads": 8,
        "num_kv_heads": 1,
        "head_size": 4,
        "context_lens": torch.tensor([128], dtype=torch.int32),
        "block_table": torch.tensor([[0]], dtype=torch.int32),
        "query": torch.ones(1, 8, 4),
        "key_cache": torch.ones(1, 128, 1, 4),
        "value_cache": torch.ones(1, 128, 1, 4),
        "scale": 0.5,
    }
    precision = SimpleNamespace(value=torch.float32)

    first = implementation.prepare_data(data, "cuda:0", precision)
    second = implementation.prepare_data(data, "cuda:0", precision)

    assert len(zeros_calls) == 2
    assert all(call[1]["device"] == "cuda:0" for call in zeros_calls)
    assert first["workspace"].untyped_storage().data_ptr() != (
        second["workspace"].untyped_storage().data_ptr()
    )
    assert torch.count_nonzero(first["workspace"]).item() == 0
    assert torch.count_nonzero(second["workspace"]).item() == 0


def test_flashattention_sdpa_context_is_selected_for_complete_repeat():
    operator = FlashAttentionOperatorTest()
    context = operator._core_operator_benchmark_context(
        "cuda:0",
        object(),
        "cuda_sdpa_flash_attention",
    )

    assert context is not None
