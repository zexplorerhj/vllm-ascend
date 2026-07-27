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
from groupgemm.groupgemm_bf16 import GroupGemmBF16OperatorTest  # noqa: E402
from groupgemm.groupgemm_int8 import GroupGemmOperatorTest  # noqa: E402
from linear.linear_operator import LinearOperatorTest  # noqa: E402
from paged_attention.base import PagedAttentionOperatorTest  # noqa: E402
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


def test_rmsnorm_timed_npu_result_keeps_auxiliary_output(monkeypatch):
    operator = RMSNormOperatorTest()
    primary = torch.tensor([1.0])
    auxiliary = torch.tensor([2.0])
    native_result = (primary, auxiliary)
    monkeypatch.setattr(
        rmsnorm_module,
        "torch_npu",
        SimpleNamespace(npu_rms_norm=lambda *args, **kwargs: native_result),
    )

    timed_result = operator._execute_core_operator(
        {
            "x": torch.tensor([3.0]),
            "gamma": torch.tensor([4.0]),
            "eps": 1.0e-6,
            "implementation": operator.NPU_IMPLEMENTATION,
            "device": "npu:0",
        },
        operator.NPU_IMPLEMENTATION,
    )
    correctness_result = operator._primary_output(timed_result)

    assert timed_result is native_result
    assert correctness_result is primary


@pytest.mark.parametrize(
    ("factory", "implementation"),
    [
        (FlashAttentionOperatorTest, "npu_flash_attention"),
        (PagedAttentionOperatorTest, "npu_fused_infer_attention_score"),
    ],
)
def test_attention_timed_result_keeps_auxiliary_output(
    factory,
    implementation,
):
    operator = factory()
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


def test_add_formal_prepare_owns_fresh_inputs_and_output():
    operator = AddOperatorTest()
    data = {
        "tensor_a": torch.arange(8, dtype=torch.float32),
        "tensor_b": torch.arange(8, dtype=torch.float32),
    }

    first = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        SimpleNamespace(value=torch.float32),
        operator.CUDA_IMPLEMENTATION,
    )
    second = operator._prepare_data_for_core_operator(
        data,
        "cpu",
        SimpleNamespace(value=torch.float32),
        operator.CUDA_IMPLEMENTATION,
    )

    for key in ("tensor_a", "tensor_b", "output"):
        assert first[key].untyped_storage().data_ptr() != (
            second[key].untyped_storage().data_ptr()
        )
    assert operator._execute_core_operator(
        first, operator.CUDA_IMPLEMENTATION
    ) is first["output"]


def test_add_formal_provider_is_usable_by_correctness_entry():
    operator = AddOperatorTest()
    data = {
        "tensor_a": torch.arange(8, dtype=torch.float32),
        "tensor_b": torch.arange(8, dtype=torch.float32),
    }

    result = operator.run_device_implementation(
        data,
        "cpu",
        SimpleNamespace(value=torch.float32),
        operator.CUDA_IMPLEMENTATION,
    )

    assert torch.equal(result, data["tensor_a"] + data["tensor_b"])


def test_linear_formal_correctness_uses_bias_free_mm_semantics():
    operator = LinearOperatorTest()
    data = {
        "input": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "bias": torch.full((3,), 1000.0),
    }

    result = operator.run_device_implementation(
        data,
        "cpu",
        SimpleNamespace(value=torch.float32),
        operator.CUDA_IMPLEMENTATION,
    )

    assert torch.equal(result, torch.mm(data["input"], data["weight"].t()))


def test_flashattention_sdpa_context_is_selected_for_complete_repeat():
    operator = FlashAttentionOperatorTest()
    context = operator._core_operator_benchmark_context(
        "cuda:0",
        object(),
        "cuda_sdpa_flash_attention",
    )

    assert context is not None
