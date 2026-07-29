"""H20 FP8 W8A8 GroupGemm providers backed by vLLM CUTLASS."""

from typing import Any, Dict, List, Optional

import torch

from fp8_utils import (
    quantize_fp8_per_row,
    quantize_fp8_weight_per_channel,
    resolve_vllm_cutlass_grouped_mm,
    resolve_vllm_cutlass_scaled_mm,
)
from groupgemm.base_groupgemm import BaseGroupGemmOperatorTest
from operator_test_framework import DeviceType, PrecisionType


class GroupGemmFp8OperatorTest(BaseGroupGemmOperatorTest):
    """Preallocated-output FP8 GroupGemm benchmark provider for H20."""

    CUDA_GROUPED_IMPLEMENTATION = (
        "cuda_vllm_cutlass_grouped_gemm_fp8_bf16"
    )
    CUDA_EXPERT_LOOP_IMPLEMENTATION = (
        "cuda_vllm_cutlass_scaled_mm_fp8_bf16_expert_loop"
    )

    # H20 grouped GEMM fails internally for the first formal point
    # (M=64, E=8, eight rows per expert). Keep one provider across the full
    # formal matrix and expose grouped GEMM only as an explicit diagnostic.
    CUDA_IMPLEMENTATION = CUDA_EXPERT_LOOP_IMPLEMENTATION
    CUDA_DIAGNOSTIC_IMPLEMENTATION = CUDA_GROUPED_IMPLEMENTATION

    def __init__(
        self,
        num_experts: int = 8,
        hidden_dim: int = 7168,
        out_channel: int = 4096,
        use_nz_format: bool = False,
    ):
        super().__init__(
            operator_name="GroupGemm_FP8",
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            out_channel=out_channel,
            use_nz_format=use_nz_format,
        )
        self.supported_precisions = [PrecisionType.FP8]
        self.supported_devices = [DeviceType.GPU]

    def get_precision_config(self) -> Dict[str, Any]:
        return {
            "input_dtype": torch.float8_e4m3fn,
            "weight_dtype": torch.float8_e4m3fn,
            "scale_dtype": torch.float32,
            "output_dtype": torch.bfloat16,
            "input_dtype_size": 1,
            "weight_dtype_size": 1,
            "output_dtype_size": 2,
        }

    def generate_precision_specific_data(
        self,
        seq_len: int,
        num_experts: int,
        hidden_dim: int,
        out_channel: int,
    ) -> Dict[str, Any]:
        return {
            "x": torch.randn(seq_len, hidden_dim, dtype=torch.float32),
            "weight": torch.randn(
                num_experts,
                hidden_dim,
                out_channel,
                dtype=torch.float32,
            ),
            "bias": None,
            "scale": None,
            "offset": None,
            "antiquant_scale": None,
            "antiquant_offset": None,
            "per_token_scale": None,
        }

    def get_npu_grouped_matmul_kwargs(
        self,
        data: Dict[str, Any],
        group_list: torch.Tensor,
    ) -> Dict[str, Any]:
        del data, group_list
        raise ValueError("FP8 GroupGemm is supported only on CUDA SM90/H20")

    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        counts = self._group_counts(data)
        outputs = []
        start = 0
        for expert, rows in enumerate(counts):
            end = start + rows
            outputs.append(
                data["x"][start:end].float()
                @ data["weight"][expert].float()
            )
            start = end
        return torch.cat(outputs, dim=0).to(torch.bfloat16)

    @staticmethod
    def _cutlass_grouped_mm():
        return resolve_vllm_cutlass_grouped_mm()

    @staticmethod
    def _cutlass_scaled_mm():
        return resolve_vllm_cutlass_scaled_mm()

    @staticmethod
    def _cuda_device_capability(device: str) -> Optional[tuple[int, int]]:
        if not device.startswith("cuda"):
            return None
        try:
            return tuple(torch.cuda.get_device_capability(device))
        except (AssertionError, RuntimeError, TypeError, ValueError):
            return None

    def get_formal_implementations(self, device: str) -> List[str]:
        if self._cuda_device_capability(device) == (9, 0):
            return [self.CUDA_IMPLEMENTATION]
        return []

    def get_available_implementations(self, device: str) -> List[str]:
        if self._cuda_device_capability(device) == (9, 0):
            return [
                self.CUDA_IMPLEMENTATION,
                self.CUDA_DIAGNOSTIC_IMPLEMENTATION,
            ]
        return []

    def _resolve_implementation(self, device: str, implementation: str) -> str:
        available = self.get_available_implementations(device)
        if implementation == "default":
            formal = self.get_formal_implementations(device)
            if not formal:
                raise ValueError(
                    "FP8 GroupGemm requires an exact CUDA SM90/H20 device; "
                    f"unsupported device {device}"
                )
            return formal[0]
        if implementation not in available:
            raise ValueError(
                f"implementation {implementation!r} requires exact CUDA "
                f"SM90/H20; device={device}, available={available}"
            )
        return implementation

    def _prepare_data_for_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Dict[str, Any]:
        implementation = self._resolve_implementation(device, implementation)
        if precision is not PrecisionType.FP8:
            raise ValueError(
                "FP8 GroupGemm provider requires PrecisionType.FP8"
            )
        if self.use_nz_format:
            raise ValueError(
                "FP8 GroupGemm does not support the Ascend NZ weight format"
            )

        counts = self._group_counts(data)
        hidden_dim = int(data["x"].shape[1])
        out_channel = int(data["weight"].shape[2])
        if hidden_dim % 16 != 0 or out_channel % 16 != 0:
            raise ValueError(
                "FP8 CUTLASS GroupGemm requires K and N to be multiples "
                f"of 16; got K={hidden_dim}, N={out_channel}"
            )
        if implementation == self.CUDA_DIAGNOSTIC_IMPLEMENTATION:
            invalid_experts = [
                expert
                for expert, rows in enumerate(counts)
                if rows < 16 or rows % 16 != 0
            ]
            if invalid_experts:
                raise ValueError(
                    "H20 grouped FP8 diagnostic requires every expert row "
                    "count to be at least 16 and 16-aligned; "
                    f"invalid experts={invalid_experts}, rows={counts}"
                )
            operator = self._cutlass_grouped_mm()
        else:
            operator = self._cutlass_scaled_mm()

        input_source = data["x"].to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        weight_source = data["weight"].to(
            device=device,
            dtype=torch.float32,
            copy=True,
        )
        activation_fp8, activation_scale = quantize_fp8_per_row(input_source)

        quantized_weights = []
        weight_scales = []
        for expert in range(len(counts)):
            weight_fp8_nk, weight_scale_n1 = (
                quantize_fp8_weight_per_channel(
                    weight_source[expert].t().contiguous()
                )
            )
            quantized_weights.append(weight_fp8_nk)
            weight_scales.append(weight_scale_n1.reshape(-1))
        weight_fp8_ekn = torch.stack(quantized_weights).transpose(1, 2)
        weight_scale_en = torch.stack(weight_scales)
        output = torch.empty(
            input_source.shape[0],
            out_channel,
            dtype=torch.bfloat16,
            device=device,
        )

        prepared = {
            "_implementation": implementation,
            "op": operator,
            "A": activation_fp8,
            "B": weight_fp8_ekn,
            "scale_a": activation_scale,
            "scale_b": weight_scale_en,
            "output": output,
        }
        if implementation == self.CUDA_IMPLEMENTATION:
            expert_inputs = []
            expert_outputs = []
            start = 0
            for expert, rows in enumerate(counts):
                end = start + rows
                expert_outputs.append(output[start:end])
                expert_inputs.append(
                    (
                        activation_fp8[start:end],
                        weight_fp8_ekn[expert],
                        activation_scale[start:end],
                        weight_scale_en[expert].reshape(1, out_channel),
                        None,
                    )
                )
                start = end
            prepared.update(
                _kernel="vllm_cutlass_scaled_mm_expert_loop",
                expert_inputs=tuple(expert_inputs),
                expert_outputs=tuple(expert_outputs),
            )
            return prepared

        expert_offsets = torch.tensor(
            [sum(counts[:expert]) for expert in range(len(counts))],
            dtype=torch.int64,
            device=device,
        )
        swap_ab = input_source.shape[0] <= 64
        problem_sizes = torch.tensor(
            [
                (
                    [out_channel, rows, hidden_dim]
                    if swap_ab
                    else [rows, out_channel, hidden_dim]
                )
                for rows in counts
            ],
            dtype=torch.int32,
            device=device,
        )
        a_strides = torch.full(
            (len(counts),),
            activation_fp8.stride(0),
            dtype=torch.int64,
            device=device,
        )
        b_strides = torch.full(
            (len(counts),),
            weight_fp8_ekn[0].stride(1),
            dtype=torch.int64,
            device=device,
        )
        c_strides = torch.full(
            (len(counts),),
            output.stride(0),
            dtype=torch.int64,
            device=device,
        )
        prepared.update(
            _kernel="vllm_cutlass_grouped_gemm",
            expert_offsets=expert_offsets,
            problem_sizes=problem_sizes,
            a_strides=a_strides,
            b_strides=b_strides,
            c_strides=c_strides,
        )
        return prepared

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> torch.Tensor:
        del implementation
        if prepared_data["_implementation"] == self.CUDA_IMPLEMENTATION:
            for expert_output, expert_input in zip(
                prepared_data["expert_outputs"],
                prepared_data["expert_inputs"],
            ):
                prepared_data["op"](expert_output, *expert_input)
            return prepared_data["output"]

        prepared_data["op"](
            prepared_data["output"],
            prepared_data["A"],
            prepared_data["B"],
            prepared_data["scale_a"],
            prepared_data["scale_b"],
            prepared_data["expert_offsets"],
            prepared_data["problem_sizes"],
            prepared_data["a_strides"],
            prepared_data["b_strides"],
            prepared_data["c_strides"],
            True,
            True,
        )
        return prepared_data["output"]

    def calculate_bandwidth(
        self,
        data: Dict[str, Any],
        avg_time_ms: float,
    ) -> Optional[float]:
        if avg_time_ms <= 0:
            return None
        seq_len = int(data.get("seq_len", 0))
        num_experts = int(data.get("num_experts", 0))
        hidden_dim = int(data.get("hidden_dim", 0))
        out_channel = int(data.get("out_channel", 0))
        if min(seq_len, num_experts, hidden_dim, out_channel) <= 0:
            return None

        input_bytes = seq_len * hidden_dim
        weight_bytes = num_experts * hidden_dim * out_channel
        output_bytes = 2 * seq_len * out_channel
        scale_bytes = 4 * (seq_len + num_experts * out_channel)
        total_bytes = (
            input_bytes
            + weight_bytes
            + output_bytes
            + scale_bytes
        )
        return total_bytes / ((avg_time_ms / 1000.0) * 1e9)

    def _declares_preallocated_output_contract(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> bool:
        impl = prepared_data.get("_implementation", implementation)
        return impl in {
            self.CUDA_IMPLEMENTATION,
            self.CUDA_DIAGNOSTIC_IMPLEMENTATION,
        }
