"""Ascend 950PR plain-FP8 W8A8 pure GroupGemm provider."""

from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

from groupgemm.base_groupgemm import BaseGroupGemmOperatorTest
from operator_test_framework import DeviceType, PrecisionType


class _BaseGroupGemmFp8NpuOperatorTest(BaseGroupGemmOperatorTest):
    """Shared preparation for 950PR FP8 pure-GMM2 providers."""

    IMPLEMENTATION = ""
    QUANTIZATION_API = ""
    PROVIDER_LABEL = ""
    K_ALIGNMENT = 16
    KERNEL = ""
    PRECISION = PrecisionType.FP8

    def __init__(
        self,
        *,
        operator_name: str,
        num_experts: int = 8,
        hidden_dim: int = 7168,
        out_channel: int = 4096,
        use_nz_format: bool = False,
    ):
        super().__init__(
            operator_name=operator_name,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            out_channel=out_channel,
            use_nz_format=use_nz_format,
        )
        self.supported_precisions = [self.PRECISION]
        self.supported_devices = [DeviceType.NPU]

    @staticmethod
    def _runtime_module():
        return torch_npu

    @staticmethod
    def _copy_to_device(
        tensor: torch.Tensor,
        *,
        device: str,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        kwargs = {"device": device, "copy": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)

    @staticmethod
    def _device_index(device: str) -> int:
        if ":" not in device:
            return 0
        try:
            return int(device.rsplit(":", 1)[1])
        except (TypeError, ValueError):
            return 0

    def _npu_device_name(self, device: str) -> str:
        runtime = self._runtime_module()
        npu_api = getattr(torch, "npu", None)
        if npu_api is None and runtime is not None:
            npu_api = getattr(runtime, "npu", None)
        getter = getattr(npu_api, "get_device_name", None)
        if not callable(getter):
            return ""
        try:
            return str(getter(self._device_index(device)))
        except TypeError:
            return str(getter())
        except (AssertionError, RuntimeError, ValueError):
            return ""

    def _runtime_supported(self) -> bool:
        runtime = self._runtime_module()
        return bool(
            runtime is not None
            and callable(getattr(runtime, self.QUANTIZATION_API, None))
            and callable(getattr(runtime, "npu_grouped_matmul", None))
        )

    def _is_supported_950pr(self, device: str) -> bool:
        return bool(
            device.startswith("npu")
            and self._npu_device_name(device)
            .upper()
            .startswith("ASCEND950PR")
            and self._runtime_supported()
        )

    def get_formal_implementations(self, device: str) -> List[str]:
        return [self.IMPLEMENTATION] if self._is_supported_950pr(device) else []

    def get_available_implementations(self, device: str) -> List[str]:
        return self.get_formal_implementations(device)

    def _resolve_implementation(
        self,
        device: str,
        implementation: str,
    ) -> str:
        available = self.get_available_implementations(device)
        if implementation == "default":
            if available:
                return available[0]
            raise RuntimeError(
                f"{self.PROVIDER_LABEL} requires Ascend950PR and torch_npu "
                f"APIs {self.QUANTIZATION_API}/npu_grouped_matmul; "
                f"device={device}, device_name={self._npu_device_name(device)!r}"
            )
        if implementation not in available:
            raise ValueError(
                f"implementation {implementation!r} is unavailable on "
                f"{device}; available={available}"
            )
        return implementation

    def get_precision_config(self) -> Dict[str, Any]:
        return {
            "input_dtype": torch.float8_e4m3fn,
            "weight_dtype": torch.float8_e4m3fn,
            "scale_dtype": self._scale_storage_dtype(),
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
            "x": torch.randn(
                seq_len,
                hidden_dim,
                dtype=torch.bfloat16,
            ),
            "weight": torch.randn(
                num_experts,
                hidden_dim,
                out_channel,
                dtype=torch.bfloat16,
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
        raise RuntimeError(
            "950PR FP8 providers build their quantized pure-GMM2 payload "
            "inside _prepare_data_for_core_operator"
        )

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

    def _validate_dimensions(
        self,
        hidden_dim: int,
        out_channel: int,
    ) -> None:
        if hidden_dim % self.K_ALIGNMENT != 0 or out_channel % 16 != 0:
            raise ValueError(
                f"{self.PROVIDER_LABEL} requires K to be a multiple of "
                f"{self.K_ALIGNMENT} and N to be a multiple of 16; "
                f"got K={hidden_dim}, N={out_channel}"
            )

    def _validate_quantized_values(
        self,
        *,
        activation: torch.Tensor,
        weight_enk: torch.Tensor,
        expected_activation_shape: Tuple[int, int],
        expected_weight_shape: Tuple[int, int, int],
    ) -> None:
        if tuple(activation.shape) != expected_activation_shape:
            raise RuntimeError(
                f"{self.QUANTIZATION_API} returned activation shape "
                f"{tuple(activation.shape)}, expected "
                f"{expected_activation_shape}"
            )
        if tuple(weight_enk.shape) != expected_weight_shape:
            raise RuntimeError(
                f"{self.QUANTIZATION_API} returned weight shape "
                f"{tuple(weight_enk.shape)}, expected {expected_weight_shape}"
            )
        if (
            activation.dtype != torch.float8_e4m3fn
            or weight_enk.dtype != torch.float8_e4m3fn
        ):
            raise RuntimeError(
                f"{self.QUANTIZATION_API} must return E4M3 activation and "
                f"weight, got {activation.dtype}/{weight_enk.dtype}"
            )

    def _normalize_scales(
        self,
        *,
        activation_scale: torch.Tensor,
        weight_scale_en: torch.Tensor,
        seq_len: int,
        num_experts: int,
        hidden_dim: int,
        out_channel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _extra_grouped_matmul_kwargs(self) -> Dict[str, Any]:
        return {}

    def _scale_storage_dtype(self) -> torch.dtype:
        raise NotImplementedError

    def _scale_payload_bytes(
        self,
        *,
        seq_len: int,
        num_experts: int,
        hidden_dim: int,
        out_channel: int,
    ) -> int:
        raise NotImplementedError

    def _prepare_data_for_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Dict[str, Any]:
        implementation = self._resolve_implementation(device, implementation)
        if precision is not self.PRECISION:
            raise ValueError(
                f"{self.PROVIDER_LABEL} requires "
                f"PrecisionType.{self.PRECISION.name}"
            )
        if self.use_nz_format:
            raise ValueError(
                f"{self.PROVIDER_LABEL} requires ND E4M3 weight layout; "
                "NZ format is unsupported"
            )

        counts = self._group_counts(data)
        seq_len = int(data["x"].shape[0])
        hidden_dim = int(data["x"].shape[1])
        num_experts = len(counts)
        out_channel = int(data["weight"].shape[2])
        self._validate_dimensions(hidden_dim, out_channel)

        runtime = self._runtime_module()
        quantize = getattr(runtime, self.QUANTIZATION_API)
        input_source = self._copy_to_device(
            data["x"],
            device=device,
            dtype=torch.bfloat16,
        )
        weight_source_ekn = self._copy_to_device(
            data["weight"],
            device=device,
            dtype=torch.bfloat16,
        )
        # Dynamic quantization reduces the final K dimension. Convert the
        # framework's [E,K,N] source to [E,N,K] before one batched call.
        weight_source_enk = weight_source_ekn.transpose(1, 2).contiguous()
        activation, activation_scale = quantize(
            input_source,
            dst_type=torch.float8_e4m3fn,
        )
        weight_enk, weight_scale_en = quantize(
            weight_source_enk,
            dst_type=torch.float8_e4m3fn,
        )
        self._validate_quantized_values(
            activation=activation,
            weight_enk=weight_enk,
            expected_activation_shape=(seq_len, hidden_dim),
            expected_weight_shape=(
                num_experts,
                out_channel,
                hidden_dim,
            ),
        )
        activation_scale, weight_scale = self._normalize_scales(
            activation_scale=activation_scale,
            weight_scale_en=weight_scale_en,
            seq_len=seq_len,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            out_channel=out_channel,
        )
        weight = weight_enk.transpose(1, 2).contiguous()
        group_list = self._copy_to_device(
            data["group_list"],
            device=device,
            dtype=torch.int64,
        )

        kwargs = {
            "x": [activation],
            "weight": [weight],
            "scale": [weight_scale],
            "bias": None,
            "per_token_scale": [activation_scale],
            "split_item": 2,
            "group_list_type": 1,
            "group_type": 0,
            "group_list": group_list,
            "output_dtype": torch.bfloat16,
            **self._extra_grouped_matmul_kwargs(),
        }
        return {
            "_implementation": implementation,
            "_kernel": self.KERNEL,
            "op": runtime.npu_grouped_matmul,
            "x": activation,
            "weight": weight,
            "per_token_scale": activation_scale,
            "weight_scale": weight_scale,
            "group_list": group_list,
            "kwargs": kwargs,
        }

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> torch.Tensor:
        actual = prepared_data.get("_implementation")
        if implementation not in ("default", actual):
            raise ValueError(
                f"prepared implementation is {actual!r}, requested "
                f"{implementation!r}"
            )
        return prepared_data["op"](**prepared_data["kwargs"])[0]

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
        scale_bytes = self._scale_payload_bytes(
            seq_len=seq_len,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            out_channel=out_channel,
        )
        total_bytes = (
            input_bytes
            + weight_bytes
            + output_bytes
            + scale_bytes
        )
        return total_bytes / ((avg_time_ms / 1000.0) * 1e9)


class GroupGemmFp8NpuOperatorTest(
    _BaseGroupGemmFp8NpuOperatorTest
):
    """950PR E4M3/FP32-scale W8A8 pure GroupGemm."""

    NPU_FP8_IMPLEMENTATION = (
        "npu_grouped_matmul_fp8_e4m3_per_token_per_channel_bf16"
    )
    IMPLEMENTATION = NPU_FP8_IMPLEMENTATION
    QUANTIZATION_API = "npu_dynamic_quant"
    PROVIDER_LABEL = "950PR plain FP8 GroupGemm"
    KERNEL = "npu_grouped_matmul_fp8_gmm2"

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

    def _scale_storage_dtype(self) -> torch.dtype:
        return torch.float32

    def _normalize_scales(
        self,
        *,
        activation_scale: torch.Tensor,
        weight_scale_en: torch.Tensor,
        seq_len: int,
        num_experts: int,
        hidden_dim: int,
        out_channel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del hidden_dim
        expected_activation = (seq_len,)
        expected_weight = (num_experts, out_channel)
        if (
            tuple(activation_scale.shape) != expected_activation
            or activation_scale.dtype != torch.float32
        ):
            raise RuntimeError(
                "npu_dynamic_quant activation scale must be FP32 [M], "
                f"got {activation_scale.dtype} "
                f"{tuple(activation_scale.shape)}"
            )
        if (
            tuple(weight_scale_en.shape) != expected_weight
            or weight_scale_en.dtype != torch.float32
        ):
            raise RuntimeError(
                "npu_dynamic_quant weight scale must be FP32 [E,N], "
                f"got {weight_scale_en.dtype} "
                f"{tuple(weight_scale_en.shape)}"
            )
        return activation_scale, weight_scale_en

    def _scale_payload_bytes(
        self,
        *,
        seq_len: int,
        num_experts: int,
        hidden_dim: int,
        out_channel: int,
    ) -> int:
        del hidden_dim
        return 4 * (seq_len + num_experts * out_channel)
