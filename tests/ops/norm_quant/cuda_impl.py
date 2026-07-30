"""Raw preallocated-output fused Norm/Quant providers for NVIDIA H20-3e."""

from typing import Any, Callable, Dict, List, Optional

import torch

from operator_test_framework import DeviceType, PrecisionType

from .base import NormQuantOperatorTestBase, NormQuantVariant


class CudaNormQuantOperatorTest(NormQuantOperatorTestBase):
    """One H20 variant with its fixed list of formal providers."""

    VLLM_STATIC_RMS = "cuda_vllm_rms_norm_static_fp8_quant_out"
    FLASHINFER_STATIC_RMS = (
        "cuda_flashinfer_rmsnorm_quant_fp8_out_pdl"
    )
    VLLM_STATIC_ADD = (
        "cuda_vllm_fused_add_rms_norm_static_fp8_quant_out"
    )
    FLASHINFER_STATIC_ADD = (
        "cuda_flashinfer_fused_add_rmsnorm_quant_fp8_out_pdl"
    )
    VLLM_DYNAMIC_ADD = (
        "cuda_vllm_fused_add_rms_norm_dynamic_per_token_fp8_quant_out"
    )

    _PROVIDERS = {
        NormQuantVariant.RMS_NORM_STATIC_FP8: (
            VLLM_STATIC_RMS,
            FLASHINFER_STATIC_RMS,
        ),
        NormQuantVariant.ADD_RMS_NORM_STATIC_FP8: (
            VLLM_STATIC_ADD,
            FLASHINFER_STATIC_ADD,
        ),
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8: (
            VLLM_DYNAMIC_ADD,
        ),
    }
    _VLLM_SYMBOLS = {
        VLLM_STATIC_RMS: "rms_norm_static_fp8_quant",
        VLLM_STATIC_ADD: "fused_add_rms_norm_static_fp8_quant",
        VLLM_DYNAMIC_ADD: "rms_norm_dynamic_per_token_quant",
    }
    _FLASHINFER_SYMBOLS = {
        FLASHINFER_STATIC_RMS: "rmsnorm_quant",
        FLASHINFER_STATIC_ADD: "fused_add_rmsnorm_quant",
    }

    def __init__(
        self,
        variant: NormQuantVariant,
        precision: PrecisionType,
    ):
        if variant not in self._PROVIDERS or precision is not PrecisionType.FP8:
            variant_name = (
                variant.value
                if isinstance(variant, NormQuantVariant)
                else repr(variant)
            )
            precision_name = (
                precision.name
                if isinstance(precision, PrecisionType)
                else repr(precision)
            )
            raise ValueError(
                "CUDA NormQuant does not support "
                f"variant={variant_name}, precision={precision_name}"
            )
        super().__init__(variant, precision)
        self.supported_devices = [DeviceType.GPU]

    @staticmethod
    def _copy_to_device(
        value: torch.Tensor,
        *,
        device: str,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return value.to(
            device=device,
            dtype=dtype or value.dtype,
            copy=True,
        )

    @staticmethod
    def _cuda_device_name(device: str) -> Optional[str]:
        if not device.startswith("cuda"):
            return None
        try:
            return str(torch.cuda.get_device_name(device))
        except (AssertionError, RuntimeError, TypeError, ValueError):
            return None

    def get_formal_implementations(self, device: str) -> List[str]:
        if self._cuda_device_name(device) != "NVIDIA H20-3e":
            return []
        return list(self._PROVIDERS[self.variant])

    def get_available_implementations(self, device: str) -> List[str]:
        return self.get_formal_implementations(device)

    def _resolve_implementation(
        self,
        device: str,
        implementation: str,
    ) -> str:
        formal = self.get_formal_implementations(device)
        if implementation == "default":
            if not formal:
                raise ValueError(
                    f"{self.operator_name} requires exact device name "
                    f"'NVIDIA H20-3e'; unsupported device {device}"
                )
            return formal[0]
        if implementation not in formal:
            raise ValueError(
                f"implementation {implementation!r} requires exact device "
                f"name 'NVIDIA H20-3e'; device={device}, formal={formal}"
            )
        return implementation

    @staticmethod
    def _resolve_vllm_callable(symbol: str) -> Callable[..., None]:
        try:
            import vllm._custom_ops  # noqa: F401
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "vLLM custom ops could not be registered"
            ) from error
        namespace = getattr(torch.ops, "_C", None)
        operator = getattr(namespace, symbol, None)
        if not callable(operator):
            raise RuntimeError(
                f"raw vLLM operator torch.ops._C.{symbol} is unavailable"
            )
        return operator

    @staticmethod
    def _resolve_flashinfer_callable(symbol: str) -> Callable[..., None]:
        try:
            import flashinfer
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "FlashInfer runtime is unavailable"
            ) from error
        operator = getattr(flashinfer, symbol, None)
        if not callable(operator):
            raise RuntimeError(
                f"FlashInfer operator flashinfer.{symbol} is unavailable"
            )
        return operator

    def _prepare_data_for_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Dict[str, Any]:
        """Copy fresh inputs and resolve one raw callable before timing."""
        resolved = self._resolve_implementation(device, implementation)
        if precision is not PrecisionType.FP8:
            raise ValueError("CUDA NormQuant requires PrecisionType.FP8")
        x_source = data["x"]
        weight_source = data["weight"]
        residual_source = data.get("residual")
        if x_source.ndim != 2 or weight_source.ndim != 1:
            raise ValueError("NormQuant requires x=[tokens, hidden], weight=[hidden]")
        if x_source.shape[1] != weight_source.shape[0]:
            raise ValueError("NormQuant x and weight hidden dimensions differ")
        if self.is_add_variant and (
            residual_source is None
            or residual_source.shape != x_source.shape
        ):
            raise ValueError("Add NormQuant requires a shape-matched residual")

        x = self._copy_to_device(
            x_source,
            device=device,
            dtype=torch.bfloat16,
        )
        weight = self._copy_to_device(
            weight_source,
            device=device,
            dtype=torch.bfloat16,
        )
        prepared: Dict[str, Any] = {
            "implementation": resolved,
            "x": x,
            "weight": weight,
            "eps": float(data["eps"]),
        }
        if residual_source is not None:
            prepared["residual"] = self._copy_to_device(
                residual_source,
                device=device,
                dtype=torch.bfloat16,
            )
            prepared["residual_seed"] = self._copy_to_device(
                residual_source,
                device=device,
                dtype=torch.bfloat16,
            )

        if resolved in self._VLLM_SYMBOLS:
            prepared["op"] = self._resolve_vllm_callable(
                self._VLLM_SYMBOLS[resolved]
            )
        else:
            prepared["op"] = self._resolve_flashinfer_callable(
                self._FLASHINFER_SYMBOLS[resolved]
            )

        output = torch.empty(
            x.shape,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        if self.is_static_variant:
            static_quant_values = self._vllm_static_quant_values(
                data["x"],
                data.get("residual"),
                data["weight"],
                float(data["eps"]),
            )
            scale_source = self._static_scale_for_reference(
                static_quant_values
            )
            prepared["static_scale"] = self._copy_to_device(
                scale_source,
                device=device,
                dtype=torch.float32,
            ).contiguous()
            prepared["output"] = output
        else:
            scales = torch.empty(
                (x.shape[0], 1),
                dtype=torch.float32,
                device=device,
            )
            prepared["outputs"] = (output, scales)

        if resolved in self._FLASHINFER_SYMBOLS:
            self._execute_core_operator(prepared, resolved)
            self._restore_mutable_graph_inputs([prepared], resolved)
        return prepared

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> Any:
        """Issue exactly one cached raw call with preallocated outputs."""
        resolved = prepared_data["implementation"]
        if implementation not in ("default", resolved):
            raise ValueError(
                "NormQuant prepared data provenance mismatch: "
                f"prepared for {resolved!r}, requested {implementation!r}"
            )
        op = prepared_data["op"]
        x = prepared_data["x"]
        weight = prepared_data["weight"]
        eps = prepared_data["eps"]
        if resolved == self.VLLM_STATIC_RMS:
            op(
                prepared_data["output"],
                x,
                weight,
                prepared_data["static_scale"],
                eps,
            )
            return prepared_data["output"]
        if resolved == self.VLLM_STATIC_ADD:
            op(
                prepared_data["output"],
                x,
                prepared_data["residual"],
                weight,
                prepared_data["static_scale"],
                eps,
            )
            return prepared_data["output"]
        if resolved == self.VLLM_DYNAMIC_ADD:
            output, scales = prepared_data["outputs"]
            op(
                output,
                x,
                weight,
                scales,
                eps,
                None,
                prepared_data["residual"],
            )
            return prepared_data["outputs"]
        if resolved == self.FLASHINFER_STATIC_RMS:
            op(
                prepared_data["output"],
                x,
                weight,
                prepared_data["static_scale"],
                eps,
                enable_pdl=True,
            )
            return prepared_data["output"]
        if resolved == self.FLASHINFER_STATIC_ADD:
            op(
                prepared_data["output"],
                x,
                prepared_data["residual"],
                weight,
                prepared_data["static_scale"],
                eps,
                enable_pdl=True,
            )
            return prepared_data["output"]
        raise ValueError(f"unsupported prepared CUDA provider {resolved!r}")

    def _declares_preallocated_output_contract(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> bool:
        resolved = prepared_data.get("implementation", implementation)
        return resolved in self._PROVIDERS[self.variant]
