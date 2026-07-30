"""Platform-independent fused RMSNorm/quantization benchmark contract."""

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import torch

from operator_test_framework import BaseOperatorTest, PrecisionType


class NormQuantVariant(str, Enum):
    RMS_NORM_STATIC_FP8 = "rms_norm_quant"
    ADD_RMS_NORM_STATIC_FP8 = "add_rms_norm_quant"
    ADD_RMS_NORM_DYNAMIC_FP8 = "add_rms_norm_dynamic_quant"
    RMS_NORM_DYNAMIC_MX = "rms_norm_dynamic_mx_quant"
    ADD_RMS_NORM_DYNAMIC_MX = "add_rms_norm_dynamic_mx_quant"


_ADD_VARIANTS = frozenset({
    NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
    NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
    NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
})
_STATIC_VARIANTS = frozenset({
    NormQuantVariant.RMS_NORM_STATIC_FP8,
    NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
})
_DYNAMIC_FP8_VARIANTS = frozenset({
    NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
})
_MX_VARIANTS = frozenset({
    NormQuantVariant.RMS_NORM_DYNAMIC_MX,
    NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
})
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_FP8_MIN_DYNAMIC_SCALE = 1.0 / (_FP8_MAX * 512.0)
_STATIC_FP8_TARGET_CODE = 384.0
_DYNAMIC_SCALE_RTOL = 2e-3
_FP8_CODE_MAX_ULP = 1


class NormQuantOperatorTestBase(BaseOperatorTest):
    """Platform-independent data, reference, bytes, and validation."""

    def __init__(
        self,
        variant: NormQuantVariant,
        precision: PrecisionType,
    ):
        if not isinstance(variant, NormQuantVariant):
            raise ValueError(f"invalid NormQuant variant: {variant!r}")
        if not isinstance(precision, PrecisionType):
            raise ValueError(f"invalid NormQuant precision: {precision!r}")
        super().__init__(
            f"NormQuant_{variant.value}_{precision.name}"
        )
        self.variant = variant
        self.precision = precision
        self.supported_precisions = [precision]

    @property
    def is_add_variant(self) -> bool:
        return self.variant in _ADD_VARIANTS

    @property
    def is_static_variant(self) -> bool:
        return self.variant in _STATIC_VARIANTS

    @property
    def is_dynamic_fp8_variant(self) -> bool:
        return self.variant in _DYNAMIC_FP8_VARIANTS

    @property
    def is_mx_variant(self) -> bool:
        return self.variant in _MX_VARIANTS

    def generate_test_data(
        self,
        tokens: int = 128,
        hidden: int = 7168,
        seed: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate bounded, deterministic CPU BF16 source tensors."""
        del kwargs
        if (
            not isinstance(tokens, int)
            or isinstance(tokens, bool)
            or tokens <= 0
        ):
            raise ValueError("tokens must be a positive non-bool int")
        if (
            not isinstance(hidden, int)
            or isinstance(hidden, bool)
            or hidden <= 0
        ):
            raise ValueError("hidden must be a positive non-bool int")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        x = (
            torch.randn(
                tokens,
                hidden,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.25
        ).to(torch.bfloat16)
        residual = None
        if self.is_add_variant:
            residual = (
                torch.randn(
                    tokens,
                    hidden,
                    generator=generator,
                    dtype=torch.float32,
                )
                * 0.25
            ).to(torch.bfloat16)
        weight = (
            1.0
            + torch.randn(
                hidden,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.05
        ).to(torch.bfloat16)
        return {
            "x": x,
            "residual": residual,
            "weight": weight,
            "eps": 1e-6,
            "metadata": {
                "tokens": tokens,
                "hidden": hidden,
                "variant": self.variant.value,
                "precision": self.precision.name,
            },
        }

    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """Compute the source-dtype add followed by FP32 RMSNorm."""
        x = data["x"]
        residual = data.get("residual")
        source_dtype = x.dtype
        normalized_input = (
            x.float()
            if residual is None
            else (
                x.to(source_dtype) + residual.to(source_dtype)
            ).float()
        )
        variance = normalized_input.square().mean(
            dim=-1,
            keepdim=True,
        )
        return (
            normalized_input
            * torch.rsqrt(variance + data["eps"])
            * data["weight"].float()
        )

    def run_device_implementation(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Any:
        """Run and validate one untimed prepared payload."""
        prepared = self._prepare_data_for_core_operator(
            data,
            device,
            precision,
            implementation,
        )
        result = self._execute_core_operator(prepared, implementation)
        self.validate_prepared_correctness(data, prepared, result)
        return result

    def run_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Any:
        return self.run_device_implementation(
            data,
            device,
            precision,
            implementation,
        )

    @staticmethod
    def observable_output_bytes(outputs: Any) -> int:
        """Count actual bytes held by nested, observable output tensors."""
        if isinstance(outputs, torch.Tensor):
            return outputs.numel() * outputs.element_size()
        if isinstance(outputs, dict):
            return sum(
                NormQuantOperatorTestBase.observable_output_bytes(value)
                for value in outputs.values()
            )
        if isinstance(outputs, (tuple, list)):
            return sum(
                NormQuantOperatorTestBase.observable_output_bytes(value)
                for value in outputs
            )
        return 0

    def _source_read_bytes(self, data: Dict[str, Any]) -> int:
        x = data["x"]
        residual = data.get("residual")
        weight = data["weight"]
        total = (
            x.numel() * x.element_size()
            + weight.numel() * weight.element_size()
        )
        if residual is not None:
            total += residual.numel() * residual.element_size()
        if self.is_static_variant:
            total += torch.empty((), dtype=torch.float32).element_size()
        return total

    def _add_result_bytes(self, data: Dict[str, Any]) -> int:
        residual = data.get("residual")
        if residual is None:
            return 0
        return residual.numel() * residual.element_size()

    def _derived_output_bytes(self, data: Dict[str, Any]) -> int:
        tokens = int(data["metadata"]["tokens"])
        hidden = int(data["metadata"]["hidden"])
        if self.is_static_variant:
            return tokens * hidden
        if self.is_dynamic_fp8_variant:
            return tokens * hidden + tokens * 4
        if self.is_mx_variant:
            quantized_bytes = (
                tokens * hidden
                if self.precision is PrecisionType.MXFP8
                else tokens * hidden // 2
            )
            scale_bytes = tokens * ((hidden + 63) // 64) * 2
            return quantized_bytes + scale_bytes
        raise ValueError(f"unsupported NormQuant variant {self.variant}")

    def logical_bytes(self, data: Dict[str, Any]) -> int:
        """Return bytes logically transferred by the fused contract."""
        return (
            self._source_read_bytes(data)
            + self._add_result_bytes(data)
            + self._derived_output_bytes(data)
        )

    def physical_bytes(
        self,
        data: Dict[str, Any],
        outputs: Optional[Any] = None,
    ) -> int:
        """Return bytes represented by actual packed/empty output tensors."""
        output_bytes = (
            self._derived_output_bytes(data)
            if outputs is None
            else self.observable_output_bytes(outputs)
        )
        add_result_bytes = self._add_result_bytes(data)
        if outputs is not None and self._contains_bf16_x_out(data, outputs):
            add_result_bytes = 0
        return self._source_read_bytes(data) + add_result_bytes + output_bytes

    @classmethod
    def _contains_bf16_x_out(
        cls,
        data: Dict[str, Any],
        outputs: Any,
    ) -> bool:
        if isinstance(outputs, torch.Tensor):
            return (
                outputs.dtype is torch.bfloat16
                and outputs.shape == data["x"].shape
            )
        if isinstance(outputs, dict):
            return any(
                cls._contains_bf16_x_out(data, value)
                for value in outputs.values()
            )
        if isinstance(outputs, (tuple, list)):
            return any(
                cls._contains_bf16_x_out(data, value)
                for value in outputs
            )
        return False

    def calculate_bandwidth(
        self,
        data: Dict[str, Any],
        avg_time_ms: float,
    ) -> float:
        if avg_time_ms <= 0:
            return 0.0
        return self.logical_bytes(data) / (avg_time_ms / 1000.0) / 1e9

    @staticmethod
    def _assert_result_aliases(
        expected: Iterable[torch.Tensor],
        observed: Any,
    ) -> None:
        expected_tuple = tuple(expected)
        observed_tuple = (
            tuple(observed)
            if isinstance(observed, (tuple, list))
            else (observed,)
        )
        if len(expected_tuple) != len(observed_tuple) or any(
            actual is not wanted
            for wanted, actual in zip(expected_tuple, observed_tuple)
        ):
            raise AssertionError(
                "result does not alias every prepared output buffer"
            )

    @staticmethod
    def _dequantize_fp8(
        output: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return output.float() * scale.float()

    @staticmethod
    def _vllm_dynamic_quant_values(
        x: torch.Tensor,
        residual_seed: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Mirror vLLM's BF16 intermediate before per-token absmax."""
        combined = x.float() + residual_seed.float()
        rms = torch.rsqrt(
            combined.square().mean(dim=-1, keepdim=True) + eps
        )
        return ((combined * rms).to(x.dtype) * weight).float()

    @staticmethod
    def _vllm_static_quant_values(
        x: torch.Tensor,
        residual_seed: Optional[torch.Tensor],
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Mirror vLLM's source-dtype add and final BF16 quant input."""
        normalized_input = (
            x
            if residual_seed is None
            else x.to(x.dtype) + residual_seed.to(x.dtype)
        )
        normalized_float = normalized_input.float()
        rms = torch.rsqrt(
            normalized_float.square().mean(dim=-1, keepdim=True) + eps
        )
        return (
            normalized_float * rms * weight.float()
        ).to(x.dtype).float()

    @staticmethod
    def _dynamic_scale_for_values(
        quant_values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            quant_values.abs().amax(dim=-1, keepdim=True) / _FP8_MAX
        ).clamp_min(_FP8_MIN_DYNAMIC_SCALE).to(torch.float32)

    @staticmethod
    def _quantize_expected_fp8(
        quant_values: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (quant_values / scale)
            .clamp(min=-_FP8_MAX, max=_FP8_MAX)
            .to(torch.float8_e4m3fn)
        )

    @staticmethod
    def _assert_fp8_codes_close(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        label: str,
    ) -> None:
        """Require matching sign and at most one E4M3 representable code."""
        actual_bits = actual.contiguous().view(torch.uint8)
        expected_bits = expected.contiguous().view(torch.uint8)
        actual_float = actual.float()
        expected_float = expected.float()
        both_zero = (actual_float == 0) & (expected_float == 0)
        sign_mismatch = ((actual_bits ^ expected_bits) & 0x80) != 0
        if bool((sign_mismatch & ~both_zero).any()):
            raise AssertionError(f"{label} sign mismatch")
        actual_magnitude = (actual_bits & 0x7F).to(torch.int16)
        expected_magnitude = (expected_bits & 0x7F).to(torch.int16)
        ulp_distance = (actual_magnitude - expected_magnitude).abs()
        if bool((ulp_distance > _FP8_CODE_MAX_ULP).any()):
            raise AssertionError(
                f"{label} exceeds {_FP8_CODE_MAX_ULP} E4M3 ULP"
            )

    def validate_prepared_correctness(
        self,
        data: Dict[str, Any],
        prepared: Dict[str, Any],
        result: Any,
    ) -> None:
        """Validate CUDA FP8 primary and mutable auxiliary outputs untimed."""
        expected_outputs = (
            tuple(prepared["outputs"])
            if "outputs" in prepared
            else (prepared["output"],)
        )
        self._assert_result_aliases(expected_outputs, result)
        output = expected_outputs[0]
        tokens, hidden = data["x"].shape
        if output.shape != (tokens, hidden):
            raise AssertionError(
                f"primary output shape {tuple(output.shape)} != "
                f"{(tokens, hidden)}"
            )
        if output.dtype is not torch.float8_e4m3fn:
            raise AssertionError(
                f"primary output dtype {output.dtype} is not FP8 E4M3"
            )
        output_codes = output.float()
        if not bool(torch.isfinite(output_codes).all()):
            raise AssertionError("primary FP8 output contains non-finite codes")
        if bool((output_codes.abs() > _FP8_MAX).any()):
            raise AssertionError("primary FP8 output exceeds E4M3 code range")

        reference = self.run_cpu_reference(data).to(output.device)
        if self.is_static_variant:
            scale = prepared["static_scale"]
            if (
                scale.shape != (1,)
                or scale.dtype is not torch.float32
                or not scale.is_contiguous()
            ):
                raise AssertionError("invalid static scale contract")
            quant_values = self._vllm_static_quant_values(
                prepared["x"],
                prepared.get("residual_seed"),
                prepared["weight"],
                prepared["eps"],
            )
            calibration_values = self._vllm_static_quant_values(
                data["x"],
                data.get("residual"),
                data["weight"],
                float(data["eps"]),
            )
            expected_scale = self._static_scale_for_reference(
                calibration_values
            ).to(scale.device)
            torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
            expected_saturation = (
                (quant_values / expected_scale).abs() >= _FP8_MAX
            )
            observed_saturation = output_codes.abs() >= _FP8_MAX
            if not torch.equal(expected_saturation, observed_saturation):
                raise AssertionError(
                    "static FP8 saturation semantics mismatch"
                )
            expected_codes = self._quantize_expected_fp8(
                quant_values,
                expected_scale,
            )
            self._assert_fp8_codes_close(
                output,
                expected_codes,
                label="static FP8 codes",
            )
        else:
            scale = expected_outputs[1]
            if scale.shape != (tokens, 1) or scale.dtype is not torch.float32:
                raise AssertionError(
                    "dynamic scale must be FP32 with shape (tokens, 1)"
                )
            if not bool(torch.isfinite(scale).all()) or bool(
                (scale <= 0).any()
            ):
                raise AssertionError(
                    "dynamic scale must contain positive finite values"
                )
            quant_values = self._vllm_dynamic_quant_values(
                prepared["x"],
                prepared["residual_seed"],
                prepared["weight"],
                prepared["eps"],
            )
            expected_scale = self._dynamic_scale_for_values(quant_values)
            try:
                torch.testing.assert_close(
                    scale,
                    expected_scale,
                    rtol=_DYNAMIC_SCALE_RTOL,
                    atol=0,
                )
            except AssertionError as error:
                raise AssertionError(
                    "dynamic scale semantics mismatch"
                ) from error
            expected_codes = self._quantize_expected_fp8(
                quant_values,
                expected_scale,
            )
            self._assert_fp8_codes_close(
                output,
                expected_codes,
                label="dynamic FP8 codes",
            )

        dequantized = self._dequantize_fp8(output, scale)
        torch.testing.assert_close(
            dequantized,
            reference,
            rtol=0.12,
            atol=0.12,
        )

        if self.is_add_variant:
            residual = prepared["residual"]
            residual_seed = prepared["residual_seed"]
            expected_residual = (
                prepared["x"].to(residual.dtype)
                + residual_seed.to(residual.dtype)
            )
            torch.testing.assert_close(
                residual,
                expected_residual,
                rtol=0,
                atol=0,
            )

    @staticmethod
    def _static_scale_for_reference(
        reference: torch.Tensor,
    ) -> torch.Tensor:
        maximum = reference.float().abs().amax().clamp_min(
            _FP8_MIN_DYNAMIC_SCALE * _STATIC_FP8_TARGET_CODE
        )
        return (
            maximum / _STATIC_FP8_TARGET_CODE
        ).reshape(1).to(torch.float32)

    def _restore_mutable_graph_inputs(
        self,
        prepared_payloads: List[Any],
        implementation: str = "default",
    ) -> None:
        del implementation
        for prepared in prepared_payloads:
            residual = prepared.get("residual")
            residual_seed = prepared.get("residual_seed")
            if residual is not None and residual_seed is not None:
                residual.copy_(residual_seed)
