"""Native fused Norm/Quant providers for Ascend 950PR."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from operator_test_framework import DeviceType, PrecisionType

from .base import NormQuantOperatorTestBase, NormQuantVariant


@dataclass(frozen=True)
class CapabilityResult:
    """One cached native capability-probe outcome."""

    supported: bool
    error: Optional[BaseException] = None


class NpuNormQuantOperatorTest(NormQuantOperatorTestBase):
    """One 950PR variant/precision and an instance-owned probe cache."""

    FP8_DTYPE_CODE = 292
    MXFP4_DTYPE_CODE = 296
    E8M0_DTYPE_CODE = 293

    RMS_STATIC_FP8 = "npu_rms_norm_quant_fp8_e4m3_static"
    ADD_STATIC_FP8 = "npu_add_rms_norm_quant_fp8_e4m3_static"
    ADD_DYNAMIC_FP8 = (
        "npu_add_rms_norm_dynamic_quant_fp8_e4m3_per_token"
    )
    RMS_MXFP8 = (
        "npu_rms_norm_dynamic_mx_quant_mxfp8_e4m3_e8m0_g32"
    )
    RMS_MXFP4 = (
        "npu_rms_norm_dynamic_mx_quant_mxfp4_e2m1_e8m0_g32"
    )
    ADD_MXFP8 = (
        "npu_add_rms_norm_dynamic_mx_quant_mxfp8_e4m3_e8m0_g32"
    )
    ADD_MXFP4 = (
        "npu_add_rms_norm_dynamic_mx_quant_mxfp4_e2m1_e8m0_g32"
    )

    _PROVIDERS = {
        (NormQuantVariant.RMS_NORM_STATIC_FP8, PrecisionType.FP8):
        RMS_STATIC_FP8,
        (NormQuantVariant.ADD_RMS_NORM_STATIC_FP8, PrecisionType.FP8):
        ADD_STATIC_FP8,
        (NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8, PrecisionType.FP8):
        ADD_DYNAMIC_FP8,
        (NormQuantVariant.RMS_NORM_DYNAMIC_MX, PrecisionType.MXFP8):
        RMS_MXFP8,
        (NormQuantVariant.RMS_NORM_DYNAMIC_MX, PrecisionType.MXFP4):
        RMS_MXFP4,
        (
            NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
            PrecisionType.MXFP8,
        ): ADD_MXFP8,
        (
            NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
            PrecisionType.MXFP4,
        ): ADD_MXFP4,
    }
    _NATIVE_OPS = {
        RMS_STATIC_FP8: "npu_rms_norm_quant",
        ADD_STATIC_FP8: "npu_add_rms_norm_quant",
        ADD_DYNAMIC_FP8: "npu_add_rms_norm_dynamic_quant",
        RMS_MXFP8: "npu_rms_norm_dynamic_mx_quant",
        RMS_MXFP4: "npu_rms_norm_dynamic_mx_quant",
        ADD_MXFP8: "npu_add_rms_norm_dynamic_mx_quant",
        ADD_MXFP4: "npu_add_rms_norm_dynamic_mx_quant",
    }
    _SCHEMA_CONTRACTS = {
        "npu_rms_norm_quant": (frozenset({"dst_dtype"}), 1),
        "npu_add_rms_norm_quant": (frozenset({"dst_type"}), 3),
        "npu_add_rms_norm_dynamic_quant": (
            frozenset({"y_dtype"}),
            5,
        ),
        "npu_rms_norm_dynamic_mx_quant": (
            frozenset({"scale_alg", "round_mode", "dst_type"}),
            3,
        ),
        "npu_add_rms_norm_dynamic_mx_quant": (
            frozenset({"scale_alg", "round_mode", "dst_type"}),
            4,
        ),
    }

    def __init__(
        self,
        variant: NormQuantVariant,
        precision: PrecisionType,
    ):
        if (variant, precision) not in self._PROVIDERS:
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
                "NPU NormQuant does not support "
                f"variant={variant_name}, precision={precision_name}"
            )
        super().__init__(variant, precision)
        self.supported_devices = [DeviceType.NPU]
        self._capability_cache: Dict[
            Tuple[str, str, str, int], CapabilityResult
        ] = {}
        self._capability_error: Optional[BaseException] = None

    @property
    def capability_error(self) -> Optional[BaseException]:
        return self._capability_error

    def _reject_formal(
        self,
        message: str,
        error: Optional[BaseException] = None,
    ) -> List[str]:
        self._capability_error = (
            error if error is not None else RuntimeError(message)
        )
        return []

    @staticmethod
    def _load_torch_npu():
        try:
            import torch_npu
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "Ascend NormQuant requires a working torch_npu runtime"
            ) from error
        return torch_npu

    @staticmethod
    def _fp8_dtype():
        return getattr(torch, "float8_e4m3fn", None)

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
    def _device_index(device: str) -> int:
        if ":" not in device:
            return 0
        return int(device.rsplit(":", maxsplit=1)[1])

    def _npu_device_name(self, device: str) -> Optional[str]:
        if not device.startswith("npu"):
            return None
        try:
            runtime = self._load_torch_npu()
            return str(
                runtime.npu.get_device_name(self._device_index(device))
            )
        except (
            AttributeError,
            IndexError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _schema_supported(cls, native_op: str) -> bool:
        namespace = getattr(torch.ops, "npu", None)
        packet = getattr(namespace, native_op, None)
        schemas = getattr(packet, "_schemas", None)
        if not isinstance(schemas, dict) or not schemas:
            return False
        required_arguments, return_arity = cls._SCHEMA_CONTRACTS[native_op]
        for schema in schemas.values():
            arguments = getattr(schema, "arguments", ())
            argument_names = {
                getattr(argument, "name", None) for argument in arguments
            }
            returns = getattr(schema, "returns", ())
            if (
                required_arguments.issubset(argument_names)
                and len(returns) == return_arity
            ):
                return True
        return False

    def _dtype_code(self) -> int:
        return (
            self.MXFP4_DTYPE_CODE
            if self.precision is PrecisionType.MXFP4
            else self.FP8_DTYPE_CODE
        )

    def _runtime_dtype_supported(self, runtime: Any) -> bool:
        if (
            self.precision is not PrecisionType.MXFP4
            and self._fp8_dtype() is None
        ):
            return False
        if self.precision in (PrecisionType.MXFP8, PrecisionType.MXFP4):
            if getattr(runtime, "float8_e8m0fnu", None) is None:
                return False
        if self.precision is PrecisionType.MXFP4:
            if getattr(runtime, "float4_e2m1fn_x2", None) is None:
                return False
        return True

    def _correctness_dependencies(self) -> Tuple[str, ...]:
        dependencies: List[str] = []
        if self.is_dynamic_fp8_variant:
            dependencies.append("npu_dynamic_quant")
        if self.is_mx_variant:
            dependencies.append("npu_dynamic_mx_quant")
        if self.is_add_variant and not self.is_static_variant:
            dependencies.append("npu_add_rms_norm")
        return tuple(dependencies)

    def _validate_probe_outputs(self, result: Any) -> None:
        fp8_dtype = self._fp8_dtype()
        if self.variant is NormQuantVariant.RMS_NORM_STATIC_FP8:
            specs = (((1, 64), fp8_dtype),)
        elif self.variant is NormQuantVariant.ADD_RMS_NORM_STATIC_FP8:
            specs = (
                ((1, 64), fp8_dtype),
                ((1, 64), fp8_dtype),
                ((1, 64), torch.bfloat16),
            )
        elif self.variant is NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8:
            specs = (
                ((1, 64), fp8_dtype),
                ((0,), fp8_dtype),
                ((1, 64), torch.bfloat16),
                ((1,), torch.float32),
                ((0,), torch.float32),
            )
        else:
            primary_spec = (
                ((1, 64), fp8_dtype)
                if self.precision is PrecisionType.MXFP8
                else ((1, 32), torch.uint8)
            )
            if self.is_add_variant:
                specs = (
                    primary_spec,
                    ((1, 64), torch.bfloat16),
                    ((1, 1, 2), torch.uint8),
                    ((0,), torch.float32),
                )
            else:
                specs = (
                    primary_spec,
                    ((1, 1, 2), torch.uint8),
                    ((0,), torch.float32),
                )
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        if len(outputs) != len(specs):
            raise RuntimeError(
                f"native probe returned {len(outputs)} outputs; "
                f"expected {len(specs)}"
            )
        for index, (output, (shape, dtype)) in enumerate(
            zip(outputs, specs)
        ):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    f"native probe output {index} is not a Tensor"
                )
            if tuple(output.shape) != shape:
                raise RuntimeError(
                    f"native probe output {index} shape "
                    f"{tuple(output.shape)} != {shape}"
                )
            if output.dtype is not dtype:
                raise RuntimeError(
                    f"native probe output {index} dtype "
                    f"{output.dtype} != {dtype}"
                )
            if not output.is_contiguous():
                raise RuntimeError(
                    f"native probe output {index} is not contiguous"
                )

    def _probe_payload(self, device: str, op: Callable[..., Any]):
        x = self._copy_to_device(
            torch.ones(1, 64, dtype=torch.bfloat16),
            device=device,
            dtype=torch.bfloat16,
        )
        gamma = self._copy_to_device(
            torch.ones(64, dtype=torch.bfloat16),
            device=device,
            dtype=torch.bfloat16,
        )
        prepared = {
            "implementation": self._PROVIDERS[
                (self.variant, self.precision)
            ],
            "op": op,
            "x": x,
            "weight": gamma,
            "eps": 1e-6,
            "scale": torch.ones_like(gamma),
            "offset": torch.zeros_like(gamma),
        }
        if self.variant is NormQuantVariant.RMS_NORM_STATIC_FP8:
            prepared["beta"] = self._copy_to_device(
                torch.zeros(64, dtype=torch.bfloat16),
                device=device,
                dtype=torch.bfloat16,
            )
        if self.is_add_variant:
            prepared["residual"] = torch.zeros_like(x)
        return prepared

    def _probe_capability(
        self,
        device: str,
        runtime: Any,
        native_op: str,
        dtype_code: int,
    ) -> CapabilityResult:
        version = str(getattr(runtime, "__version__", "unknown"))
        key = (device, version, native_op, dtype_code)
        cached = self._capability_cache.get(key)
        if cached is not None:
            self._capability_error = cached.error
            return cached
        try:
            op = getattr(runtime, native_op)
            result = self._invoke_native(
                self._probe_payload(device, op)
            )
            self._validate_probe_outputs(result)
            synchronize = getattr(runtime.npu, "synchronize", None)
            if not callable(synchronize):
                raise RuntimeError(
                    "torch_npu.npu.synchronize is unavailable"
                )
            synchronize(self._device_index(device))
            probe_result = CapabilityResult(True)
        except Exception as error:
            probe_result = CapabilityResult(False, error)
        self._capability_cache[key] = probe_result
        self._capability_error = probe_result.error
        return probe_result

    def get_formal_implementations(self, device: str) -> List[str]:
        if not device.startswith("npu"):
            return self._reject_formal(
                f"{self.operator_name} requires an NPU device; got {device}"
            )
        try:
            runtime = self._load_torch_npu()
        except RuntimeError as error:
            return self._reject_formal(
                "torch_npu runtime is unavailable",
                error,
            )
        device_name = self._npu_device_name(device)
        if not (device_name or "").startswith("Ascend950PR"):
            return self._reject_formal(
                f"{self.operator_name} requires a device name beginning "
                f"'Ascend950PR'; got {device_name!r}"
            )
        implementation = self._PROVIDERS[(self.variant, self.precision)]
        native_op = self._NATIVE_OPS[implementation]
        if not callable(getattr(runtime, native_op, None)):
            return self._reject_formal(
                f"torch_npu runtime symbol {native_op} is unavailable"
            )
        if not self._schema_supported(native_op):
            return self._reject_formal(
                f"torch.ops.npu.{native_op} schema does not satisfy "
                "the captured 950PR ABI"
            )
        if not self._runtime_dtype_supported(runtime):
            return self._reject_formal(
                f"{implementation} required native dtype is unavailable"
            )
        for dependency in self._correctness_dependencies():
            if not callable(getattr(runtime, dependency, None)):
                return self._reject_formal(
                    f"{implementation} untimed correctness dependency "
                    f"{dependency} is unavailable"
                )
        capability = self._probe_capability(
            device,
            runtime,
            native_op,
            self._dtype_code(),
        )
        return [implementation] if capability.supported else []

    def get_available_implementations(self, device: str) -> List[str]:
        return self.get_formal_implementations(device)

    def _resolve_implementation(
        self,
        device: str,
        implementation: str,
    ) -> str:
        formal = self.get_formal_implementations(device)
        if implementation == "default" and formal:
            return formal[0]
        if implementation in formal:
            return implementation
        raise ValueError(
            f"{self.operator_name} requires an Ascend950PR device and "
            f"native E4M3 capability; device={device}, formal={formal}"
        )

    def _invoke_native(self, prepared_data: Dict[str, Any]) -> Any:
        implementation = prepared_data["implementation"]
        op = prepared_data["op"]
        x = prepared_data["x"]
        gamma = prepared_data["weight"]
        eps = prepared_data["eps"]
        if implementation == self.RMS_STATIC_FP8:
            return op(
                x,
                gamma,
                prepared_data["beta"],
                prepared_data["scale"],
                prepared_data["offset"],
                eps,
                dst_dtype=self._fp8_dtype(),
            )
        if implementation == self.ADD_STATIC_FP8:
            return op(
                x,
                prepared_data["residual"],
                gamma,
                prepared_data["scale"],
                prepared_data["offset"],
                None,
                None,
                None,
                axis=-1,
                epsilon=eps,
                div_mode=True,
                dst_type=self.FP8_DTYPE_CODE,
            )
        if implementation == self.ADD_DYNAMIC_FP8:
            return op(
                x,
                prepared_data["residual"],
                gamma,
                smooth_scale1=None,
                smooth_scale2=None,
                beta=None,
                epsilon=eps,
                output_mask=[True, False],
                y_dtype=self._fp8_dtype(),
            )
        kwargs = {
            "beta": None,
            "epsilon": eps,
            "scale_alg": 0,
            "round_mode": "rint",
            "dst_type": self._dtype_code(),
        }
        if implementation in (self.RMS_MXFP8, self.RMS_MXFP4):
            return op(x, gamma, **kwargs)
        return op(x, prepared_data["residual"], gamma, **kwargs)

    def _prepare_data_for_core_operator(
        self,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> Dict[str, Any]:
        """Copy fresh inputs and cache one allocating native callable."""
        resolved = self._resolve_implementation(device, implementation)
        if precision is not self.precision:
            raise ValueError(
                f"{self.operator_name} requires "
                f"PrecisionType.{self.precision.name}"
            )
        x_source = data["x"]
        weight_source = data["weight"]
        residual_source = data.get("residual")
        if x_source.ndim != 2 or weight_source.ndim != 1:
            raise ValueError(
                "NormQuant requires x=[tokens, hidden], weight=[hidden]"
            )
        if x_source.shape[1] != weight_source.shape[0]:
            raise ValueError(
                "NormQuant x and weight hidden dimensions differ"
            )
        if self.is_add_variant and (
            residual_source is None
            or residual_source.shape != x_source.shape
        ):
            raise ValueError(
                "Add NormQuant requires a shape-matched residual"
            )
        if self.is_mx_variant and x_source.shape[1] % 64 != 0:
            raise ValueError(
                "MX NormQuant hidden dimension must be a multiple of 64"
            )

        runtime = self._load_torch_npu()
        native_op = self._NATIVE_OPS[resolved]
        op = getattr(runtime, native_op, None)
        if not callable(op):
            raise RuntimeError(
                f"torch_npu.{native_op} became unavailable after probe"
            )
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
            "op": op,
            "runtime": runtime,
            "device": device,
            "x": x,
            "weight": weight,
            "eps": float(data["eps"]),
        }
        if resolved == self.RMS_STATIC_FP8:
            prepared["beta"] = self._copy_to_device(
                torch.zeros_like(weight_source),
                device=device,
                dtype=torch.bfloat16,
            )
        if self.is_static_variant:
            prepared["scale"] = self._copy_to_device(
                torch.ones_like(weight_source),
                device=device,
                dtype=torch.bfloat16,
            )
            prepared["offset"] = self._copy_to_device(
                torch.zeros_like(weight_source),
                device=device,
                dtype=torch.bfloat16,
            )
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
        return prepared

    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> Any:
        """Issue one native allocating call and retain every returned output."""
        resolved = prepared_data["implementation"]
        if implementation not in ("default", resolved):
            raise ValueError(
                "NormQuant prepared data provenance mismatch: "
                f"prepared for {resolved!r}, requested {implementation!r}"
            )
        return self._invoke_native(prepared_data)

    def _retained_prepared_input_bytes(
        self,
        data: Dict[str, Any],
    ) -> int:
        """Count every tensor retained in one native prepared payload."""
        x = data["x"]
        weight = data["weight"]
        x_bytes = x.numel() * x.element_size()
        weight_bytes = weight.numel() * weight.element_size()
        total = x_bytes + weight_bytes
        if self.is_add_variant:
            # The native residual and immutable restore seed are both retained.
            total += 2 * x_bytes
        if self.is_static_variant:
            # Ascend's static ABI requires vector scale and offset tensors.
            total += 2 * weight_bytes
        if self.variant is NormQuantVariant.RMS_NORM_STATIC_FP8:
            # This native ABI also requires a materialized beta vector.
            total += weight_bytes
        return total

    def _native_output_contract_bytes(
        self,
        data: Dict[str, Any],
    ) -> int:
        """Count the complete allocating NPU tuple, including x_out."""
        tokens, hidden = data["x"].shape
        matrix_elements = tokens * hidden
        if self.is_static_variant:
            total = matrix_elements
            if self.is_add_variant:
                total += matrix_elements
        elif self.is_dynamic_fp8_variant:
            total = matrix_elements + tokens * 4
        elif self.is_mx_variant:
            quantized = (
                matrix_elements
                if self.precision is PrecisionType.MXFP8
                else matrix_elements // 2
            )
            total = quantized + tokens * ((hidden + 63) // 64) * 2
        else:
            raise ValueError(
                f"unsupported NPU NormQuant variant {self.variant}"
            )
        if self.is_add_variant:
            total += matrix_elements * data["x"].element_size()
        return total

    def _native_input_bytes(self, data: Dict[str, Any]) -> int:
        """Count tensor bytes passed to the native NPU operator ABI."""
        x = data["x"]
        weight = data["weight"]
        x_bytes = x.numel() * x.element_size()
        weight_bytes = weight.numel() * weight.element_size()
        total = x_bytes + weight_bytes
        if self.is_add_variant:
            total += x_bytes
        if self.is_static_variant:
            total += 2 * weight_bytes
        if self.variant is NormQuantVariant.RMS_NORM_STATIC_FP8:
            total += weight_bytes
        return total

    def physical_bytes(
        self,
        data: Dict[str, Any],
        outputs: Optional[Any] = None,
    ) -> int:
        """Return the retained native NPU input plus full tuple output bytes."""
        output_bytes = (
            self._native_output_contract_bytes(data)
            if outputs is None
            else self.observable_output_bytes(outputs)
        )
        return self._native_input_bytes(data) + output_bytes

    @staticmethod
    def _result_tuple(result: Any, expected_arity: int) -> Tuple[Any, ...]:
        outputs = result if isinstance(result, tuple) else (result,)
        if len(outputs) != expected_arity:
            raise AssertionError(
                f"native result has {len(outputs)} outputs, expected "
                f"{expected_arity}"
            )
        if not all(isinstance(output, torch.Tensor) for output in outputs):
            raise AssertionError("every native output must be a Tensor")
        return outputs

    @staticmethod
    def _assert_tensor_contract(
        tensor: torch.Tensor,
        *,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        label: str,
    ) -> None:
        if tuple(tensor.shape) != shape:
            raise AssertionError(
                f"{label} shape {tuple(tensor.shape)} != {shape}"
            )
        if tensor.dtype is not dtype:
            raise AssertionError(
                f"{label} dtype {tensor.dtype} != {dtype}"
            )
        if not tensor.is_contiguous():
            raise AssertionError(f"{label} must be contiguous")

    @classmethod
    def _assert_empty_output(
        cls,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        label: str,
    ) -> None:
        cls._assert_tensor_contract(
            tensor,
            shape=(0,),
            dtype=dtype,
            label=label,
        )
        if tensor.numel() != 0:
            raise AssertionError(f"{label} disabled output must be empty")

    @staticmethod
    def _assert_same_storage_values(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        label: str,
    ) -> None:
        actual_bytes = actual.contiguous().view(torch.uint8)
        expected_bytes = expected.contiguous().view(torch.uint8)
        if not torch.equal(actual_bytes, expected_bytes):
            raise AssertionError(f"{label} storage values mismatch")

    @staticmethod
    def _decode_e8m0_scales(scale: torch.Tensor) -> torch.Tensor:
        exponent = scale.to(torch.int16).float() - 127.0
        return torch.pow(
            torch.tensor(2.0, device=scale.device),
            exponent,
        )

    @staticmethod
    def _decode_mxfp4(packed: torch.Tensor) -> torch.Tensor:
        value_table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32,
            device=packed.device,
        )
        low = packed & 0x0F
        high = packed >> 4

        def decode(nibbles: torch.Tensor) -> torch.Tensor:
            magnitude = value_table[(nibbles & 0x07).long()]
            return torch.where(
                (nibbles & 0x08) != 0,
                -magnitude,
                magnitude,
            )

        unpacked = torch.empty(
            (*packed.shape[:-1], packed.shape[-1] * 2),
            dtype=torch.float32,
            device=packed.device,
        )
        unpacked[..., 0::2] = decode(low)
        unpacked[..., 1::2] = decode(high)
        return unpacked

    def _reference_on_device(
        self,
        data: Dict[str, Any],
        prepared: Dict[str, Any],
    ) -> torch.Tensor:
        return self._copy_to_device(
            self.run_cpu_reference(data).to(torch.bfloat16),
            device=prepared["device"],
            dtype=torch.bfloat16,
        )

    def _quantization_reference_on_device(
        self,
        data: Dict[str, Any],
        prepared: Dict[str, Any],
    ) -> torch.Tensor:
        if not self.is_add_variant:
            return self._reference_on_device(data, prepared)
        add_rms_norm = getattr(
            prepared["runtime"],
            "npu_add_rms_norm",
            None,
        )
        if not callable(add_rms_norm):
            raise AssertionError(
                "standalone npu_add_rms_norm is unavailable"
            )
        outputs = self._result_tuple(
            add_rms_norm(
                prepared["x"],
                prepared["residual_seed"],
                prepared["weight"],
                prepared["eps"],
            ),
            3,
        )
        reference = outputs[0]
        self._assert_tensor_contract(
            reference,
            shape=tuple(prepared["x"].shape),
            dtype=torch.bfloat16,
            label="native unfused AddRMSNorm reference",
        )
        return reference

    @staticmethod
    def _validate_x_out(
        prepared: Dict[str, Any],
        x_out: torch.Tensor,
    ) -> None:
        expected = (
            prepared["x"].to(torch.bfloat16)
            + prepared["residual_seed"].to(torch.bfloat16)
        )
        try:
            torch.testing.assert_close(
                x_out,
                expected,
                rtol=0,
                atol=0,
            )
        except AssertionError as error:
            raise AssertionError("NPU x_out values mismatch") from error

    def _validate_plain_correctness(
        self,
        data: Dict[str, Any],
        prepared: Dict[str, Any],
        result: Any,
    ) -> None:
        tokens, hidden = data["x"].shape
        reference = self.run_cpu_reference(data).to(prepared["x"].device)
        if self.variant is NormQuantVariant.RMS_NORM_STATIC_FP8:
            outputs = self._result_tuple(result, 1)
        elif self.variant is NormQuantVariant.ADD_RMS_NORM_STATIC_FP8:
            outputs = self._result_tuple(result, 3)
        else:
            outputs = self._result_tuple(result, 5)
        primary = outputs[0]
        self._assert_tensor_contract(
            primary,
            shape=(tokens, hidden),
            dtype=self._fp8_dtype(),
            label="plain FP8 primary",
        )

        if self.is_static_variant:
            scale = prepared["scale"].float()
            offset = prepared["offset"].float()
            dequantized = (
                primary.float() - offset.reshape(1, hidden)
            ) / scale.reshape(1, hidden)
            if self.is_add_variant:
                self._assert_tensor_contract(
                    outputs[1],
                    shape=(tokens, hidden),
                    dtype=self._fp8_dtype(),
                    label="static secondary",
                )
                self._assert_tensor_contract(
                    outputs[2],
                    shape=(tokens, hidden),
                    dtype=torch.bfloat16,
                    label="static x_out",
                )
                self._validate_x_out(prepared, outputs[2])
        else:
            self._assert_empty_output(
                outputs[1],
                dtype=self._fp8_dtype(),
                label="dynamic secondary disabled",
            )
            self._assert_tensor_contract(
                outputs[2],
                shape=(tokens, hidden),
                dtype=torch.bfloat16,
                label="dynamic x_out",
            )
            self._validate_x_out(prepared, outputs[2])
            scale = outputs[3]
            self._assert_tensor_contract(
                scale,
                shape=(tokens,),
                dtype=torch.float32,
                label="dynamic FP8 scale",
            )
            self._assert_empty_output(
                outputs[4],
                dtype=torch.float32,
                label="dynamic scale2 disabled",
            )
            dynamic_quant = getattr(
                prepared["runtime"],
                "npu_dynamic_quant",
                None,
            )
            if not callable(dynamic_quant):
                raise AssertionError(
                    "standalone npu_dynamic_quant is unavailable"
                )
            expected_primary, expected_scale = dynamic_quant(
                self._quantization_reference_on_device(data, prepared),
                dst_type=self._fp8_dtype(),
            )
            self._assert_same_storage_values(
                primary,
                expected_primary,
                label="dynamic FP8 primary",
            )
            try:
                torch.testing.assert_close(
                    scale,
                    expected_scale,
                    rtol=1e-6,
                    atol=0,
                )
            except AssertionError as error:
                raise AssertionError(
                    "dynamic FP8 scale mismatch"
                ) from error
            dequantized = primary.float() * scale.reshape(tokens, 1)

        torch.testing.assert_close(
            dequantized,
            reference,
            rtol=0.15,
            atol=0.15,
        )

    def _validate_mx_correctness(
        self,
        data: Dict[str, Any],
        prepared: Dict[str, Any],
        result: Any,
    ) -> None:
        tokens, hidden = data["x"].shape
        expected_arity = 4 if self.is_add_variant else 3
        outputs = self._result_tuple(result, expected_arity)
        primary = outputs[0]
        if self.is_add_variant:
            x_out = outputs[1]
            scale = outputs[2]
            rstd = outputs[3]
        else:
            scale = outputs[1]
            rstd = outputs[2]
        primary_shape = (
            (tokens, hidden)
            if self.precision is PrecisionType.MXFP8
            else (tokens, hidden // 2)
        )
        primary_dtype = (
            self._fp8_dtype()
            if self.precision is PrecisionType.MXFP8
            else torch.uint8
        )
        self._assert_tensor_contract(
            primary,
            shape=primary_shape,
            dtype=primary_dtype,
            label=f"{self.precision.name} primary",
        )
        self._assert_tensor_contract(
            scale,
            shape=(tokens, hidden // 64, 2),
            dtype=torch.uint8,
            label="MX E8M0 scale",
        )
        self._assert_empty_output(
            rstd,
            dtype=torch.float32,
            label="MX rstd disabled",
        )
        if self.is_add_variant:
            self._assert_tensor_contract(
                x_out,
                shape=(tokens, hidden),
                dtype=torch.bfloat16,
                label="MX x_out",
            )
            self._validate_x_out(prepared, x_out)

        dynamic_mx_quant = getattr(
            prepared["runtime"],
            "npu_dynamic_mx_quant",
            None,
        )
        if not callable(dynamic_mx_quant):
            raise AssertionError(
                "standalone npu_dynamic_mx_quant is unavailable"
            )
        expected_primary, expected_scale = dynamic_mx_quant(
            self._quantization_reference_on_device(data, prepared),
            scale_alg=0,
            round_mode="rint",
            dst_type=self._dtype_code(),
        )
        if not torch.equal(scale, expected_scale):
            raise AssertionError("MX E8M0 scale metadata mismatch")
        if self.precision is PrecisionType.MXFP8:
            self._assert_same_storage_values(
                primary,
                expected_primary,
                label="MXFP8 primary",
            )
            decoded_codes = primary.float()
            tolerance = {"rtol": 0.15, "atol": 0.15}
        else:
            actual_low = primary & 0x0F
            expected_low = expected_primary & 0x0F
            actual_high = primary >> 4
            expected_high = expected_primary >> 4
            if not torch.equal(actual_low, expected_low):
                raise AssertionError(
                    "MXFP4 packed E2M1 low values mismatch"
                )
            if not torch.equal(actual_high, expected_high):
                raise AssertionError(
                    "MXFP4 packed E2M1 high values mismatch"
                )
            decoded_codes = self._decode_mxfp4(primary)
            tolerance = {"rtol": 0.3, "atol": 0.6}

        decoded_scales = self._decode_e8m0_scales(scale).reshape(
            tokens, hidden // 32
        )
        dequantized = (
            decoded_codes.reshape(tokens, hidden // 32, 32)
            * decoded_scales.unsqueeze(-1)
        ).reshape(tokens, hidden)
        reference = self.run_cpu_reference(data).to(dequantized.device)
        torch.testing.assert_close(
            dequantized,
            reference,
            **tolerance,
        )

    def validate_prepared_correctness(
        self,
        data: Dict[str, Any],
        prepared: Dict[str, Any],
        result: Any,
    ) -> None:
        """Validate allocating NPU primary and auxiliary outputs untimed."""
        if self.is_mx_variant:
            self._validate_mx_correctness(data, prepared, result)
        else:
            self._validate_plain_correctness(data, prepared, result)

    def _declares_preallocated_output_contract(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> bool:
        del prepared_data, implementation
        return False
