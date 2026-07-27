"""Framework operator implementation for Qwen3.5 recurrent GDN."""

from __future__ import annotations

from typing import Any

import torch

from operator_test_framework import BaseOperatorTest, DeviceType, PrecisionType

from .impl import (
    CudaFlaDirectOutRecurrentImpl,
    CudaFlaRecurrentImpl,
    NpuBuiltinRecurrentImpl,
)


NUM_KEY_HEADS = 4
NUM_VALUE_HEADS = 16
HEAD_DIM = 128
BASE_SEED = 20260721
TOKEN_COUNTS = {"decode": 1, "mtp3": 4}


class RecurrentGatedDeltaRuleOperatorTest(BaseOperatorTest):
    """Same logical recurrent core on CANN builtin and vLLM CUDA FLA."""

    def __init__(self):
        super().__init__("RecurrentGatedDeltaRule")
        self.supported_precisions = [PrecisionType.BF16]
        self.supported_devices = [DeviceType.NPU, DeviceType.GPU]
        self.implementations = {
            NpuBuiltinRecurrentImpl.name: NpuBuiltinRecurrentImpl(),
            CudaFlaRecurrentImpl.name: CudaFlaRecurrentImpl(),
            CudaFlaDirectOutRecurrentImpl.name: CudaFlaDirectOutRecurrentImpl(),
        }

    @staticmethod
    def case_seed(mode: str, batch_size: int) -> int:
        return BASE_SEED + (0 if mode == "decode" else 100_000) + batch_size

    def generate_test_data(
        self, mode: str = "decode", batch_size: int = 1, **_: Any
    ) -> dict[str, Any]:
        if mode not in TOKEN_COUNTS:
            raise ValueError(f"unsupported mode: {mode}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        tokens_per_sequence = TOKEN_COUNTS[mode]
        total_tokens = batch_size * tokens_per_sequence
        generator = torch.Generator(device="cpu")
        seed = self.case_seed(mode, batch_size)
        generator.manual_seed(seed)

        def rand(*shape: int) -> torch.Tensor:
            return torch.rand(shape, dtype=torch.float32, generator=generator)

        return {
            "query": rand(total_tokens, NUM_KEY_HEADS, HEAD_DIM).to(torch.bfloat16),
            "key": rand(total_tokens, NUM_KEY_HEADS, HEAD_DIM).to(torch.bfloat16),
            "value": rand(total_tokens, NUM_VALUE_HEADS, HEAD_DIM).to(torch.bfloat16),
            "beta": rand(total_tokens, NUM_VALUE_HEADS).to(torch.bfloat16),
            "g": rand(total_tokens, NUM_VALUE_HEADS),
            "metadata": {
                "mode": mode,
                "batch_size": batch_size,
                "tokens_per_sequence": tokens_per_sequence,
                "total_tokens": total_tokens,
                "num_key_heads": NUM_KEY_HEADS,
                "num_value_heads": NUM_VALUE_HEADS,
                "head_dim": HEAD_DIM,
                "scale": HEAD_DIM**-0.5,
                "seed": seed,
                "input_policy": "exact Yuque harness: uniform Q/K/V/beta/g; no in-op QK norm",
            },
        }

    def cpu_reference_with_state(
        self, data: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = data["query"].float()
        k = data["key"].float()
        value = data["value"].float()
        beta = data["beta"].float()
        g = data["g"].float()
        metadata = data["metadata"]
        batch_size = int(metadata["batch_size"])
        tokens_per_sequence = int(metadata["tokens_per_sequence"])
        total_tokens = int(metadata["total_tokens"])
        output = torch.empty(total_tokens, NUM_VALUE_HEADS, HEAD_DIM)
        states = torch.empty(
            total_tokens, NUM_VALUE_HEADS, HEAD_DIM, HEAD_DIM
        )
        for sequence in range(batch_size):
            state = torch.zeros(NUM_VALUE_HEADS, HEAD_DIM, HEAD_DIM)
            start = sequence * tokens_per_sequence
            for offset in range(tokens_per_sequence):
                token = start + offset
                for value_head in range(NUM_VALUE_HEADS):
                    key_head = value_head // (NUM_VALUE_HEADS // NUM_KEY_HEADS)
                    state[value_head].mul_(torch.exp(g[token, value_head]))
                    residual = value[token, value_head] - torch.mv(
                        state[value_head], k[token, key_head]
                    )
                    residual.mul_(beta[token, value_head])
                    state[value_head].add_(
                        residual[:, None] * k[token, key_head][None, :]
                    )
                    output[token, value_head] = torch.mv(
                        state[value_head], q[token, key_head]
                    ) * metadata["scale"]
                states[token].copy_(state)
        return output, states

    def run_cpu_reference(self, data: dict[str, Any]) -> torch.Tensor:
        return self.cpu_reference_with_state(data)[0]

    def _resolve_implementation(self, device: str, implementation: str) -> str:
        formal = self.get_formal_implementations(device)
        if implementation == "default":
            if not formal:
                raise ValueError(f"unsupported recurrent device: {device}")
            return formal[0]
        if implementation not in formal:
            raise ValueError(
                f"implementation {implementation!r} is not formal for "
                f"{device}; formal={formal}"
            )
        return implementation

    def get_available_implementations(self, device: str) -> list[str]:
        return self.get_formal_implementations(device)

    def get_formal_implementations(self, device: str) -> list[str]:
        if device.startswith("cuda"):
            return [CudaFlaDirectOutRecurrentImpl.name]
        if device.startswith("npu"):
            return [NpuBuiltinRecurrentImpl.name]
        return []

    def _prepare_data_for_core_operator(
        self,
        data: dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> dict[str, Any]:
        if precision != PrecisionType.BF16:
            raise ValueError("RecurrentGatedDeltaRule benchmark is BF16 only")
        resolved = self._resolve_implementation(device, implementation)
        prepared = self.implementations[resolved].prepare_data(data, device)
        prepared["_implementation"] = resolved
        return prepared

    def _execute_core_operator(
        self, prepared_data: dict[str, Any], implementation: str = "default"
    ) -> torch.Tensor:
        resolved = prepared_data["_implementation"]
        if implementation != "default" and implementation != resolved:
            raise ValueError(
                f"prepared for {resolved}, execute requested {implementation}"
            )
        return self.implementations[resolved].execute_core_operator(prepared_data)

    def run_core_operator(
        self,
        data: dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> torch.Tensor:
        prepared = self._prepare_data_for_core_operator(
            data, device, precision, implementation
        )
        return self._execute_core_operator(prepared, implementation)

    def run_device_implementation(
        self,
        data: dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
    ) -> torch.Tensor:
        output = self.run_core_operator(data, device, precision, implementation)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        else:
            torch.npu.synchronize()
        return output.detach().reshape(-1, NUM_VALUE_HEADS, HEAD_DIM).float().cpu()

    def calculate_throughput(self, data: dict[str, Any], time_ms: float) -> float:
        if time_ms <= 0:
            return 0.0
        return int(data["metadata"]["total_tokens"]) * 1000.0 / time_ms

    def correctness(
        self, mode: str, device: str, implementation: str
    ) -> dict[str, float | bool]:
        data = self.generate_test_data(mode=mode, batch_size=1)
        prepared = self._prepare_data_for_core_operator(
            data, device, PrecisionType.BF16, implementation
        )
        with torch.inference_mode():
            output = self._execute_core_operator(prepared, implementation)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        else:
            torch.npu.synchronize()
        actual = output.detach().reshape(-1, NUM_VALUE_HEADS, HEAD_DIM).float().cpu()
        state_actual = prepared["state"].detach()
        if device.startswith("cuda"):
            state_actual = state_actual[1:]
        state_actual = state_actual.float().cpu()
        reference, state_reference = self.cpu_reference_with_state(data)

        def metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float, float]:
            difference = (lhs - rhs).abs()
            cosine = torch.nn.functional.cosine_similarity(
                lhs.reshape(1, -1), rhs.reshape(1, -1), dim=1
            ).item()
            return cosine, difference.max().item(), difference.mean().item()

        cosine, max_abs, mean_abs = metrics(actual, reference)
        state_cosine, state_max_abs, state_mean_abs = metrics(
            state_actual, state_reference
        )
        passed = bool(
            torch.isfinite(actual).all()
            and torch.isfinite(state_actual).all()
            and cosine >= 0.995
            and state_cosine >= 0.995
            and torch.allclose(actual, reference, rtol=0.10, atol=0.08)
            and torch.allclose(
                state_actual, state_reference, rtol=0.10, atol=0.08
            )
        )
        return {
            "cosine": cosine,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "state_cosine": state_cosine,
            "state_max_abs": state_max_abs,
            "state_mean_abs": state_mean_abs,
            "passed": passed,
        }
