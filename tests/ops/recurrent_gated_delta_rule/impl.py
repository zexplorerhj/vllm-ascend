"""Device providers for recurrent gated-delta-rule."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None


class NpuBuiltinRecurrentImpl:
    name = "npu_cann_builtin"

    def __init__(self) -> None:
        self._operator: Callable[..., torch.Tensor] | None = None

    def operator(self) -> Callable[..., torch.Tensor]:
        if self._operator is not None:
            return self._operator
        if torch_npu is None:
            raise RuntimeError("torch_npu is unavailable")
        if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
            raise RuntimeError(
                "ASCEND_CUSTOM_OPP_PATH must be unset for the CANN-builtin provider"
            )
        if not hasattr(torch.ops.npu, "npu_recurrent_gated_delta_rule"):
            raise RuntimeError("npu_recurrent_gated_delta_rule is unavailable")
        operator = torch.ops.npu.npu_recurrent_gated_delta_rule
        schema = str(operator.default._schema)
        if not schema.startswith("npu::npu_recurrent_gated_delta_rule("):
            raise RuntimeError(f"unexpected operator schema: {schema}")
        self._operator = operator
        return self._operator

    def prepare_data(self, data: dict[str, Any], device: str) -> dict[str, Any]:
        # Resolve and validate the provider before the timed region.
        self.operator()
        total_tokens = int(data["metadata"]["total_tokens"])
        batch_size = int(data["metadata"]["batch_size"])
        tokens_per_sequence = int(data["metadata"]["tokens_per_sequence"])
        prepared = {
            "query": data["query"].to(device=device, copy=True).contiguous(),
            "key": data["key"].to(device=device, copy=True).contiguous(),
            "value": data["value"].to(device=device, copy=True).contiguous(),
            "beta": data["beta"].to(device=device, copy=True).contiguous(),
            "g": data["g"].to(device=device, copy=True).contiguous(),
            "state": torch.zeros(
                total_tokens,
                data["metadata"]["num_value_heads"],
                data["metadata"]["head_dim"],
                data["metadata"]["head_dim"],
                dtype=torch.bfloat16,
                device=device,
            ),
            "actual_seq_lengths": torch.full(
                (batch_size,),
                tokens_per_sequence,
                dtype=torch.int32,
                device=device,
            ),
            "ssm_state_indices": torch.arange(
                total_tokens, dtype=torch.int32, device=device
            ),
            "num_accepted_tokens": None,
            "scale": data["metadata"]["scale"],
        }
        if tokens_per_sequence > 1:
            prepared["num_accepted_tokens"] = torch.full(
                (batch_size,),
                tokens_per_sequence,
                dtype=torch.int32,
                device=device,
            )
        return prepared

    def execute_core_operator(self, prepared: dict[str, Any]) -> torch.Tensor:
        kwargs = {
            "query": prepared["query"],
            "key": prepared["key"],
            "value": prepared["value"],
            "state": prepared["state"],
            "beta": prepared["beta"],
            "scale": prepared["scale"],
            "actual_seq_lengths": prepared["actual_seq_lengths"],
            "ssm_state_indices": prepared["ssm_state_indices"],
            "g": prepared["g"],
        }
        if prepared["num_accepted_tokens"] is not None:
            kwargs["num_accepted_tokens"] = prepared["num_accepted_tokens"]
        return self.operator()(**kwargs)


class CudaFlaRecurrentImpl:
    name = "cuda_vllm_fla"

    def __init__(self) -> None:
        self._operator: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None

    def operator(self) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        if self._operator is not None:
            return self._operator
        errors = []
        for module_name in (
            "vllm.third_party.flash_linear_attention.ops",
            "vllm.model_executor.layers.fla.ops",
        ):
            try:
                module = __import__(
                    module_name, fromlist=["fused_recurrent_gated_delta_rule"]
                )
                self._operator = getattr(module, "fused_recurrent_gated_delta_rule")
                return self._operator
            except Exception as exc:
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("vLLM FLA provider unavailable: " + " | ".join(errors))

    def prepare_data(self, data: dict[str, Any], device: str) -> dict[str, Any]:
        # Import/fallback is provider setup, not recurrent-kernel work.
        self.operator()
        total_tokens = int(data["metadata"]["total_tokens"])
        batch_size = int(data["metadata"]["batch_size"])
        tokens_per_sequence = int(data["metadata"]["tokens_per_sequence"])
        query = data["query"].to(device=device, copy=True).contiguous()
        key = data["key"].to(device=device, copy=True).contiguous()
        value = data["value"].to(device=device, copy=True).contiguous()
        beta = data["beta"].to(device=device, copy=True).contiguous()
        g = data["g"].to(device=device, copy=True).contiguous()
        prepared = {
            "query": query.unsqueeze(0),
            "key": key.unsqueeze(0),
            "value": value.unsqueeze(0),
            "beta": beta.unsqueeze(0),
            "g": g.unsqueeze(0),
            # FLA reserves slot zero as NULL_BLOCK_ID.
            "state": torch.zeros(
                total_tokens + 1,
                data["metadata"]["num_value_heads"],
                data["metadata"]["head_dim"],
                data["metadata"]["head_dim"],
                dtype=torch.bfloat16,
                device=device,
            ),
            "cu_seqlens": torch.arange(
                0,
                total_tokens + 1,
                tokens_per_sequence,
                dtype=torch.int32,
                device=device,
            ),
            "ssm_state_indices": torch.arange(
                1, total_tokens + 1, dtype=torch.int32, device=device
            ).reshape(batch_size, tokens_per_sequence),
            "num_accepted_tokens": None,
            "scale": data["metadata"]["scale"],
        }
        if tokens_per_sequence > 1:
            prepared["num_accepted_tokens"] = torch.full(
                (batch_size,),
                tokens_per_sequence,
                dtype=torch.int32,
                device=device,
            )
        return prepared

    def execute_core_operator(self, prepared: dict[str, Any]) -> torch.Tensor:
        output, _ = self.operator()(
            q=prepared["query"],
            k=prepared["key"],
            v=prepared["value"],
            g=prepared["g"],
            beta=prepared["beta"],
            scale=prepared["scale"],
            initial_state=prepared["state"],
            inplace_final_state=True,
            cu_seqlens=prepared["cu_seqlens"],
            ssm_state_indices=prepared["ssm_state_indices"],
            num_accepted_tokens=prepared["num_accepted_tokens"],
            use_qk_l2norm_in_kernel=False,
        )
        return output


class CudaFlaDirectOutRecurrentImpl(CudaFlaRecurrentImpl):
    """Inference-only launch of the same vLLM FLA kernel into a prepared output."""

    name = "cuda_vllm_fla_direct_out"

    def __init__(self) -> None:
        super().__init__()
        self._kernel_module: Any | None = None

    def kernel_module(self) -> Any:
        if self._kernel_module is not None:
            return self._kernel_module
        errors = []
        for module_name in (
            "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
            "vllm.model_executor.layers.fla.ops.fused_recurrent",
        ):
            try:
                self._kernel_module = __import__(module_name, fromlist=["*"])
                return self._kernel_module
            except Exception as exc:
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "vLLM FLA direct kernel unavailable: " + " | ".join(errors)
        )

    def prepare_data(self, data: dict[str, Any], device: str) -> dict[str, Any]:
        # Resolve the internal kernel outside Event timing. Do not call the
        # parent public-API resolver because this provider intentionally bypasses
        # its autograd/output-allocation wrapper.
        self.kernel_module()
        total_tokens = int(data["metadata"]["total_tokens"])
        batch_size = int(data["metadata"]["batch_size"])
        tokens_per_sequence = int(data["metadata"]["tokens_per_sequence"])
        query = data["query"].to(device=device, copy=True).contiguous()
        key = data["key"].to(device=device, copy=True).contiguous()
        value = data["value"].to(device=device, copy=True).contiguous()
        prepared = {
            "query": query.unsqueeze(0),
            "key": key.unsqueeze(0),
            "value": value.unsqueeze(0),
            "beta": data["beta"].to(device=device, copy=True).contiguous().unsqueeze(0),
            "g": data["g"].to(device=device, copy=True).contiguous().unsqueeze(0),
            # Official FLA allocates [NK, *value.shape], where NK=1 for D=128.
            "output": torch.empty(
                1,
                1,
                total_tokens,
                data["metadata"]["num_value_heads"],
                data["metadata"]["head_dim"],
                dtype=torch.bfloat16,
                device=device,
            ),
            "state": torch.zeros(
                total_tokens + 1,
                data["metadata"]["num_value_heads"],
                data["metadata"]["head_dim"],
                data["metadata"]["head_dim"],
                dtype=torch.bfloat16,
                device=device,
            ),
            "cu_seqlens": torch.arange(
                0,
                total_tokens + 1,
                tokens_per_sequence,
                dtype=torch.int32,
                device=device,
            ),
            "ssm_state_indices": torch.arange(
                1, total_tokens + 1, dtype=torch.int32, device=device
            ).reshape(batch_size, tokens_per_sequence),
            "num_accepted_tokens": None,
            "scale": data["metadata"]["scale"],
        }
        if tokens_per_sequence > 1:
            prepared["num_accepted_tokens"] = torch.full(
                (batch_size,),
                tokens_per_sequence,
                dtype=torch.int32,
                device=device,
            )
        return prepared

    def execute_core_operator(self, prepared: dict[str, Any]) -> torch.Tensor:
        module = self.kernel_module()
        query = prepared["query"]
        key = prepared["key"]
        value = prepared["value"]
        batch, total_tokens, num_heads, key_dim = key.shape
        num_value_heads = value.shape[2]
        value_dim = value.shape[-1]
        num_sequences = prepared["cu_seqlens"].numel() - 1
        block_k = module.triton.next_power_of_2(key_dim)
        block_v = min(module.triton.next_power_of_2(value_dim), 32)
        num_k_blocks = module.triton.cdiv(key_dim, block_k)
        num_v_blocks = module.triton.cdiv(value_dim, block_v)
        if num_k_blocks != 1:
            raise RuntimeError(f"direct-out provider requires NK=1, got {num_k_blocks}")
        state = prepared["state"]
        indices = prepared["ssm_state_indices"]
        if indices.ndim == 1:
            stride_indices_seq, stride_indices_tok = indices.stride(0), 1
        else:
            stride_indices_seq, stride_indices_tok = indices.stride()
        grid = (num_k_blocks, num_v_blocks, num_sequences * num_value_heads)
        module.fused_recurrent_gated_delta_rule_fwd_kernel[grid](
            q=query,
            k=key,
            v=value,
            g=prepared["g"],
            beta=prepared["beta"],
            o=prepared["output"],
            h0=state,
            ht=state,
            cu_seqlens=prepared["cu_seqlens"],
            ssm_state_indices=indices,
            num_accepted_tokens=prepared["num_accepted_tokens"],
            scale=prepared["scale"],
            N=num_sequences,
            T=total_tokens,
            B=batch,
            H=num_heads,
            HV=num_value_heads,
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            stride_init_state_token=state.stride(0),
            stride_final_state_token=state.stride(0),
            stride_indices_seq=stride_indices_seq,
            stride_indices_tok=stride_indices_tok,
            IS_BETA_HEADWISE=prepared["beta"].ndim == value.ndim,
            USE_QK_L2NORM_IN_KERNEL=False,
            INPLACE_FINAL_STATE=True,
            IS_KDA=False,
            num_warps=1,
            num_stages=3,
        )
        return prepared["output"].squeeze(0)
