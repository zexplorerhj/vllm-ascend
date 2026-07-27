"""CUDA implementations used by the PagedAttention operator benchmark."""

import math
from typing import Any, Dict

import torch
import torch.nn.functional as F


class FlashInferPagedKVImpl:
    """FlashInfer paged-KV decode attention with a preplanned ``out=`` path.

    ``prepare_data`` intentionally owns all allocation and planning.  The
    framework's timed region therefore contains only ``wrapper.run`` and every
    prepared invocation has independent Q/K/V/output/workspace storage.
    """

    name = "cuda_flashinfer_fa2"

    def __init__(self, backend: str = "fa2", workspace_bytes: int = 128 << 20):
        if backend != "fa2":
            raise ValueError(
                f"formal FlashInfer provider requires backend='fa2', got {backend!r}"
            )
        self._wrapper_cls = None
        self.backend = backend
        self.workspace_bytes = workspace_bytes

    def _wrapper_class(self):
        if self._wrapper_cls is not None:
            return self._wrapper_cls
        try:
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"FlashInfer paged decode is unavailable: {exc}"
            ) from exc
        self._wrapper_cls = BatchDecodeWithPagedKVCacheWrapper
        return self._wrapper_cls

    @staticmethod
    def _copy_tensor(
        tensor: torch.Tensor,
        *,
        device: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # ``copy=True`` matters when the benchmark base tensors already live on
        # CUDA: plain ``to`` would otherwise return the same storage for all V2
        # invocations and defeat the fresh-storage protocol.
        return tensor.to(device=device, dtype=dtype, copy=True).contiguous()

    def prepare_data(
        self, data: Dict[str, Any], device: str, precision
    ) -> Dict[str, Any]:
        if not device.startswith("cuda"):
            raise ValueError(f"FlashInfer requires a CUDA device, got {device}")

        block_size = int(data["block_size"])
        if block_size != 128:
            raise ValueError(
                f"formal FlashInfer FA2 requires block_size=128, got {block_size}"
            )
        num_heads = int(data["num_heads"])
        num_kv_heads = int(data["num_kv_heads"])
        head_size = int(data["head_size"])
        if num_heads % num_kv_heads:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads "
                f"({num_kv_heads})"
            )
        if num_heads != 8 * num_kv_heads:
            raise ValueError(
                "formal FlashInfer FA2 requires GQA 8:1, got "
                f"{num_heads}:{num_kv_heads}"
            )

        context_lens_cpu = data["context_lens"].to(
            device="cpu", dtype=torch.int32
        )
        block_table_cpu = data["block_table"].to(
            device="cpu", dtype=torch.int32
        )
        if torch.any(context_lens_cpu <= 0):
            raise ValueError("FlashInfer requires every context length to be positive")
        page_counts = torch.div(
            context_lens_cpu + block_size - 1,
            block_size,
            rounding_mode="floor",
        )
        if int(page_counts.max()) > block_table_cpu.shape[1]:
            raise ValueError(
                "block_table does not contain enough logical pages for "
                f"context_lens: need {int(page_counts.max())}, have "
                f"{block_table_cpu.shape[1]}"
            )
        indptr_cpu = torch.empty(
            context_lens_cpu.numel() + 1, dtype=torch.int32
        )
        indptr_cpu[0] = 0
        torch.cumsum(page_counts, dim=0, out=indptr_cpu[1:])
        indices_cpu = torch.cat([
            block_table_cpu[row, :int(page_count)]
            for row, page_count in enumerate(page_counts.tolist())
        ])
        num_cache_blocks = int(data["key_cache"].shape[0])
        if int(indices_cpu.min()) < 0 or int(indices_cpu.max()) >= num_cache_blocks:
            raise ValueError(
                "block_table contains an out-of-range physical page id for "
                f"a cache with {num_cache_blocks} blocks"
            )
        last_page_len_cpu = (context_lens_cpu - 1).remainder(block_size) + 1

        query = self._copy_tensor(
            data["query"], device=device, dtype=precision.value
        )
        key_cache = self._copy_tensor(
            data["key_cache"], device=device, dtype=precision.value
        )
        value_cache = self._copy_tensor(
            data["value_cache"], device=device, dtype=precision.value
        )
        block_table = self._copy_tensor(
            block_table_cpu, device=device, dtype=torch.int32
        )
        context_lens = self._copy_tensor(
            context_lens_cpu, device=device, dtype=torch.int32
        )
        indptr = self._copy_tensor(
            indptr_cpu, device=device, dtype=torch.int32
        )
        indices = self._copy_tensor(
            indices_cpu, device=device, dtype=torch.int32
        )
        last_page_len = self._copy_tensor(
            last_page_len_cpu, device=device, dtype=torch.int32
        )
        workspace = torch.zeros(
            self.workspace_bytes, dtype=torch.uint8, device=device
        )
        output = torch.empty_like(query)

        wrapper = self._wrapper_class()(
            workspace,
            "NHD",
            use_tensor_cores=True,
            backend=self.backend,
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            pos_encoding_mode="NONE",
            q_data_type=precision.value,
            kv_data_type=precision.value,
            o_data_type=precision.value,
            sm_scale=float(data["scale"]),
            block_tables=block_table,
            seq_lens=context_lens,
        )

        return {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "output": output,
            "block_table": block_table,
            "context_lens": context_lens,
            "indptr": indptr,
            "indices": indices,
            "last_page_len": last_page_len,
            "workspace": workspace,
            "wrapper": wrapper,
        }

    def execute_core_operator(self, prepared_data: Dict[str, Any]) -> torch.Tensor:
        return prepared_data["wrapper"].run(
            prepared_data["query"],
            (prepared_data["key_cache"], prepared_data["value_cache"]),
            out=prepared_data["output"],
        )

    def run_full_implementation(
        self, data: Dict[str, Any], device: str, precision
    ) -> torch.Tensor:
        prepared_data = self.prepare_data(data, device, precision)
        return self.execute_core_operator(prepared_data).cpu().float()


class FlashAttentionPagedKVImpl:
    """Paged-KV decode attention provided by flash-attn."""

    name = "cuda_flash_attn_with_kvcache"

    def __init__(self) -> None:
        try:
            from flash_attn import flash_attn_with_kvcache
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(f"flash_attn_with_kvcache is unavailable: {exc}") from exc
        self._operator = flash_attn_with_kvcache

    def prepare_data(
        self, data: Dict[str, Any], device: str, precision
    ) -> Dict[str, Any]:
        block_size = data["block_size"]
        if block_size % 256 != 0:
            raise ValueError(
                "flash_attn_with_kvcache paged KV requires block_size to be "
                f"a multiple of 256, got {block_size}"
            )

        num_heads = data["num_heads"]
        num_kv_heads = data["num_kv_heads"]
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads "
                f"({num_kv_heads})"
            )

        query = data["query"].to(
            device=device, dtype=precision.value, copy=True
        )
        if query.ndim == 3:
            query = query.unsqueeze(1)

        return {
            "query": query.contiguous(),
            "key_cache": data["key_cache"].to(
                device=device, dtype=precision.value, copy=True
            ).contiguous(),
            "value_cache": data["value_cache"].to(
                device=device, dtype=precision.value, copy=True
            ).contiguous(),
            "block_table": data["block_table"].to(
                device=device, dtype=torch.int32, copy=True
            ).contiguous(),
            "context_lens": data["context_lens"].to(
                device=device, dtype=torch.int32, copy=True
            ).contiguous(),
            "scale": data["scale"],
        }

    def execute_core_operator(self, prepared_data: Dict[str, Any]) -> torch.Tensor:
        output = self._operator(
            q=prepared_data["query"],
            k_cache=prepared_data["key_cache"],
            v_cache=prepared_data["value_cache"],
            cache_seqlens=prepared_data["context_lens"],
            block_table=prepared_data["block_table"],
            softmax_scale=prepared_data["scale"],
            causal=True,
        )
        return output.squeeze(1)

    def run_full_implementation(
        self, data: Dict[str, Any], device: str, precision
    ) -> torch.Tensor:
        prepared_data = self.prepare_data(data, device, precision)
        return self.execute_core_operator(prepared_data).cpu().float()


class TorchSDPAPagedKVFallbackImpl:
    """Semantic CUDA fallback when flash-attn is not installed.

    This implementation gathers the logical pages and runs one SDPA call per
    sequence. It is intentionally exposed under a distinct provider name: its
    latency is not representative of a fused paged-attention kernel.
    """

    name = "cuda_torch_sdpa_paged_kv_fallback"

    def prepare_data(
        self, data: Dict[str, Any], device: str, precision
    ) -> Dict[str, Any]:
        query = data["query"].to(
            device=device, dtype=precision.value, copy=True
        )
        return {
            "query": query.contiguous(),
            "key_cache": data["key_cache"].to(
                device=device, dtype=precision.value, copy=True
            ).contiguous(),
            "value_cache": data["value_cache"].to(
                device=device, dtype=precision.value, copy=True
            ).contiguous(),
            "block_table": data["block_table"].to(
                device=device, dtype=torch.int64, copy=True
            ).contiguous(),
            "context_lens": data["context_lens"].to(
                device=device, dtype=torch.int32, copy=True
            ).contiguous(),
            "block_size": data["block_size"],
            "scale": data["scale"],
            "num_heads": data["num_heads"],
            "num_kv_heads": data["num_kv_heads"],
        }

    def execute_core_operator(self, prepared_data: Dict[str, Any]) -> torch.Tensor:
        outputs = []
        block_size = prepared_data["block_size"]
        num_heads = prepared_data["num_heads"]
        num_kv_heads = prepared_data["num_kv_heads"]

        for batch_idx in range(prepared_data["query"].shape[0]):
            seq_len = int(prepared_data["context_lens"][batch_idx].item())
            blocks_needed = math.ceil(seq_len / block_size)
            block_ids = prepared_data["block_table"][batch_idx, :blocks_needed]
            key = prepared_data["key_cache"].index_select(0, block_ids)
            value = prepared_data["value_cache"].index_select(0, block_ids)
            key = key.reshape(-1, num_kv_heads, key.shape[-1])[:seq_len]
            value = value.reshape(-1, num_kv_heads, value.shape[-1])[:seq_len]

            query = prepared_data["query"][batch_idx].view(
                1, num_heads, 1, -1
            )
            key = key.permute(1, 0, 2).unsqueeze(0)
            value = value.permute(1, 0, 2).unsqueeze(0)
            if num_heads != num_kv_heads:
                repeat_factor = num_heads // num_kv_heads
                key = key.repeat_interleave(repeat_factor, dim=1)
                value = value.repeat_interleave(repeat_factor, dim=1)

            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=False,
                scale=prepared_data["scale"],
            )
            outputs.append(output.squeeze(0).squeeze(1))

        return torch.stack(outputs, dim=0)

    def run_full_implementation(
        self, data: Dict[str, Any], device: str, precision
    ) -> torch.Tensor:
        prepared_data = self.prepare_data(data, device, precision)
        return self.execute_core_operator(prepared_data).cpu().float()
