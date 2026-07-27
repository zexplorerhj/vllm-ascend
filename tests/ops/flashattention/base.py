from contextlib import nullcontext
import torch
import math
from typing import Dict, Any, List, Optional
from operator_test_framework import BaseOperatorTest, PrecisionType, DeviceType

class FlashAttentionOperatorTest(BaseOperatorTest):
    """FlashAttention算子测试基类"""
    
    def __init__(self):
        super().__init__("flash_attention")
        self.supported_precisions = [PrecisionType.BF16, PrecisionType.FP16]
        self.supported_devices = [
            DeviceType.CPU,
            DeviceType.NPU,
            DeviceType.GPU,
        ]
        
        # 导入具体实现
        from .impl import (
            FlashAttentionCudaImpl,
            FlashAttentionExternalCudaImpl,
            FlashAttentionNpuImpl,
        )
        
        self.implementations = {
            "npu_flash_attention": FlashAttentionNpuImpl(),
            "cuda_sdpa_flash_attention": FlashAttentionCudaImpl(),
            "cuda_flash_attn_func": FlashAttentionExternalCudaImpl(),
        }

    @staticmethod
    def _primary_output(result):
        """Select the correctness tensor while timed paths retain native tuples."""
        return result[0] if isinstance(result, (tuple, list)) else result
    
    def generate_test_data(
        self,
        batch_size: int = 4,
        num_heads: int = 32,
        seq_len: int = 128,
        head_size: int = 128,
        num_kv_heads: int = None,
        input_layout: str = "BNSD",
        sparse_mode: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """生成FlashAttention测试数据"""
        
        if num_kv_heads is None:
            num_kv_heads = num_heads
            
        # 生成query, key, value (BNSD base)
        query_bnsd = torch.randn(
            batch_size, num_heads, seq_len, head_size,
            dtype=torch.float32
        )
        key_bnsd = torch.randn(
            batch_size, num_kv_heads, seq_len, head_size,
            dtype=torch.float32
        )
        value_bnsd = torch.randn(
            batch_size, num_kv_heads, seq_len, head_size,
            dtype=torch.float32
        )
        
        # Prepare actual data based on layout
        data = {
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'head_size': head_size,
            'input_layout': input_layout,
            'sparse_mode': sparse_mode,
            'cpu_query': query_bnsd, # Keep BNSD for CPU reference
            'cpu_key': key_bnsd,
            'cpu_value': value_bnsd,
            'metadata': {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'num_heads': num_heads,
                'num_kv_heads': num_kv_heads,
                'head_size': head_size,
                'sparse_mode': sparse_mode,
                'test_name': 'flash_attention'
            }
        }
        
        if input_layout == "BNSD":
            data['query'] = query_bnsd
            data['key'] = key_bnsd
            data['value'] = value_bnsd
            
        elif input_layout == "TND":
            # Check if variable sequence lengths are requested
            if kwargs.get('variable_seq_lengths', False):
                # Generate random sequence lengths
                import random
                min_seq_len = kwargs.get('min_seq_len', 1)
                actual_seq_lengths = []
                for _ in range(batch_size):
                    actual_seq_lengths.append(random.randint(min_seq_len, seq_len))
                
                # For CPU reference, we need to mask out padding tokens
                # Create mask for BNSD
                mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
                for i, l in enumerate(actual_seq_lengths):
                    mask[i, :, :, l:] = True # Mask out future tokens (padding)
                    # Also need to handle causal mask if sparse_mode is set?
                    # SDPA handles is_causal, but for variable length padding we might need explicit attn_mask
                
                # If sparse_mode is causal, we combine masks?
                # For simplicity in this test gen, let's rely on SDPA's attn_mask support
                # Note: SDPA attn_mask: logical True means "ignore" (in some versions) or add -inf.
                # PyTorch SDPA docs: "Binary mask ... True values indicate elements that should take part in attention" (Wait, check docs)
                # Actually usually: float mask with -inf for masked out. Or bool mask where True = allow?
                # PyTorch docs: "attn_mask: ... Boolean mask ... True indicates that the element *is attended to*."
                # So we want False for padding.
                
                # Let's rebuild mask: True = valid, False = padding
                attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.bool) # All False
                for i, l in enumerate(actual_seq_lengths):
                    attn_mask[i, :, :l, :l] = True
                    
                # If causal, we need to apply causal mask on top
                if sparse_mode in [2, 3]:
                    # Create causal mask for each sequence
                    # Causal mask: [i, j] is valid if j <= i
                    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
                    # attn_mask &= causal_mask
                    # Need to broadcast correctly: attn_mask is [B, 1, S, S]
                    # causal_mask is [S, S]
                    attn_mask = attn_mask & causal_mask.unsqueeze(0).unsqueeze(0)
                
                # Store mask for CPU reference
                # We need to pass this to run_cpu_reference
                data['cpu_attn_mask'] = attn_mask
                
                # Construct TND data by concatenating valid tokens
                # TND format usually packs tokens: [Token1_seq1, Token2_seq1, ..., Token1_seq2, ...]
                tnd_queries = []
                tnd_keys = []
                tnd_values = []
                
                for i in range(batch_size):
                    l = actual_seq_lengths[i]
                    # BNSD -> (N, S, D) -> (S, N, D)
                    q_s = query_bnsd[i, :, :l, :].transpose(0, 1)
                    k_s = key_bnsd[i, :, :l, :].transpose(0, 1)
                    v_s = value_bnsd[i, :, :l, :].transpose(0, 1)
                    tnd_queries.append(q_s)
                    tnd_keys.append(k_s)
                    tnd_values.append(v_s)
                
                data['query'] = torch.cat(tnd_queries, dim=0)
                data['key'] = torch.cat(tnd_keys, dim=0)
                data['value'] = torch.cat(tnd_values, dim=0)
                
                data['actual_seq_lengths'] = actual_seq_lengths
                data['actual_seq_lengths_kv'] = actual_seq_lengths # Assume same length for KV for now
                
                # Update metadata
                data['metadata']['actual_seq_lengths'] = actual_seq_lengths
                
            else:
                # Fixed sequence length (original logic)
                total_tokens = batch_size * seq_len
                data['query'] = query_bnsd.transpose(1, 2).reshape(total_tokens, num_heads, head_size)
                data['key'] = key_bnsd.transpose(1, 2).reshape(total_tokens, num_kv_heads, head_size)
                data['value'] = value_bnsd.transpose(1, 2).reshape(total_tokens, num_kv_heads, head_size)
                
                # Generate actual_seq_lengths
                data['actual_seq_lengths'] = [seq_len] * batch_size
                data['actual_seq_lengths_kv'] = [seq_len] * batch_size
            
            # Optional: Generate atten_mask if needed (e.g. for causal)
            # For sparse_mode=3 (right down causal), NPU op handles it, but we can pass mask if needed
            # User example passed atten_mask=attn_metadata.attn_mask
            if sparse_mode != 0:
                 # Create a causal mask for testing if needed, though sparse_mode usually handles it
                 pass
            
            # Add block_table stub if user wants to test that path (though data is contiguous here)
            # For strict TND without PagedAttention, block_table might not be used, 
            # but user provided code passes it.
            # We will leave block_table as None by default unless specifically requested in kwargs?
            # Or we can generate a dummy one.
            if kwargs.get('use_block_table', False):
                 # Simple dummy block table
                 block_size = kwargs.get('block_size', 128)
                 # Note: for variable seq lengths, num_blocks per seq varies
                 # But block_table usually has shape (batch, max_num_blocks)
                 max_blocks = (seq_len + block_size - 1) // block_size
                 data['block_table'] = torch.arange(max_blocks * batch_size, dtype=torch.int32).reshape(batch_size, max_blocks)
                 data['block_size'] = block_size

        else:
            raise ValueError(f"Test generation does not support layout {input_layout}")
            
        return data
    
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """运行CPU参考实现 (使用PyTorch原生SDPA)"""
        # Prefer pre-stored BNSD data for CPU reference
        query = data.get('cpu_query', data['query'])
        key = data.get('cpu_key', data['key'])
        value = data.get('cpu_value', data['value'])
        
        # 处理 GQA (Grouped Query Attention)
        # 如果 key/value 的 heads 数少于 query，需要进行 repeat/expand
        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        if num_kv_heads < num_heads:
            # key: [B, H_kv, S, D] -> [B, H_q, S, D]
            num_groups = num_heads // num_kv_heads
            # repeat_interleave repeats elements, e.g. [1, 2] -> [1, 1, 2, 2]
            key = key.repeat_interleave(num_groups, dim=1)
            value = value.repeat_interleave(num_groups, dim=1)
        
        # 处理 sparse_mode
        is_causal = data.get('sparse_mode', 0) in [2, 3] # 2: left up causal, 3: right down causal
        
        attn_mask = data.get('cpu_attn_mask')
        
        # PyTorch SDPA expects BNSD
        
        if attn_mask is not None:
             # If explicit mask provided (e.g. for variable lengths), use it
             # And disable is_causal because we integrated causal into mask if needed
             # Note: SDPA with attn_mask does not support is_causal=True in some versions?
             # Docs: "The argument is_causal is mutually exclusive with attn_mask"
             
             # Expand mask if needed for GQA?
             # attn_mask shape is [B, 1, S, S] usually. It broadcasts over heads.
             # So no expansion needed for heads.
             
             output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=attn_mask,
                is_causal=False
            )
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, 
                is_causal=is_causal
            )
        
        return output
    
    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现方式"""
        return self.get_formal_implementations(device)

    def get_formal_implementations(self, device: str) -> List[str]:
        """Return every fixed provider reported by formal curves."""
        if device.startswith("cuda"):
            return [
                "cuda_sdpa_flash_attention",
                "cuda_flash_attn_func",
            ]
        if device.startswith("npu"):
            return ["npu_flash_attention"]
        return []

    def _resolve_implementation(self, device: str, implementation: str) -> str:
        formal = self.get_formal_implementations(device)
        if implementation == "default":
            if not formal:
                raise ValueError(f"FlashAttention 不支持设备 {device}")
            return formal[0]
        if implementation not in formal:
            raise ValueError(
                f"实现 {implementation!r} 不适用于 {device}; formal={formal}"
            )
        return implementation

    def _core_operator_benchmark_context(
        self,
        device: str,
        precision: PrecisionType,
        implementation: str,
    ):
        del precision
        resolved = self._resolve_implementation(device, implementation)
        if resolved == "cuda_sdpa_flash_attention":
            return self.implementations[resolved].flash_backend_context()
        return nullcontext()
        
    def run_device_implementation(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行设备实现"""
        implementation = self._resolve_implementation(device, implementation)
        impl = self.implementations[implementation]
        prepared = impl.prepare_data(data, device, precision)
        if implementation == "cuda_sdpa_flash_attention":
            with impl.flash_backend_context():
                native_result = impl.execute_core_operator_in_active_context(
                    prepared
                )
        else:
            native_result = impl.execute_core_operator(prepared)
        output = self._primary_output(native_result)
        if implementation == "cuda_flash_attn_func":
            output = output.transpose(1, 2)
        return output.cpu().float()

    def calculate_flops(self, data: Dict[str, Any], mode: str = "fwd") -> Optional[float]:
        metadata = data.get("metadata", {})
        batch_size = metadata.get("batch_size", data.get("batch_size"))
        num_heads = metadata.get("num_heads", data.get("num_heads"))
        seq_len = metadata.get("seq_len", data.get("seq_len"))
        head_size = metadata.get("head_size", data.get("head_size"))
        sparse_mode = data.get("sparse_mode", metadata.get("sparse_mode", 0))

        if None in (batch_size, num_heads, seq_len, head_size):
            return None

        flops_per_matmul = 2.0 * batch_size * num_heads * seq_len * seq_len * head_size
        total_flops = 2.0 * flops_per_matmul
        if sparse_mode in [2, 3]:
            total_flops *= 0.5
        if mode == "bwd":
            total_flops *= 2.5
        return float(total_flops)

    def calculate_throughput(self, data: Dict[str, Any], time_ms: float) -> Optional[float]:
        flops = self.calculate_flops(data, mode="fwd")
        if flops is None or time_ms <= 0:
            return None
        return (flops / (time_ms / 1000.0)) / 1e9

    def calculate_tflops(self, data: Dict[str, Any], time_ms: float, mode: str = "fwd") -> Optional[float]:
        flops = self.calculate_flops(data, mode=mode)
        if flops is None or time_ms <= 0:
            return None
        return (flops / (time_ms / 1000.0)) / 1e12

    # 为了兼容性保留的方法
    def _prepare_data_for_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str
    ) -> Dict[str, Any]:
        """准备核心算子数据（兼容性方法）"""
        implementation = self._resolve_implementation(device, implementation)
        impl = self.implementations[implementation]
        return {
            "_implementation": implementation,
            "provider_data": impl.prepare_data(data, device, precision),
        }
    
    def _execute_core_operator(
        self,
        prepared_data: Dict[str, Any],
        implementation: str,
    ) -> Any:
        """执行核心算子（兼容性方法）"""
        stored_implementation = prepared_data["_implementation"]
        if (
            implementation != "default"
            and implementation != stored_implementation
        ):
            raise ValueError(
                "FlashAttention prepared data provenance mismatch: "
                f"prepared for {stored_implementation!r}, but execution "
                f"requested {implementation!r}"
            )
        implementation = stored_implementation
        provider_data = prepared_data["provider_data"]
        if implementation in self.implementations:
            impl = self.implementations[implementation]
            if implementation == "cuda_sdpa_flash_attention":
                return impl.execute_core_operator_in_active_context(
                    provider_data
                )
            return impl.execute_core_operator(provider_data)
        else:
            raise ValueError(f"不支持的实现方式: {implementation}")
