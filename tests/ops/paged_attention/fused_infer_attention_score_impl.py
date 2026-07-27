"""
PagedAttention使用npu_fused_infer_attention_score算子实现
参考: https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/attention_v1.py
"""

from typing import Any, Callable, Dict

import torch


class FusedInferAttentionScoreImpl:
    """PagedAttention使用npu_fused_infer_attention_score实现"""
    
    def __init__(self):
        self.name = "npu_fused_infer_attention_score"

    @staticmethod
    def _out_operator() -> Callable[..., Any]:
        """Resolve the registered out overload outside the measured region."""
        try:
            return torch.ops.npu.npu_fused_infer_attention_score.out
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                "torch.ops.npu.npu_fused_infer_attention_score.out "
                "is unavailable"
            ) from exc
    
    def prepare_data(self, data: Dict[str, Any], device: str, precision) -> Dict[str, Any]:
        """准备数据用于核心算子执行 - TND格式"""
        query = data['query'].to(
            dtype=precision.value, device=device, copy=True
        )
        key_cache = data['key_cache'].to(
            dtype=precision.value, device=device, copy=True
        )
        value_cache = data['value_cache'].to(
            dtype=precision.value, device=device, copy=True
        )
        block_table = data['block_table'].to(
            device=device, copy=True
        )
        
        # 准备序列长度列表
        seq_lens_list = data['context_lens'].tolist()
        
        # 构造 seq_lens_q 为 Cumulative Sum
        # TND 模式下，actual_seq_lengths 需要是 Query 的累积长度 (End Points)
        # Decode 阶段，每个 query 长度为 1
        # 例如: [1, 2, 3, ..., batch_size]
        batch_size = len(seq_lens_list)
        query_lens_list = list(range(1, batch_size + 1))
        
        block_size = data['block_size']
        
        # 重塑query为TND格式 (num_tokens, num_heads, head_size)
        # 假设 batch_size 个 token, 每个长度为 1
        # query shape: [batch_size, num_heads, head_size]
        # 对于 decode 阶段，seq_len=1，所以 T=batch_size
        num_tokens = query.shape[0]
        query_reshaped = query.view(num_tokens, data['num_heads'], -1)
        
        # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
        # value_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
        num_block, block_size, _, _ = key_cache.shape
        key_cache = key_cache.view(num_block, block_size, -1)
        value_cache = value_cache.view(num_block, block_size, -1)

        attn_mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(device=device)

        operator_kwargs = {
            'query': query_reshaped,
            'key': key_cache,
            'value': value_cache,
            'atten_mask': attn_mask,
            'block_table': block_table,
            'input_layout': "TND",
            'block_size': block_size,
            'actual_seq_lengths': query_lens_list,
            'actual_seq_lengths_kv': seq_lens_list,
            'num_heads': data['num_heads'],
            'num_key_value_heads': data['num_kv_heads'],
            'scale': data['scale'],
            'sparse_mode': 3,
            'softmax_lse_flag': False,
        }
        output = torch.empty_like(query_reshaped)
        softmax_lse = torch.empty(
            1,
            dtype=torch.float32,
            device=query_reshaped.device,
        )
        return {
            'operator': self._out_operator(),
            'operator_kwargs': operator_kwargs,
            'out': [output, softmax_lse],
        }
    
    def execute_core_operator(self, prepared_data: Dict[str, Any]):
        """执行已注册的 out 重载并返回原生输出 tuple。"""
        return prepared_data['operator'](
            **prepared_data['operator_kwargs'],
            out=prepared_data['out'],
        )
    
    def run_full_implementation(self, data: Dict[str, Any], device: str, precision) -> torch.Tensor:
        """运行完整实现"""
        prepared_data = self.prepare_data(data, device, precision)
        native_result = self.execute_core_operator(prepared_data)
        output = native_result[0]
        
        # 后处理
        operator_kwargs = prepared_data['operator_kwargs']
        num_heads = operator_kwargs['num_heads']
        head_size = data['head_size']
        num_tokens = operator_kwargs['query'].shape[0] # T dimension
        
        # 调整输出形状 [num_tokens, num_heads, head_size] -> [batch_size, num_heads, head_size]
        # 在 decode 阶段 num_tokens == batch_size
        output = output.view(num_tokens, num_heads, head_size)
        
        return output.cpu().float()
