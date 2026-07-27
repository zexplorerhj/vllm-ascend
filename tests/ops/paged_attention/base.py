"""
PagedAttention算子测试基类
包含通用的数据生成、CPU参考实现等
"""

import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import math
from typing import Dict, Any, List, Tuple
from operator_test_framework import BaseOperatorTest, PrecisionType, DeviceType


class PagedAttentionOperatorTest(BaseOperatorTest):
    """PagedAttention算子测试基类"""
    
    def __init__(self):
        super().__init__("paged_attention")
        self.supported_precisions = [PrecisionType.BF16]  # 只测试BF16精度
        self.supported_devices = [DeviceType.CPU]
        if torch_npu is not None and torch_npu.npu.is_available():
            self.supported_devices.append(DeviceType.NPU)
        if torch.cuda.is_available():
            self.supported_devices.append(DeviceType.GPU)
        
        # 导入具体实现
        from .original_impl import OriginalPagedAttentionImpl
        from .fused_infer_attention_score_impl import FusedInferAttentionScoreImpl
        from .cuda_impl import FlashInferPagedKVImpl
        
        self.implementations = {
            "npu_original": OriginalPagedAttentionImpl(),
            "npu_fused_infer_attention_score": FusedInferAttentionScoreImpl()
        }

        self.npu_implementation = "npu_fused_infer_attention_score"

        # Provider construction is import-safe. FlashInfer is resolved lazily
        # during prepare and failure is explicit; formal curves never fallback.
        cuda_impl = FlashInferPagedKVImpl(backend="fa2")
        self.implementations[cuda_impl.name] = cuda_impl
        self.cuda_implementation = cuda_impl.name

    @staticmethod
    def _primary_output(result):
        """Select the correctness tensor while timing retains native tuples."""
        return result[0] if isinstance(result, (tuple, list)) else result
    
    def generate_test_data(
        self,
        batch_size: int = 4,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_size: int = 128,
        max_seq_len: int = 2048,
        num_blocks: int = 1000,
        block_size: int = 128,
        **kwargs
    ) -> Dict[str, Any]:
        """生成PagedAttention测试数据"""
        
        # 生成随机序列长度
        context_lens = torch.randint(
            low=1, 
            high=max_seq_len + 1, 
            size=(batch_size,),
            dtype=torch.int32
        )
        
        # 生成query
        query = torch.randn(
            batch_size, num_heads, head_size,
            dtype=torch.float32
        )
        
        # 生成key_cache和value_cache
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size,
            dtype=torch.float32
        )
        value_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size,
            dtype=torch.float32
        )
        
        # 生成block_table
        block_table = torch.zeros(batch_size, (max_seq_len + block_size - 1) // block_size, dtype=torch.int32)
        for i in range(batch_size):
            seq_len = context_lens[i].item()
            num_blocks_needed = (seq_len + block_size - 1) // block_size
            # 随机分配blocks，块ID范围是 [0, num_blocks-1]
            # 每个序列的块可以是非连续的，分散在不同的物理块中
            available_blocks = torch.randperm(num_blocks)[:num_blocks_needed]
            block_table[i, :num_blocks_needed] = available_blocks
        
        # 计算scale
        scale = 1.0 / math.sqrt(head_size)
        
        return {
            'query': query,
            'key_cache': key_cache,
            'value_cache': value_cache,
            'block_table': block_table,
            'context_lens': context_lens,
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'head_size': head_size,
            'scale': scale,
            'block_size': block_size,
            'metadata': {
                'batch_size': batch_size,
                'max_seq_len': max_seq_len,
                'num_blocks': num_blocks,
                'test_name': 'paged_attention'
            }
        }

    def generate_latency_test_data(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        head_size: int = 128,
        num_blocks: int = 0,
        block_size: int = 128,
        seed: int = 0,
        storage_device: str = "cpu",
    ) -> Dict[str, Any]:
        """Generate a compact, deterministic decode-attention benchmark input.

        ``num_blocks=0`` allocates disjoint physical pages for every batch row.
        Supplying a smaller explicit pool is allowed for shared-prefix/cache
        experiments and is recorded in metadata, but is not the default used
        by the latency curves.
        """
        if batch_size <= 0 or max_seq_len <= 0:
            raise ValueError("batch_size and max_seq_len must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        blocks_per_sequence = math.ceil(max_seq_len / block_size)
        disjoint_num_blocks = blocks_per_sequence * batch_size
        effective_num_blocks = num_blocks or disjoint_num_blocks
        if effective_num_blocks < blocks_per_sequence:
            raise ValueError(
                f"num_blocks={effective_num_blocks} is too small for "
                f"max_seq_len={max_seq_len}, block_size={block_size}; "
                f"need at least {blocks_per_sequence}"
            )

        tensor_device = torch.device(storage_device)
        if tensor_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA storage requested but unavailable: {storage_device}"
            )

        query_generator = torch.Generator(device=tensor_device)
        key_generator = torch.Generator(device=tensor_device)
        value_generator = torch.Generator(device=tensor_device)
        query_generator.manual_seed(seed)
        key_generator.manual_seed(seed + 1)
        value_generator.manual_seed(seed + 2)
        query = torch.randn(
            batch_size,
            num_heads,
            head_size,
            generator=query_generator,
            dtype=torch.bfloat16,
            device=tensor_device,
        )
        key_cache = torch.randn(
            effective_num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            generator=key_generator,
            dtype=torch.bfloat16,
            device=tensor_device,
        )
        value_cache = torch.randn(
            effective_num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            generator=value_generator,
            dtype=torch.bfloat16,
            device=tensor_device,
        )

        context_lens = torch.full(
            (batch_size,), max_seq_len, dtype=torch.int32
        )
        base_blocks = torch.arange(blocks_per_sequence, dtype=torch.int32)
        block_rows = []
        disjoint_pages = effective_num_blocks >= disjoint_num_blocks
        for batch_idx in range(batch_size):
            offset = (
                seed + batch_idx * blocks_per_sequence
            ) % effective_num_blocks
            block_rows.append((base_blocks + offset) % effective_num_blocks)
        block_table = torch.stack(block_rows, dim=0)

        return {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "block_table": block_table,
            "context_lens": context_lens,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "scale": 1.0 / math.sqrt(head_size),
            "block_size": block_size,
            "metadata": {
                "batch_size": batch_size,
                "max_seq_len": max_seq_len,
                "num_blocks": effective_num_blocks,
                "block_size": block_size,
                "blocks_per_sequence": blocks_per_sequence,
                "page_pool_policy": "disjoint" if disjoint_pages else "shared",
                "seed": seed,
                "storage_device": str(tensor_device),
                "test_name": "paged_attention_latency",
            },
        }
    
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """运行CPU参考实现"""
        return self._cpu_paged_attention(
            query=data['query'],
            key_cache=data['key_cache'],
            value_cache=data['value_cache'],
            block_table=data['block_table'],
            context_lens=data['context_lens'],
            num_heads=data['num_heads'],
            num_kv_heads=data['num_kv_heads'],
            head_size=data['head_size'],
            scale=data['scale'],
            block_size=data['block_size']
        )
    
    def run_device_implementation(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行设备实现"""
        prepared = self._prepare_data_for_core_operator(
            data, device, precision, implementation
        )
        native_result = self._execute_core_operator(prepared, implementation)
        output = self._primary_output(native_result)
        num_heads = prepared.get("num_heads")
        head_size = prepared.get("head_size")
        if num_heads is not None and head_size is not None:
            output = output.view(-1, num_heads, head_size)
        return output.cpu().float()
    
    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现方式"""
        return self.get_formal_implementations(device)

    def get_formal_implementations(self, device: str) -> List[str]:
        """Return formal providers without probing optional accelerator libs."""
        if device.startswith("npu"):
            return [self.npu_implementation]
        if device.startswith("cuda"):
            return [self.cuda_implementation]
        return []

    def get_preferred_implementation(self, device: str) -> str:
        """Return the provider used by cross-hardware latency curves."""
        if device.startswith("cuda"):
            return self.cuda_implementation
        if device.startswith("npu"):
            return self.npu_implementation
        raise ValueError(f"PagedAttention does not support device {device}")
    
    def calculate_flops(self, data: Dict[str, Any]) -> float:
        """计算PagedAttention的FLOPS
        
        Args:
            data: 测试数据
            
        Returns:
            float: FLOPS数量
        """
        # 直接从顶层获取参数，更简洁
        batch_size = data['metadata']['batch_size']
        num_heads = data['num_heads']
        num_kv_heads = data['num_kv_heads']
        head_size = data['head_size']
        context_lens = data['context_lens']
        
        total_flops = 0
        
        for seq_len in context_lens:
            seq_len = seq_len.item() if hasattr(seq_len, 'item') else seq_len
            
            # PagedAttention的FLOPS计算（修正版本）：
            # Q: [batch_size, num_heads, head_size]
            # K: [seq_len, num_kv_heads, head_size] -> 扩展到 [seq_len, num_heads, head_size]
            # V: [seq_len, num_kv_heads, head_size] -> 扩展到 [seq_len, num_heads, head_size]
            
            # PagedAttention FLOPS详细计算：
            
            # 注意：这里计算的是单个序列的FLOPS，因为外层已经有for循环遍历每个batch
            
            # 1. Q @ K^T矩阵乘法
            # Q: [1, num_heads, 1, head_size] (当前序列的当前token)
            # K: [1, num_heads, seq_len, head_size] (当前序列的历史tokens)
            # 结果: [1, num_heads, 1, seq_len]
            # FLOPS: num_heads * seq_len * head_size * 2 (乘法+加法)
            qk_flops = num_heads * seq_len * head_size * 2
            
            # 2. Scale操作 (除以sqrt(head_size))
            # 对attention scores进行缩放: num_heads * seq_len
            scale_flops = num_heads * seq_len
            
            # 3. Softmax计算
            # 包括: max查找 + exp + sum + 归一化
            # 每个元素需要: 1(max) + 1(sub) + 1(exp) + 1(sum) + 1(div) = 5 ops
            softmax_flops = num_heads * seq_len * 5
            
            # 4. Attention @ V矩阵乘法
            # Attention: [1, num_heads, 1, seq_len]
            # V: [1, num_heads, seq_len, head_size]
            # 结果: [1, num_heads, 1, head_size]
            # FLOPS: num_heads * seq_len * head_size * 2
            av_flops = num_heads * seq_len * head_size * 2
            
            total_flops += qk_flops + scale_flops + softmax_flops + av_flops
        
        return float(total_flops)

    def calculate_throughput(self, data: Dict[str, Any], avg_time_ms: float) -> float:
        """计算吞吐量（GOPS - 每秒十亿次操作）
        
        Args:
            data: 测试数据
            avg_time_ms: 平均执行时间（毫秒）
            
        Returns:
            float: GOPS值
        """
        if avg_time_ms <= 0:
            return 0.0
            
        flops = self.calculate_flops(data)
        
        # 转换为GOPS：FLOPS / (时间_秒 * 10^9)
        avg_time_s = avg_time_ms / 1000.0
        gops = flops / (avg_time_s * 1e9)
        
        return gops

    def calculate_bandwidth(self, data: Dict[str, Any], avg_time_ms: float) -> float:
        """计算内存带宽（GB/s）
        
        Args:
            data: 测试数据
            avg_time_ms: 平均执行时间（毫秒）
            
        Returns:
            float: 内存带宽（GB/s）
        """
        if avg_time_ms <= 0:
            return 0.0
            
        # 直接从顶层获取参数，与calculate_flops保持一致
        batch_size = data['metadata']['batch_size']
        num_heads = data['num_heads']
        num_kv_heads = data['num_kv_heads']
        head_size = data['head_size']
        block_size = data['block_size']
        context_lens = data['context_lens']
        
        # 假设使用BF16，每个元素2字节
        dtype_size = 2
        
        total_bytes = 0
        
        for seq_len in context_lens:
            seq_len = seq_len.item() if hasattr(seq_len, 'item') else seq_len
            
            # PagedAttention内存访问计算（修正版本）：
            
            # 注意：这里计算的是单个序列的内存访问，因为外层已经有for循环遍历每个batch
            
            # 1. Query读取: [1, num_heads, head_size] (当前序列)
            query_bytes = num_heads * head_size * dtype_size
            
            # 2. Key/Value Cache读取（PagedAttention的核心）
            num_blocks_needed = math.ceil(seq_len / block_size)
            # 实际读取的KV数据量（考虑页对齐）
            actual_kv_len = num_blocks_needed * block_size
            
            # 基础KV Cache读取：只读取实际存储的KV头
            kv_cache_bytes = actual_kv_len * num_kv_heads * head_size * dtype_size * 2  # K和V
            
            # 3. KV头扩展的内存开销
            kv_expand_bytes = 0
            if num_kv_heads != num_heads and num_kv_heads > 0:
                # KV头扩展主要是内存访问开销，不是额外存储
                # 每个KV头需要被多个Query头访问
                expand_ratio = num_heads // num_kv_heads
                # 扩展访问：每个KV元素被访问expand_ratio次
                # 但实际实现中通常是广播，内存访问开销较小
                # 这里估算为额外的重复访问开销
                kv_expand_bytes = kv_cache_bytes * (expand_ratio - 1) * 0.1  # 10%的额外开销
            
            # 4. 中间结果存储
            # Attention scores: [1, num_heads, seq_len] (当前序列)
            attention_scores_bytes = num_heads * seq_len * dtype_size
            # Softmax中间结果（max, sum等）
            softmax_temp_bytes = num_heads * dtype_size * 2  # max + sum
            
            # 5. Output写入: [1, num_heads, head_size] (当前序列)
            output_bytes = num_heads * head_size * dtype_size
            
            # 6. Block table访问（整数索引）- 当前序列的块索引
            block_table_bytes = num_blocks_needed * 4  # int32
            
            total_bytes += (query_bytes + kv_cache_bytes + kv_expand_bytes + 
                          attention_scores_bytes + softmax_temp_bytes + 
                          output_bytes + block_table_bytes)
        
        # 转换为GB/s：总字节数 / (时间_秒 * 10^9)
        avg_time_s = avg_time_ms / 1000.0
        bandwidth_gb_s = total_bytes / (avg_time_s * 1e9)
        
        return bandwidth_gb_s
    
    # 核心算子接口
    def run_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行核心算子操作（兼容性方法）"""
        if implementation == "default":
            implementation = (
                self.cuda_implementation if device.startswith("cuda")
                else self.npu_implementation
            )
            
        if implementation in self.implementations:
            impl = self.implementations[implementation]
            prepared_data = impl.prepare_data(data, device, precision)
            return impl.execute_core_operator(prepared_data)
        else:
            raise ValueError(f"不支持的实现方式: {implementation}")
    
    # 为了兼容性保留的方法
    def _prepare_data_for_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str
    ) -> Dict[str, Any]:
        """准备核心算子数据（兼容性方法）"""
        if implementation == "default":
            implementation = self.get_preferred_implementation(device)
        formal = self.get_formal_implementations(device)
        if implementation not in formal:
            raise ValueError(
                f"实现 {implementation!r} 不是 {device} formal provider; "
                f"formal={formal}"
            )
        block_size = int(data.get("block_size", 0))
        if block_size != 128:
            raise ValueError(
                f"formal PagedAttention requires block_size=128, got {block_size}"
            )
        if implementation in self.implementations:
            impl = self.implementations[implementation]
            prepared_data = impl.prepare_data(data, device, precision)
            prepared_data["_implementation"] = implementation
            return prepared_data
        else:
            raise ValueError(f"不支持的实现方式: {implementation}")
    
    def _execute_core_operator(self, prepared_data: Dict[str, Any], implementation: str) -> Any:
        """执行核心算子（兼容性方法）"""
        if implementation == "default":
            implementation = prepared_data.get("_implementation", "default")
        if implementation in self.implementations:
            impl = self.implementations[implementation]
            return impl.execute_core_operator(prepared_data)
        else:
            raise ValueError(f"不支持的实现方式: {implementation}")

    def _verify_preallocated_output_aliases(
        self,
        prepared_data_list: List[Dict[str, Any]],
        outputs: List[Any],
        implementation: str,
    ) -> int:
        """Verify ``out=`` providers returned each invocation's owned buffer."""
        if len(prepared_data_list) != len(outputs):
            raise RuntimeError(
                "prepared/output count mismatch: "
                f"{len(prepared_data_list)} != {len(outputs)}"
            )
        resolved = implementation
        if resolved == "default" and prepared_data_list:
            resolved = prepared_data_list[0].get("_implementation", "default")
        if resolved == "npu_fused_infer_attention_score":
            for index, (prepared_data, result) in enumerate(
                zip(prepared_data_list, outputs)
            ):
                if not isinstance(result, tuple) or len(result) != 2:
                    raise RuntimeError(
                        "NPU PagedAttention out overload returned an "
                        f"invalid result at {index}"
                    )
                expected_outputs = prepared_data["out"]
                if (
                    expected_outputs[0].untyped_storage().data_ptr()
                    == expected_outputs[1].untyped_storage().data_ptr()
                ):
                    raise RuntimeError(
                        "NPU PagedAttention primary and LSE out buffers alias "
                        f"each other at {index}"
                    )
                for output_index, (actual, expected) in enumerate(
                    zip(result, expected_outputs)
                ):
                    if not isinstance(actual, torch.Tensor):
                        raise RuntimeError(
                            "NPU PagedAttention out overload returned a "
                            f"non-tensor at {index}:{output_index}"
                        )
                    aliases = (
                        actual.untyped_storage().data_ptr()
                        == expected.untyped_storage().data_ptr()
                        and actual.data_ptr() == expected.data_ptr()
                        and actual.storage_offset()
                        == expected.storage_offset()
                        and actual.shape == expected.shape
                        and actual.dtype == expected.dtype
                        and actual.device == expected.device
                        and actual.stride() == expected.stride()
                    )
                    if not aliases:
                        raise RuntimeError(
                            "NPU PagedAttention output does not alias "
                            f"preallocated out at {index}:{output_index}"
                        )
            return len(outputs)

        if resolved not in {"cuda_flashinfer_fa2", "npu_original"}:
            return 0

        for index, (prepared_data, output) in enumerate(
            zip(prepared_data_list, outputs)
        ):
            expected = prepared_data["output"]
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    f"preallocated provider returned non-tensor output at {index}"
                )
            if (
                output.untyped_storage().data_ptr()
                != expected.untyped_storage().data_ptr()
            ):
                raise RuntimeError(
                    f"provider output does not alias preallocated out at {index}"
                )
        return len(outputs)
    
    def _cpu_paged_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        scale: float,
        block_size: int = 128
    ) -> torch.Tensor:
        """CPU参考实现"""
        batch_size = query.shape[0]
        output = torch.zeros_like(query)
        
        for i in range(batch_size):
            seq_len = context_lens[i].item()
            num_blocks = (seq_len + block_size - 1) // block_size
            
            # 获取当前序列的blocks
            blocks = block_table[i, :num_blocks]
            
            # 计算attention
            q = query[i].unsqueeze(0)  # [1, num_heads, head_size]
            
            # 收集key和value
            keys = []
            values = []
            for block_idx in blocks:
                block_start = 0
                block_end = min(block_size, seq_len - len(keys) * block_size)
                if block_end <= 0:
                    break
                    
                k_block = key_cache[block_idx, block_start:block_end, :]  # [block_len, num_kv_heads, head_size]
                v_block = value_cache[block_idx, block_start:block_end, :]  # [block_len, num_kv_heads, head_size]
                
                keys.append(k_block)  # [block_len, num_kv_heads, head_size]
                values.append(v_block)  # [block_len, num_kv_heads, head_size]
            
            if keys:
                k = torch.cat(keys, dim=0)  # [seq_len, num_kv_heads, head_size]
                v = torch.cat(values, dim=0)  # [seq_len, num_kv_heads, head_size]
                
                # 扩展kv heads到match query heads
                if num_kv_heads != num_heads:
                    repeat_factor = num_heads // num_kv_heads
                    k = k.repeat_interleave(repeat_factor, dim=1)
                    v = v.repeat_interleave(repeat_factor, dim=1)
                
                k = k.transpose(0, 1)  # [num_heads, seq_len, head_size]
                v = v.transpose(0, 1)  # [num_heads, seq_len, head_size]
                
                # 计算attention scores
                # q: [1, num_heads, head_size], k: [num_heads, seq_len, head_size]
                # 需要调整维度以进行正确的矩阵乘法
                q_expanded = q.transpose(0, 1)  # [num_heads, 1, head_size]
                scores = torch.matmul(q_expanded, k.transpose(-2, -1)) * scale  # [num_heads, 1, seq_len]
                attn_weights = torch.softmax(scores, dim=-1)
                
                # 应用attention weights
                out = torch.matmul(attn_weights, v)  # [num_heads, 1, head_size]
                out = out.squeeze(1)  # [num_heads, head_size]
                output[i] = out
        
        return output
