"""
GroupGemm算子基础抽象类
提供INT8和BF16版本的通用实现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import csv
from typing import Dict, Any, List, Optional, Tuple
from operator_test_framework import BaseOperatorTest, PrecisionType
import numpy as np
from abc import ABC, abstractmethod


class BaseGroupGemmOperatorTest(BaseOperatorTest, ABC):
    """GroupGemm算子基础抽象类"""

    CUDA_BF16_IMPLEMENTATION = "cuda_bmm_balanced_grouped_mm_jagged_bf16"
    CUDA_INT8_IMPLEMENTATION = "cuda_vllm_cutlass_scaled_mm_bf16"
    CUDA_INT8_FALLBACK_IMPLEMENTATION = (
        "cuda_torch_int_mm_col_major_raw_int32"
    )
    
    def __init__(self, operator_name: str = None, num_experts: int = 8, hidden_dim: int = 7168, 
                 out_channel: int = 4096, use_nz_format: bool = False):
        # 如果没有指定operator_name，根据是否使用NZ格式自动设置
        if operator_name is None:
            operator_name = "GroupGemm_NZ" if use_nz_format else "GroupGemm"
        
        super().__init__(operator_name)
        self.num_experts = num_experts      # 专家数量，默认8
        self.hidden_dim = hidden_dim        # 隐藏维度
        self.out_channel = out_channel      # 输出通道
        self.use_nz_format = use_nz_format  # 是否使用NZ格式（格式29）
        
    @abstractmethod
    def get_precision_config(self) -> Dict[str, Any]:
        """获取精度配置
        
        Returns:
            Dict[str, Any]: 精度配置字典，包含数据类型和输出类型
        """
        pass
    
    @abstractmethod
    def generate_precision_specific_data(self, seq_len: int, num_experts: int, 
                                       hidden_dim: int, out_channel: int) -> Dict[str, Any]:
        """生成精度特定的测试数据
        
        Args:
            seq_len: 序列长度
            num_experts: 专家数量
            hidden_dim: 隐藏维度
            out_channel: 输出通道
            
        Returns:
            Dict[str, Any]: 精度特定的数据
        """
        pass
    
    @abstractmethod
    def get_npu_grouped_matmul_kwargs(self, data: Dict[str, Any], group_list: torch.Tensor) -> Dict[str, Any]:
        """获取npu_grouped_matmul的参数
        
        Args:
            data: 测试数据
            group_list: 分组列表
            
        Returns:
            Dict[str, Any]: npu_grouped_matmul的参数字典
        """
        pass
    
    def _apply_nz_format(self, weight: torch.Tensor) -> torch.Tensor:
        """应用NZ格式（格式29）到权重张量
        
        Args:
            weight: 权重张量，必须在NPU设备上
            
        Returns:
            torch.Tensor: NZ格式的权重张量
        """
        try:
            # 确保张量在NPU设备上
            if not weight.device.type == 'npu':
                raise ValueError(f"NZ格式转换需要在NPU设备上进行，当前设备: {weight.device}")
            
            # 使用torch_npu.npu_format_cast将权重转换为NZ格式（格式29）
            nz_weight = torch_npu.npu_format_cast(weight, 29)
            return nz_weight
        except Exception as e:
            print(f"警告：NZ格式转换失败，使用原始格式: {e}")
            return weight
    
    def generate_test_data(self, seq_len: int = 1024, num_experts: int = None, 
                          hidden_dim: int = None, out_channel: int = None, **kwargs) -> Dict[str, Any]:
        """生成测试数据
        
        Args:
            seq_len: 序列长度
            num_experts: 专家数量，如果不指定则使用默认值
            hidden_dim: 隐藏维度，如果不指定则使用默认值
            out_channel: 输出通道，如果不指定则使用默认值
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 测试数据字典
        """
        if num_experts is None:
            num_experts = self.num_experts
        if hidden_dim is None:
            hidden_dim = self.hidden_dim
        if out_channel is None:
            out_channel = self.out_channel
            
        # 计算分组信息（通用逻辑）
        base = seq_len // num_experts
        remainder = seq_len % num_experts
        
        group_item_array = [base] * num_experts
        for i in range(remainder):
            group_item_array[i] += 1
            
        group_list = torch.tensor(group_item_array, dtype=torch.int64)
        
        # 获取精度特定的数据
        precision_data = self.generate_precision_specific_data(seq_len, num_experts, hidden_dim, out_channel)
        
        # 合并通用数据和精度特定数据
        result = {
            'group_list': group_list,
            'seq_len': seq_len,
            'num_experts': num_experts,
            'hidden_dim': hidden_dim,
            'out_channel': out_channel,
            'use_nz_format': self.use_nz_format
        }
        result.update(precision_data)
        
        return result
    
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """CPU参考实现（简化版本）
        
        Args:
            data: 测试数据
            
        Returns:
            torch.Tensor: CPU计算结果
        """
        # 对于 groupgemm，CPU参考实现比较复杂
        # 这里返回一个占位符结果
        seq_len = data['seq_len']
        out_channel = data.get('out_channel', self.out_channel)
        precision_config = self.get_precision_config()
        output_dtype = precision_config.get('output_dtype', torch.bfloat16)
        return torch.zeros((seq_len, out_channel), dtype=output_dtype)
    
    def run_device_implementation(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> Any:
        """设备实现
        
        Args:
            data: 测试数据
            device: 设备类型
            precision: 精度类型
            implementation: 实现方式
            
        Returns:
            torch.Tensor: 设备计算结果
        """
        prepared_data = self._prepare_data_for_core_operator(
            data, device, precision, implementation
        )
        result = self._execute_core_operator(
            prepared_data, prepared_data["_implementation"]
        )
        return self._coalesce_group_outputs(result)
    
    def run_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> Any:
        """运行核心算子操作（用于精确性能测试）
        
        Args:
            data: 测试数据
            device: 设备类型
            precision: 精度类型
            implementation: 实现方式
            
        Returns:
            torch.Tensor: 计算结果
        """
        prepared_data = self._prepare_data_for_core_operator(
            data, device, precision, implementation
        )
        return self._execute_core_operator(
            prepared_data, prepared_data["_implementation"]
        )

    def _resolve_implementation(self, device: str, implementation: str) -> str:
        """解析并验证当前设备的 GroupGemm 实现。"""
        formal = self.get_formal_implementations(device)
        if not formal:
            raise RuntimeError(f"{device} 上没有可用的 GroupGemm 实现")

        if implementation == "default":
            return formal[0]
        if implementation not in formal:
            raise ValueError(
                f"实现 {implementation!r} 不适用于 {device}；"
                f"formal 实现: {formal}"
            )
        return implementation

    @staticmethod
    def _cuda_grouped_mm_callable():
        """优先返回 PyTorch 2.11 公开的 grouped_mm 包装。"""
        functional_op = getattr(torch.nn.functional, "grouped_mm", None)
        if functional_op is not None:
            return functional_op
        return getattr(torch, "_grouped_mm", None)

    @staticmethod
    def _cuda_cutlass_scaled_mm_callable():
        """Resolve the low-level out= CUTLASS op after lazy vLLM registration."""
        try:
            from vllm import _custom_ops as vllm_ops  # noqa: F401
        except (ImportError, OSError, RuntimeError):
            return None

        c_namespace = getattr(torch.ops, "_C", None)
        if c_namespace is None:
            return None
        op = getattr(c_namespace, "cutlass_scaled_mm", None)
        return op if callable(op) else None

    @staticmethod
    def _coalesce_group_outputs(result: Any) -> torch.Tensor:
        """把完整设备调用的 group 输出统一成 [sum(M), N] Tensor。

        核心吞吐计时直接调用 ``_execute_core_operator``，因此多 expert
        CUTLASS 输出的 cat 不会混入 GEMM kernel 时间。
        """
        if isinstance(result, torch.Tensor):
            if result.dim() == 3:
                return result.reshape(-1, result.shape[-1])
            return result
        if isinstance(result, (tuple, list)):
            if not result:
                raise RuntimeError("GroupGemm 返回了空的 expert 输出")
            if len(result) == 1:
                return result[0]
            if not all(isinstance(value, torch.Tensor) for value in result):
                raise TypeError("GroupGemm expert 输出必须全部是 Tensor")
            return torch.cat(tuple(result), dim=0)
        raise TypeError(f"未知 GroupGemm 输出类型: {type(result)!r}")

    @staticmethod
    def _group_counts(data: Dict[str, Any]) -> List[int]:
        """在进入设备核心算子前验证 expert token 分组。"""
        group_list = data.get("group_list")
        if not isinstance(group_list, torch.Tensor) or group_list.dim() != 1:
            raise ValueError("group_list 必须是 1D Tensor")
        x = data.get("x")
        weight = data.get("weight")
        if (not isinstance(x, torch.Tensor) or x.dim() != 2
                or not isinstance(weight, torch.Tensor) or weight.dim() != 3):
            raise ValueError("GroupGemm 要求 x 为 2D，weight 为 3D Tensor")
        if x.shape[1] != weight.shape[1]:
            raise ValueError(
                f"x 与 weight 的 K 维不一致: {x.shape[1]} != {weight.shape[1]}"
            )

        counts = [int(value) for value in group_list.detach().cpu().tolist()]
        if not counts or any(value <= 0 for value in counts):
            raise ValueError(f"group_list 中每个 expert 必须至少有一个 token: {counts}")
        if len(counts) != int(weight.shape[0]):
            raise ValueError(
                f"group_list expert 数 {len(counts)} 与 weight 的 expert 维 "
                f"{weight.shape[0]} 不一致"
            )
        if sum(counts) != int(x.shape[0]):
            raise ValueError(
                f"group_list token 总数 {sum(counts)} 与 x.shape[0]="
                f"{x.shape[0]} 不一致"
            )
        return counts

    def _prepare_cuda_core_data(
        self, data: Dict[str, Any], device: str, implementation: str
    ) -> Dict[str, Any]:
        """CUDA 预处理：转移不计时，计时区间只包含 GEMM。"""
        if self.use_nz_format:
            raise ValueError("NZ 是 Ascend 权重格式，CUDA GroupGemm 不支持 --use-nz-format")

        counts = self._group_counts(data)
        hidden_dim = int(data["x"].shape[1])
        out_channel = int(data["weight"].shape[2])
        if hidden_dim % 8 != 0 or out_channel % 8 != 0:
            raise ValueError(
                "CUDA GroupGemm 要求 K(hidden_dim) 和 N(out_channel) "
                f"都是 8 的倍数，当前为 K={hidden_dim}, N={out_channel}"
            )

        if implementation == self.CUDA_BF16_IMPLEMENTATION:
            if (data["x"].dtype != torch.bfloat16
                    or data["weight"].dtype != torch.bfloat16):
                raise TypeError(
                    "CUDA torch.bmm 路径要求 x/weight 都是 torch.bfloat16"
                )
            if len(set(counts)) != 1:
                raise ValueError(
                    "formal CUDA BF16 GroupGemm requires balanced expert rows "
                    f"for torch.bmm, got {counts}"
                )

            x = data["x"].to(device=device, copy=True)
            weight = data["weight"].to(device=device, copy=True)
            rows_per_expert = counts[0]
            output = torch.empty(
                len(counts),
                rows_per_expert,
                out_channel,
                dtype=torch.bfloat16,
                device=device,
            )
            return {
                "_implementation": implementation,
                "_kernel": "torch_bmm_cublas",
                "op": torch.bmm,
                "mat_a": x.reshape(
                    len(counts), rows_per_expert, hidden_dim
                ),
                "mat_b": weight,
                "output": output,
            }

        if implementation == self.CUDA_INT8_IMPLEMENTATION:
            scaled_mm = self._cuda_cutlass_scaled_mm_callable()
            if scaled_mm is None:
                raise RuntimeError(
                    "当前 vLLM 安装没有可用的 CUTLASS cutlass_scaled_mm"
                )
            if hidden_dim % 16 != 0 or out_channel % 16 != 0:
                raise ValueError(
                    "vLLM CUTLASS scaled-mm 要求 K 和 N 都是 16 的倍数，"
                    f"当前为 K={hidden_dim}, N={out_channel}"
                )
            if data["x"].dtype != torch.int8 or data["weight"].dtype != torch.int8:
                raise TypeError("CUDA INT8 CUTLASS 路径要求 x/weight 都是 torch.int8")

            scale = data.get("scale")
            per_token_scale = data.get("per_token_scale")
            expected_scale_shape = (len(counts), out_channel)
            if (not isinstance(scale, torch.Tensor)
                    or tuple(scale.shape) != expected_scale_shape):
                raise ValueError(
                    "INT8 GroupGemm scale 必须是 [num_experts, N]，"
                    f"期望 {expected_scale_shape}，实际 "
                    f"{getattr(scale, 'shape', None)}"
                )
            if (not isinstance(per_token_scale, torch.Tensor)
                    or per_token_scale.numel() != int(data["x"].shape[0])):
                raise ValueError(
                    "INT8 GroupGemm per_token_scale 必须为每个 token 提供一个 scale"
                )

            x = data["x"].to(device=device, copy=True)
            # CUTLASS 的 B=[K,N] 要求 column-major。先生成连续 [E,N,K]，
            # 再零拷贝转置，得到每个 expert stride=(1,K)。
            weight = data["weight"].to(device=device, copy=True)
            weight = weight.transpose(1, 2).contiguous().transpose(1, 2)
            token_scale = per_token_scale.to(
                device=device, dtype=torch.float32, copy=True
            )
            token_scale = token_scale.reshape(-1, 1)
            weight_scale = scale.to(
                device=device, dtype=torch.float32, copy=True
            )

            expert_inputs = []
            expert_outputs = []
            start = 0
            for expert, rows in enumerate(counts):
                end = start + rows
                expert_inputs.append((
                    x[start:end],
                    weight[expert],
                    token_scale[start:end],
                    weight_scale[expert].reshape(1, out_channel),
                ))
                expert_outputs.append(torch.empty(
                    rows,
                    out_channel,
                    dtype=torch.bfloat16,
                    device=device,
                ))
                start = end

            return {
                "_implementation": implementation,
                "_kernel": "vllm_cutlass_scaled_mm",
                "op": scaled_mm,
                "expert_inputs": expert_inputs,
                "expert_outputs": tuple(expert_outputs),
            }

        if implementation == self.CUDA_INT8_FALLBACK_IMPLEMENTATION:
            int_mm = getattr(torch, "_int_mm", None)
            if int_mm is None:
                raise RuntimeError("当前 PyTorch 没有 torch._int_mm")
            if data["x"].dtype != torch.int8 or data["weight"].dtype != torch.int8:
                raise TypeError("CUDA INT8 路径要求 x/weight 都是 torch.int8")

            # 当前 H20/PyTorch 栈实测 torch._int_mm 要求 M > 16 且
            # M 为 32 的倍数。profile 显示它落到 CUTLASS SM80
            # compatibility kernel，而不是 cuBLASLt；这里不做 padding。
            invalid_groups = [
                index for index, rows in enumerate(counts)
                if rows <= 16 or rows % 32 != 0
            ]
            if invalid_groups:
                raise ValueError(
                    "H20 上的 CUDA torch._int_mm 要求每个 expert 的 "
                    "M > 16 且 M % 32 == 0；"
                    f"不满足的 expert={invalid_groups}, group_rows={counts}。"
                    "为保持 FLOPS 语义不做隐式 padding"
                )

            x = data["x"].to(device)
            weight = data["weight"].to(device)
            weight = weight.transpose(1, 2).contiguous().transpose(1, 2)
            expert_inputs = []
            start = 0
            for expert, rows in enumerate(counts):
                end = start + rows
                expert_inputs.append((x[start:end], weight[expert]))
                start = end

            return {
                "_implementation": implementation,
                "_kernel": "torch_int_mm_col_major_raw_int32",
                "op": int_mm,
                "expert_inputs": expert_inputs,
            }

        raise ValueError(f"未知 CUDA GroupGemm 实现: {implementation}")
    
    def _prepare_data_for_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> Dict[str, Any]:
        """为核心算子准备数据（数据预处理，不计入性能测试时间）
        
        Args:
            data: 原始测试数据
            device: 设备类型
            precision: 精度类型
            implementation: 实现方式
            
        Returns:
            Dict[str, Any]: 准备好的数据，包含 npu_grouped_matmul 的所有参数
        """
        implementation = self._resolve_implementation(device, implementation)

        if device.startswith("cuda"):
            return self._prepare_cuda_core_data(data, device, implementation)
        if not device.startswith("npu"):
            raise ValueError(f"GroupGemm 不支持设备 {device}")
        if torch_npu is None:
            raise RuntimeError("torch_npu 未安装，无法运行 npu_grouped_matmul")

        # 将数据移动到设备
        device_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                device_data[key] = value.to(device=device, copy=True)
            else:
                device_data[key] = value

        # 如果使用NZ格式，对权重进行格式转换
        if self.use_nz_format and 'weight' in device_data:
            device_data['weight'] = self._apply_nz_format(device_data['weight'])

        kwargs = self.get_npu_grouped_matmul_kwargs(
            device_data, device_data['group_list']
        )
        if self.get_precision_config().get("input_dtype") == torch.bfloat16:
            kwargs["bias"] = None
        return {"_implementation": implementation, "kwargs": kwargs}

    def _declares_preallocated_output_contract(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> bool:
        impl = prepared_data.get("_implementation", implementation)
        return impl in {
            self.CUDA_BF16_IMPLEMENTATION,
            self.CUDA_INT8_IMPLEMENTATION,
        }
    
    def _execute_core_operator(
        self, 
        prepared_data: Dict[str, Any], 
        implementation: str = "default"
    ) -> Any:
        """执行核心算子操作（只计算核心算子执行时间）
        
        Args:
            prepared_data: 预处理好的数据（来自 _prepare_data_for_core_operator）
            implementation: 实现方式
            
        Returns:
            torch.Tensor: 计算结果
        """
        actual_implementation = prepared_data.get("_implementation")
        if implementation != "default" and implementation != actual_implementation:
            raise ValueError(
                f"预处理数据使用 {actual_implementation}，执行时却请求 {implementation}"
            )

        if actual_implementation == "npu_grouped_matmul":
            return torch_npu.npu_grouped_matmul(**prepared_data["kwargs"])
        if actual_implementation == self.CUDA_BF16_IMPLEMENTATION:
            prepared_data["op"](
                prepared_data["mat_a"],
                prepared_data["mat_b"],
                out=prepared_data["output"],
            )
            output = prepared_data["output"]
            return output.reshape(-1, output.shape[-1])
        if actual_implementation == self.CUDA_INT8_IMPLEMENTATION:
            for expert_input, expert_output in zip(
                prepared_data["expert_inputs"],
                prepared_data["expert_outputs"],
            ):
                expert_x, expert_weight, token_scale, weight_scale = (
                    expert_input
                )
                prepared_data["op"](
                    expert_output,
                    expert_x,
                    expert_weight,
                    token_scale,
                    weight_scale,
                    None,
                )
            return prepared_data["expert_outputs"]
        if actual_implementation == self.CUDA_INT8_FALLBACK_IMPLEMENTATION:
            # 诊断 fallback：raw INT32 accumulator，不应用 scale。
            return tuple(
                prepared_data["op"](expert_x, expert_weight)
                for expert_x, expert_weight in prepared_data["expert_inputs"]
            )
        raise ValueError(f"未知 GroupGemm 实现: {actual_implementation}")

    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现方式
        
        Args:
            device: 设备类型
            
        Returns:
            List[str]: 可用实现列表
        """
        return self.get_formal_implementations(device)

    def get_formal_implementations(self, device: str) -> List[str]:
        """Return formal providers without probing optional accelerator libs."""
        if device.startswith("npu"):
            return ["npu_grouped_matmul"]
        if device.startswith("cuda"):
            input_dtype = self.get_precision_config().get("input_dtype")
            if input_dtype == torch.bfloat16:
                return [self.CUDA_BF16_IMPLEMENTATION]
            if input_dtype == torch.int8:
                return [self.CUDA_INT8_IMPLEMENTATION]
        return []
    
    def calculate_flops(self, data: Dict[str, Any]) -> Optional[float]:
        """计算FLOPS
        
        Args:
            data: 测试数据
            
        Returns:
            Optional[float]: FLOPS数量
        """
        seq_len = data.get('seq_len', 0)
        hidden_dim = data.get('hidden_dim', 0)
        out_channel = data.get('out_channel', 0)
        num_experts = data.get('num_experts', 0)
        
        if seq_len <= 0 or hidden_dim <= 0 or out_channel <= 0 or num_experts <= 0:
            return None
        
        # GroupGemm的FLOPS计算：总的矩阵乘法运算量
        # 每个token都要与对应expert的权重相乘
        # 矩阵乘法: seq_len * hidden_dim * out_channel * 2 (乘法+加法)
        # 注意：这里seq_len是总的序列长度，每个token只与一个expert计算
        total_flops = seq_len * hidden_dim * out_channel * 2
        
        return float(total_flops)

    def calculate_throughput(self, data: Dict[str, Any], avg_time_ms: float) -> Optional[float]:
        """计算吞吐量（GOPS - 每秒十亿次操作）
        
        Args:
            data: 测试数据
            avg_time_ms: 平均执行时间（毫秒）
            
        Returns:
            Optional[float]: GOPS值
        """
        if avg_time_ms <= 0:
            return None
            
        flops = self.calculate_flops(data)
        if flops is None:
            return None
            
        # 转换为GOPS：FLOPS / (时间_秒 * 10^9)
        avg_time_s = avg_time_ms / 1000.0
        gops = flops / (avg_time_s * 1e9)
        
        return gops

    def calculate_bandwidth(self, data: Dict[str, Any], avg_time_ms: float) -> Optional[float]:
        """计算内存带宽（GB/s）
        
        Args:
            data: 测试数据
            avg_time_ms: 平均执行时间（毫秒）
            
        Returns:
            Optional[float]: 内存带宽（GB/s）
        """
        if avg_time_ms <= 0:
            return None
            
        # 计算总内存访问量（字节）
        seq_len = data.get('seq_len', 0)
        hidden_dim = data.get('hidden_dim', 0)
        out_channel = data.get('out_channel', 0)
        num_experts = data.get('num_experts', 0)
        
        if seq_len <= 0 or hidden_dim <= 0 or out_channel <= 0 or num_experts <= 0:
            return None
            
        precision_config = self.get_precision_config()
        input_dtype_size = precision_config.get('input_dtype_size', 2)  # BF16默认2字节
        weight_dtype_size = precision_config.get('weight_dtype_size', 2)
        output_dtype_size = precision_config.get('output_dtype_size', 2)
        implementation = data.get("benchmark_implementation")
        if implementation == self.CUDA_INT8_FALLBACK_IMPLEMENTATION:
            # torch._int_mm 返回 INT32 accumulator。
            output_dtype_size = 4
        
        # 计算内存访问量（字节）
        # 输入读取：seq_len * hidden_dim * input_dtype_size
        # 权重读取：num_experts * hidden_dim * out_channel * weight_dtype_size
        # 输出写入：seq_len * out_channel * output_dtype_size
        input_bytes = seq_len * hidden_dim * input_dtype_size
        weight_bytes = num_experts * hidden_dim * out_channel * weight_dtype_size
        output_bytes = seq_len * out_channel * output_dtype_size
        
        # 如果有bias，也要计算bias的内存访问
        if (data.get('bias') is not None
                and implementation != self.CUDA_BF16_IMPLEMENTATION):
            bias_dtype_size = precision_config.get('bias_dtype_size', 4)  # FP32默认4字节
            bias_bytes = num_experts * out_channel * bias_dtype_size
        else:
            bias_bytes = 0
            
        # 如果有scale等量化参数，也要计算
        scale_bytes = 0
        if (data.get('scale') is not None
                and implementation != self.CUDA_INT8_FALLBACK_IMPLEMENTATION):
            # CUDA CUTLASS 在 prepare 中把 NPU 的 BF16 channel scale 转 FP32。
            scale_dtype_size = (
                4 if implementation == self.CUDA_INT8_IMPLEMENTATION else 2
            )
            scale_bytes += num_experts * out_channel * scale_dtype_size
        if (data.get('per_token_scale') is not None
                and implementation != self.CUDA_INT8_FALLBACK_IMPLEMENTATION):
            scale_bytes += seq_len * 4  # FP32
            
        total_bytes = input_bytes + weight_bytes + output_bytes + bias_bytes + scale_bytes
        
        # 转换为GB/s：总字节数 / (时间_秒 * 10^9)
        avg_time_s = avg_time_ms / 1000.0
        bandwidth_gb_s = total_bytes / (avg_time_s * 1e9)
        
        return bandwidth_gb_s


    
    def calculate_memory_usage(self, data: Dict[str, Any]) -> Dict[str, float]:
        """计算内存使用量（详细版本）
        
        Args:
            data: 测试数据
            
        Returns:
            Dict[str, float]: 内存使用量信息（单位：MB）
        """
        seq_len = data.get('seq_len', 0)
        hidden_dim = data.get('hidden_dim', 0)
        out_channel = data.get('out_channel', 0)
        num_experts = data.get('num_experts', 0)
        
        precision_config = self.get_precision_config()
        input_dtype_size = precision_config.get('input_dtype_size', 2)  # BF16默认2字节
        weight_dtype_size = precision_config.get('weight_dtype_size', 2)
        output_dtype_size = precision_config.get('output_dtype_size', 2)
        
        # 计算各部分内存使用（字节）
        input_memory = seq_len * hidden_dim * input_dtype_size
        weight_memory = num_experts * hidden_dim * out_channel * weight_dtype_size
        output_memory = seq_len * out_channel * output_dtype_size
        
        # 转换为MB
        input_mb = input_memory / (1024 * 1024)
        weight_mb = weight_memory / (1024 * 1024)
        output_mb = output_memory / (1024 * 1024)
        total_mb = input_mb + weight_mb + output_mb
        
        return {
            'input_memory_mb': input_mb,
            'weight_memory_mb': weight_mb,
            'output_memory_mb': output_mb,
            'total_memory_mb': total_mb
        }
    
    def validate_result(self, result: torch.Tensor, data: Dict[str, Any]) -> bool:
        """验证结果的有效性
        
        Args:
            result: 计算结果
            data: 测试数据
            
        Returns:
            bool: 验证是否通过
        """
        if result is None:
            return False
        
        expected_shape = (data['seq_len'], data.get('out_channel', self.out_channel))
        if result.shape != expected_shape:
            return False
        
        # 检查是否有NaN或Inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            return False
        
        return True
