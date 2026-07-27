import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
from typing import Dict, Any, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operator_test_framework import BaseOperatorTest, PrecisionType, DeviceType


class AddOperatorTest(BaseOperatorTest):
    """Add算子测试实现"""

    CUDA_IMPLEMENTATION = "cuda_torch_add_out"
    NPU_IMPLEMENTATION = "npu_torch_add_out"
    
    def __init__(self):
        super().__init__("Add")
        self.supported_precisions = [PrecisionType.FP16, PrecisionType.BF16]
        self.supported_devices = [
            DeviceType.CPU,
            DeviceType.NPU,
            DeviceType.GPU,
        ]
    
    def generate_test_data(
        self,
        shape: tuple = (1024, 1024),
        value_range: tuple = (-1.0, 1.0),
        **kwargs
    ) -> Dict[str, Any]:
        """生成Add算子测试数据"""
        
        # 生成两个随机tensor
        tensor_a = torch.rand(shape) * (value_range[1] - value_range[0]) + value_range[0]
        tensor_b = torch.rand(shape) * (value_range[1] - value_range[0]) + value_range[0]
        
        return {
            'tensor_a': tensor_a,
            'tensor_b': tensor_b,
            'metadata': {
                'shape': shape,
                'value_range': value_range,
                'operator_type': 'Add',
                'total_elements': tensor_a.numel()
            }
        }
    
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """运行CPU参考实现"""
        return data['tensor_a'] + data['tensor_b']
    
    def run_device_implementation(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行设备实现"""

        implementation = self._resolve_implementation(device, implementation)
        prepared = self._prepare_data_for_core_operator(
            data, device, precision, implementation
        )
        return self._execute_core_operator(
            prepared, implementation
        ).cpu().float()
    
    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现列表"""
        return self.get_formal_implementations(device)

    def get_formal_implementations(self, device: str) -> List[str]:
        """Return the fixed native provider used by formal curves."""
        if device.startswith("cuda"):
            return [self.CUDA_IMPLEMENTATION]
        if device.startswith("npu"):
            return [self.NPU_IMPLEMENTATION]
        return []

    def _resolve_implementation(
        self, device: str, implementation: str
    ) -> str:
        """Resolve and validate the formal provider for ``device``."""
        formal = self.get_formal_implementations(device)
        if implementation == "default":
            if not formal:
                raise ValueError(f"Add 不支持设备 {device}")
            return formal[0]
        if implementation not in formal:
            raise ValueError(
                f"实现 {implementation!r} 不适用于 {device}; formal={formal}"
            )
        return implementation
    
    def calculate_throughput(self, data: Dict[str, Any], time_ms: float) -> float:
        """计算吞吐量（GFLOPS）"""
        total_elements = data['metadata']['total_elements']
        # Add操作每个元素1次浮点运算
        total_ops = total_elements
        return (total_ops / (time_ms / 1000)) / 1e9  # GFLOPS
    
    def get_precision_config(self, precision: PrecisionType = None) -> Dict[str, int]:
        """获取精度配置"""
        if precision == PrecisionType.FP16:
            return {'dtype_size': 2}
        elif precision == PrecisionType.BF16:
            return {'dtype_size': 2}
        elif precision == PrecisionType.FP32:
            return {'dtype_size': 4}
        else:
            # 默认为2字节 (FP16/BF16)
            return {'dtype_size': 2}

    def calculate_bandwidth(self, data: Dict[str, Any], avg_time_ms: float, precision: PrecisionType = None) -> Optional[float]:
        """计算内存带宽（GB/s）
        
        Args:
            data: 测试数据
            avg_time_ms: 平均执行时间（毫秒）
            precision: 精度类型
            
        Returns:
            Optional[float]: 内存带宽（GB/s）
        """
        if avg_time_ms <= 0:
            return None
            
        # 获取元素总数
        total_elements = data['metadata']['total_elements']
        
        # 获取精度大小
        precision_config = self.get_precision_config(precision)
        dtype_size = precision_config.get('dtype_size', 2)
        
        # 计算总内存访问量（字节）
        # 读取tensor_a: total_elements * dtype_size
        # 读取tensor_b: total_elements * dtype_size
        # 写入result: total_elements * dtype_size
        total_bytes = 3 * total_elements * dtype_size
        
        # 转换为GB/s：总字节数 / (时间_秒 * 10^9)
        avg_time_s = avg_time_ms / 1000.0
        bandwidth_gb_s = total_bytes / (avg_time_s * 1e9)
        
        return bandwidth_gb_s
    
    def _prepare_data_for_core_operator(self, data: Dict[str, Any], device: str, precision: PrecisionType, implementation: str = "default") -> Dict[str, Any]:
        """为核心算子准备数据（排除预处理开销）"""
        # 将数据移动到目标设备和精度
        implementation = self._resolve_implementation(device, implementation)
        tensor_a = data['tensor_a'].to(
            dtype=precision.value, device=device, copy=True
        )
        tensor_b = data['tensor_b'].to(
            dtype=precision.value, device=device, copy=True
        )
        output = torch.empty_like(tensor_a)
        
        return {
            'tensor_a': tensor_a,
            'tensor_b': tensor_b,
            'output': output,
            'implementation': implementation
        }
    
    def _execute_core_operator(self, prepared_data: Dict[str, Any], implementation: str = "default") -> torch.Tensor:
        """执行核心算子（只测量核心计算，不包括数据移动）"""
        # 根据实现类型选择不同的加法方式
        impl = prepared_data.get('implementation', implementation)
        if impl in (self.CUDA_IMPLEMENTATION, self.NPU_IMPLEMENTATION):
            return torch.add(
                prepared_data['tensor_a'],
                prepared_data['tensor_b'],
                out=prepared_data['output'],
            )
        if impl == "torch_add" or impl == "default":
            return torch.add(prepared_data['tensor_a'], prepared_data['tensor_b'])
        elif impl == "operator_add":
            return prepared_data['tensor_a'] + prepared_data['tensor_b']  # 使用操作符重载
        else:
            raise ValueError(f"不支持的实现类型: {impl}")
