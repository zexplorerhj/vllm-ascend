import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operator_test_framework import BaseOperatorTest, PrecisionType, DeviceType


class LinearOperatorTest(BaseOperatorTest):
    """Linear算子测试实现"""

    CUDA_IMPLEMENTATION = "cuda_torch_mm_out"
    NPU_IMPLEMENTATION = "npu_torch_linear"
    
    def __init__(self):
        super().__init__("Linear")
        self.supported_precisions = [PrecisionType.FP16, PrecisionType.BF16]
        self.supported_devices = [DeviceType.NPU, DeviceType.GPU]
    
    def generate_test_data(
        self,
        batch_size: int = 32,
        input_dim: int = 1024,
        output_dim: int = 512,
        bias: bool = True,
        value_range: tuple = (-1.0, 1.0),
        **kwargs
    ) -> Dict[str, Any]:
        """生成Linear算子测试数据
        
        Args:
            batch_size: 批次大小
            input_dim: 输入维度
            output_dim: 输出维度
            bias: 是否使用偏置
            value_range: 数值范围
        """
        
        # 生成输入tensor (batch_size, input_dim)
        input_tensor = torch.rand(batch_size, input_dim) * (value_range[1] - value_range[0]) + value_range[0]
        
        # 生成权重tensor (output_dim, input_dim)
        weight_tensor = torch.rand(output_dim, input_dim) * (value_range[1] - value_range[0]) + value_range[0]
        
        # 生成偏置tensor (output_dim,) 如果需要的话
        bias_tensor = None
        if bias:
            bias_tensor = torch.rand(output_dim) * (value_range[1] - value_range[0]) + value_range[0]
        
        return {
            'input': input_tensor,
            'weight': weight_tensor,
            'bias': bias_tensor,
            'metadata': {
                'batch_size': batch_size,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'has_bias': bias,
                'value_range': value_range,
                'operator_type': 'Linear',
                'total_elements': input_tensor.numel() + weight_tensor.numel() + (bias_tensor.numel() if bias_tensor is not None else 0),
                'flops': batch_size * input_dim * output_dim * 2  # 每个输出元素需要input_dim次乘法和加法
            }
        }
    
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """运行CPU参考实现 - 使用torch.nn.functional.linear"""
        input_tensor = data['input'].cpu()
        weight_tensor = data['weight'].cpu()
        bias_tensor = data['bias'].cpu() if data['bias'] is not None else None
        
        # 使用torch.nn.functional.linear作为参考实现
        result = F.linear(input_tensor, weight_tensor, bias_tensor)
        
        return result
    
    def run_device_implementation(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行设备实现"""

        if implementation in (
            self.CUDA_IMPLEMENTATION,
            self.NPU_IMPLEMENTATION,
        ):
            prepared = self._prepare_data_for_core_operator(
                data, device, precision, implementation
            )
            return self._execute_core_operator(prepared, implementation)
        
        # 将数据移动到指定设备和精度
        input_tensor = data['input'].to(device=device, dtype=precision.value)
        weight_tensor = data['weight'].to(device=device, dtype=precision.value)
        bias_tensor = data['bias'].to(device=device, dtype=precision.value) if data['bias'] is not None else None
        
        # 确保设备同步
        if "npu" in device:
            torch_npu.npu.synchronize()
        elif "cuda" in device:
            torch.cuda.synchronize()
        
        # 执行Linear操作 - 使用torch.nn.functional.linear
        result = F.linear(input_tensor, weight_tensor, bias_tensor)
        
        # 确保计算完成
        if "npu" in device:
            torch_npu.npu.synchronize()
        elif "cuda" in device:
            torch.cuda.synchronize()
        
        return result
    
    def run_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行核心算子操作（用于精确性能测试）- 使用torch.nn.functional.linear"""
        
        return F.linear(data['input'], data['weight'], data['bias'])
    
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
            Dict[str, Any]: 准备好的数据
        """
        if implementation == "default":
            formal = self.get_formal_implementations(device)
            if not formal:
                raise ValueError(f"Linear 不支持设备 {device}")
            implementation = formal[0]

        input_tensor = data['input'].to(
            device=device, dtype=precision.value, copy=True
        )
        weight = data['weight'].to(
            device=device, dtype=precision.value, copy=True
        )
        prepared_data = {
            'input': input_tensor,
            'weight': weight,
            'implementation': implementation,
        }
        if implementation == self.CUDA_IMPLEMENTATION:
            prepared_data['weight_t'] = weight.t()
            prepared_data['output'] = torch.empty(
                input_tensor.shape[0],
                weight.shape[0],
                dtype=precision.value,
                device=device,
            )
        return prepared_data
    
    def _execute_core_operator(
        self, 
        prepared_data: Dict[str, Any], 
        implementation: str = "default"
    ) -> torch.Tensor:
        """执行核心算子操作（只计算核心算子执行时间）
        
        Args:
            prepared_data: 预处理好的数据（来自 _prepare_data_for_core_operator）
            implementation: 实现方式
            
        Returns:
            torch.Tensor: 计算结果
        """
        impl = prepared_data.get('implementation', implementation)
        if impl == self.CUDA_IMPLEMENTATION:
            return torch.mm(
                prepared_data['input'],
                prepared_data['weight_t'],
                out=prepared_data['output'],
            )
        if impl == self.NPU_IMPLEMENTATION:
            return F.linear(
                prepared_data['input'],
                prepared_data['weight'],
                None,
            )
        raise ValueError(f"不支持的 Linear 实现: {impl}")
    
    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现方式"""
        return self.get_formal_implementations(device)

    def get_formal_implementations(self, device: str) -> List[str]:
        """Return the fixed native provider used by formal curves."""
        if device.startswith("cuda"):
            return [self.CUDA_IMPLEMENTATION]
        if device.startswith("npu"):
            return [self.NPU_IMPLEMENTATION]
        return []
    
    def calculate_flops(self, data: Dict[str, Any]) -> Optional[float]:
        """计算浮点运算次数 (FLOPS)
        
        Args:
            data: 测试数据
            
        Returns:
            Optional[float]: 浮点运算次数
        """
        metadata = data.get('metadata', {})
        return metadata.get('flops', None)
    
    def calculate_throughput(self, data: Dict[str, Any], time_ms: float) -> float:
        """计算吞吐量 (GFLOPS)"""
        flops = self.calculate_flops(data)
        if flops is None or time_ms <= 0:
            return 0.0
        # 转换为GFLOPS (每秒十亿次浮点运算)
        gflops = (flops / (time_ms / 1000.0)) / 1e9
        return gflops
    
    def get_precision_config(self, precision: PrecisionType = None) -> Dict[str, Any]:
        """获取精度配置信息
        
        Args:
            precision: 精度类型，如果不指定则使用默认值
            
        Returns:
            Dict[str, Any]: 精度配置信息
        """
        if precision is None:
            precision = PrecisionType.BF16  # 默认使用BF16
            
        if precision == PrecisionType.FP16:
            return {
                'input_dtype_size': 2,   # FP16: 2字节
                'weight_dtype_size': 2,  # FP16: 2字节
                'output_dtype_size': 2,  # FP16: 2字节
                'bias_dtype_size': 2,    # FP16: 2字节
            }
        elif precision == PrecisionType.BF16:
            return {
                'input_dtype_size': 2,   # BF16: 2字节
                'weight_dtype_size': 2,  # BF16: 2字节
                'output_dtype_size': 2,  # BF16: 2字节
                'bias_dtype_size': 4,    # FP32: 4字节（通常bias用FP32）
            }
        else:
            # 默认配置
            return {
                'input_dtype_size': 2,
                'weight_dtype_size': 2,
                'output_dtype_size': 2,
                'bias_dtype_size': 4,
            }
    
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
            
        # 获取数据维度 - 从metadata中获取
        metadata = data.get('metadata', {})
        batch_size = metadata.get('batch_size', 0)
        input_dim = metadata.get('input_dim', 0)
        output_dim = metadata.get('output_dim', 0)
        
        if batch_size <= 0 or input_dim <= 0 or output_dim <= 0:
            return None
        
        # 动态获取精度配置
        precision_config = self.get_precision_config(precision)
        input_dtype_size = precision_config.get('input_dtype_size', 2)
        weight_dtype_size = precision_config.get('weight_dtype_size', 2)
        output_dtype_size = precision_config.get('output_dtype_size', 2)
        bias_dtype_size = precision_config.get('bias_dtype_size', 4)
        
        # 计算内存访问量（字节）
        # 输入读取：batch_size * input_dim * input_dtype_size
        # 权重读取：output_dim * input_dim * weight_dtype_size
        # 输出写入：batch_size * output_dim * output_dtype_size
        input_bytes = batch_size * input_dim * input_dtype_size
        weight_bytes = output_dim * input_dim * weight_dtype_size
        output_bytes = batch_size * output_dim * output_dtype_size
        
        # 如果有bias，也要计算bias的内存访问
        bias_bytes = 0
        if data.get('bias') is not None:
            bias_bytes = output_dim * bias_dtype_size
            
        total_bytes = input_bytes + weight_bytes + output_bytes + bias_bytes
        
        # 转换为GB/s：总字节数 / (时间_秒 * 10^9)
        avg_time_s = avg_time_ms / 1000.0
        bandwidth_gb_s = total_bytes / (avg_time_s * 1e9)
        
        return bandwidth_gb_s
