import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path to import framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from operator_test_framework import BaseOperatorTest, PrecisionType, DeviceType


class RMSNormOperatorTest(BaseOperatorTest):
    """RMSNorm算子测试实现"""

    CUDA_IMPLEMENTATION = "cuda_vllm_rms_norm_out"
    NPU_IMPLEMENTATION = "npu_torch_npu_rms_norm"
    
    def __init__(self):
        super().__init__("RMSNorm")
        self.supported_precisions = [PrecisionType.FP16, PrecisionType.BF16, PrecisionType.FP32]
        self.supported_devices = []
        if torch_npu is not None:
            self.supported_devices.append(DeviceType.NPU)
        if torch.cuda.is_available():
            self.supported_devices.append(DeviceType.GPU)

    @staticmethod
    def _primary_output(result):
        """Select the tensor used for correctness without changing timed returns."""
        return result[0] if isinstance(result, (tuple, list)) else result

    @staticmethod
    def _cuda_rms_norm_callable():
        try:
            from vllm import _custom_ops as vllm_ops
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(f"vLLM RMSNorm provider is unavailable: {exc}") from exc
        operator = getattr(vllm_ops, "rms_norm", None)
        if not callable(operator):
            raise RuntimeError("vllm._custom_ops.rms_norm is unavailable")
        return operator
    
    def generate_test_data(
        self,
        shape: tuple = (32, 2048),  # typical batch size and hidden size
        eps: float = 1e-6,
        value_range: tuple = (-1.0, 1.0),
        **kwargs
    ) -> Dict[str, Any]:
        """生成RMSNorm测试数据"""

        # input tensor
        x = torch.rand(shape) * (value_range[1] - value_range[0]) + value_range[0]
        
        # gamma (scale) parameter, usually initialized to 1s or random
        # The last dimension is the normalized dimension
        normalized_shape = shape[-1]
        gamma = torch.rand(normalized_shape) * (value_range[1] - value_range[0]) + value_range[0]
        
        return {
            'x': x,
            'gamma': gamma,
            'eps': eps,
            'metadata': {
                'shape': shape,
                'eps': eps,
                'operator_type': 'RMSNorm',
                'total_elements': x.numel(),
                # RMSNorm flops: per element: square, sum (reduction), sqrt, div, mul
                # approx 4 ops per element depending on implementation details
                'flops': x.numel() * 4 
            }
        }
    
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """运行CPU参考实现"""
        x = data['x']
        gamma = data['gamma']
        eps = data['eps']
        
        # Manual RMSNorm implementation
        # x: [..., d]
        # rms = sqrt(mean(x**2) + eps)
        # y = x / rms * gamma
        
        # Cast to float32 for higher precision reference calculation
        x_float = x.float()
        gamma_float = gamma.float()
        
        mean_square = torch.mean(x_float ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + eps)
        result = x_float * torch.rsqrt(mean_square + eps) * gamma_float
        
        return result
    
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
        result = self._execute_core_operator(prepared, implementation)
        return self._primary_output(result).cpu().float()

    def _prepare_data_for_core_operator(self, data: Dict[str, Any], device: str, precision: PrecisionType, implementation: str = "default") -> Dict[str, Any]:
        """为核心算子准备数据（排除预处理开销）"""
        implementation = self._resolve_implementation(device, implementation)
        x = data['x'].to(
            device=device, dtype=precision.value, copy=True
        )
        gamma = data['gamma'].to(
            device=device, dtype=precision.value, copy=True
        )

        prepared = {
            'x': x,
            'gamma': gamma,
            'eps': data['eps'],
            'implementation': implementation,
            'device': device
        }
        if implementation == self.CUDA_IMPLEMENTATION:
            prepared['operator'] = self._cuda_rms_norm_callable()
            prepared['output'] = torch.empty_like(x)
        elif implementation == self.NPU_IMPLEMENTATION:
            if (
                torch_npu is None
                or not hasattr(torch_npu, 'npu_rms_norm')
            ):
                raise RuntimeError("torch_npu.npu_rms_norm is unavailable")
        return prepared

    def _execute_core_operator(self, prepared_data: Dict[str, Any], implementation: str = "default") -> torch.Tensor:
        """执行核心算子（只测量核心计算，不包括数据移动）"""
        x = prepared_data['x']
        gamma = prepared_data['gamma']
        eps = prepared_data['eps']
        device = prepared_data.get('device', '')
        impl = prepared_data.get('implementation', implementation)
        
        if impl == self.CUDA_IMPLEMENTATION:
            prepared_data['operator'](
                prepared_data['output'],
                x,
                gamma,
                eps,
            )
            return prepared_data['output']
        if impl == self.NPU_IMPLEMENTATION:
            return torch_npu.npu_rms_norm(x, gamma, epsilon=eps)
        if impl != "manual":
            raise ValueError(f"不支持的 RMSNorm 实现: {impl}")
        
        # Manual implementation
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        result = x * torch.rsqrt(mean_square + eps) * gamma
        return result

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
                raise ValueError(f"RMSNorm 不支持设备 {device}")
            return formal[0]
        if implementation not in formal:
            raise ValueError(
                f"实现 {implementation!r} 不适用于 {device}; formal={formal}"
            )
        return implementation
    
    def calculate_throughput(self, data: Dict[str, Any], time_ms: float) -> Optional[float]:
        """计算吞吐量（GFLOPS）"""
        # Approx ops: square, mean(sum+div), add eps, sqrt, div, mul
        # Let's say 4 ops per element for simplicity
        flops = data['metadata']['flops']
        return (flops / (time_ms / 1000)) / 1e9  # GFLOPS
        
    def calculate_bandwidth(self, data: Dict[str, Any], avg_time_ms: float, precision: PrecisionType = None) -> Optional[float]:
        """计算内存带宽（GB/s）"""
        # Read x (once), Read gamma (broadcast, negligible for large batch), Write y (once)
        # Total bytes = 2 * num_elements * sizeof(dtype)
        
        num_elements = data['metadata']['total_elements']
        dtype_size = self.get_precision_config(precision)['dtype_size']
        
        total_bytes = 2 * num_elements * dtype_size
        
        return (total_bytes / (avg_time_ms / 1000)) / 1e9 # GB/s

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
