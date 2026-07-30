import gc
import statistics
from contextlib import nullcontext

import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None
import time
import math
import numpy as np
import os
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

class ProfilerBackend(Enum):
    """Profiler后端类型"""
    NPU = "npu"
    CUDA = "cuda"

@dataclass
class ProfilerConfig:
    """Profiler配置类"""
    backend: ProfilerBackend
    trace_file_path: Optional[str] = None
    kernel_names: Optional[List[str]] = None
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    experimental_config: Optional[Dict[str, Any]] = None
    # Schedule 配置
    schedule_wait: int = 0
    schedule_warmup: int = 0
    schedule_active: int = 20
    schedule_repeat: int = 1
    schedule_skip_first: int = 1

class ProfilerFactory:
    """Profiler工厂类 - 根据后端类型创建相应的profiler"""
    
    @staticmethod
    def create_profiler(config: ProfilerConfig):
        """创建profiler实例 - 简化版本，参考run_unified_profile_test的实现
        
        Args:
            config: Profiler配置
            
        Returns:
            profiler实例
        """
        if config.backend == ProfilerBackend.NPU:
            return ProfilerFactory._create_npu_profiler(config)
        elif config.backend == ProfilerBackend.CUDA:
            return ProfilerFactory._create_cuda_profiler(config)
        else:
            raise ValueError(f"不支持的profiler后端: {config.backend}")
    
    @staticmethod
    def _create_npu_profiler(config: ProfilerConfig):
        """创建NPU profiler - 完整版本"""
        try:
            import torch_npu
            
            # 处理实验性配置
            exp_config = config.experimental_config or {}
            profile_level = exp_config.get('profile_level', 'Level1')
            aic_metrics = exp_config.get('aic_metrics', 'PipeUtilization')
            export_type = exp_config.get('export_type', 'Text')
            
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=getattr(torch_npu.profiler.ProfilerLevel, profile_level, torch_npu.profiler.ProfilerLevel.Level1),
                aic_metrics=getattr(torch_npu.profiler.AiCMetrics, aic_metrics, torch_npu.profiler.AiCMetrics.PipeUtilization),
                export_type=getattr(torch_npu.profiler.ExportType, export_type, torch_npu.profiler.ExportType.Text)
            )
            
            profiler_kwargs = {
                'activities': [
                    torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU
                ],
                'with_stack': config.with_stack,
                'record_shapes': config.record_shapes,
                'profile_memory': config.profile_memory,
                'experimental_config': experimental_config,
                'schedule': torch_npu.profiler.schedule(
                    wait=config.schedule_wait,
                    warmup=config.schedule_warmup,
                    active=config.schedule_active,
                    repeat=config.schedule_repeat,
                    skip_first=config.schedule_skip_first
                )
            }
            
            # 如果有trace文件路径，添加处理器
            if config.trace_file_path:
                profiler_kwargs['on_trace_ready'] = torch_npu.profiler.tensorboard_trace_handler(
                    config.trace_file_path, 
                )
            
            return torch_npu.profiler.profile(**profiler_kwargs)
            
        except ImportError:
            raise ImportError("torch_npu未安装，无法使用NPU profiler")
    
    @staticmethod
    def _create_cuda_profiler(config: ProfilerConfig):
        """创建CUDA profiler - 完整版本"""
        try:
            profiler_kwargs = {
                'activities': [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                'record_shapes': config.record_shapes,
                'profile_memory': config.profile_memory,
                'with_stack': config.with_stack,
                'schedule': torch.profiler.schedule(
                    wait=config.schedule_wait,
                    warmup=config.schedule_warmup,
                    active=config.schedule_active,
                    repeat=config.schedule_repeat
                )
            }
            
            # 如果有trace文件路径，添加处理器
            if config.trace_file_path:
                profiler_kwargs['on_trace_ready'] = torch.profiler.tensorboard_trace_handler(
                    config.trace_file_path
                )
            
            return torch.profiler.profile(**profiler_kwargs)
            
        except Exception as e:
            raise RuntimeError(f"创建CUDA profiler失败: {e}")
    

class PrecisionType(Enum):
    """支持的精度类型"""
    FP16 = torch.float16
    BF16 = torch.bfloat16
    FP32 = torch.float32
    FP8 = torch.float8_e4m3fn
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    INT8 = torch.int8

class DeviceType(Enum):
    """支持的设备类型"""
    CPU = "cpu"
    NPU = "npu"
    GPU = "cuda"

@dataclass
class AccuracyMetrics:
    """精度测试指标"""
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    cosine_similarity: float
    mse: float
    rmse: float
    precision_type: str
    operator_name: str
    
    def __str__(self):
        return (f"精度指标 ({self.operator_name} - {self.precision_type}):\n"
                f"  最大绝对误差: {self.max_abs_error:.6e}\n"
                f"  平均绝对误差: {self.mean_abs_error:.6e}\n"
                f"  最大相对误差: {self.max_rel_error:.6e}\n"
                f"  平均相对误差: {self.mean_rel_error:.6e}\n"
                f"  余弦相似度: {self.cosine_similarity:.6f}\n"
                f"  MSE: {self.mse:.6e}\n"
                f"  RMSE: {self.rmse:.6e}")


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    avg_time_ms: float
    throughput: Optional[float]  # 可选的吞吐量指标
    precision_type: str
    device_type: str
    operator_name: str
    iterations: int = 20  # 测试迭代次数
    throughput_ops_per_sec: Optional[float] = None  # 每秒操作数吞吐量
    tops: Optional[float] = None  # TOPS (每秒万亿次操作)
    bandwidth_gb_s: Optional[float] = None  # 内存带宽 (GB/s)
    framework_api: Optional[str] = None
    warmup_iterations: int = 0
    preallocated_input_sets: int = 0
    independent_storage_sets_verified: int = 0
    independent_output_storage_sets_verified: int = 0
    preallocated_output_aliases_verified: int = 0
    output_allocation_mode: Optional[str] = None
    output_storage_policy: Optional[str] = None
    protocol_version: Optional[str] = None
    repeats: int = 1
    repeat_samples_ms: List[float] = field(default_factory=list)
    aggregation: str = "single_repeat"
    preallocated_invocations_per_repeat: int = 0
    input_storage_sets_verified: int = 0
    input_storage_ptr_count: int = 0
    output_storage_sets_verified: int = 0
    output_storage_ptr_count: int = 0
    output_tensor_count: int = 0
    input_reuse_within_repeat: bool = False
    timing_method: Optional[str] = None
    timing_semantics: Optional[str] = None
    timed_output_capture_policy: Optional[str] = None
    preallocated_output_contract: Optional[str] = None
    output_alias_verification_scope: Optional[str] = None
    preallocated_output_contract_invocations_per_repeat: int = 0
    output_verification_replay_invocations_per_repeat: int = 0
    total_operator_calls_per_repeat: int = 0
    workspace_allocation_policy: Optional[str] = None
    dispatch_loop_policy: Optional[str] = None
    device_stabilization_policy: Optional[str] = None
    device_stabilization_timed: bool = False
    task_queue_enable: Optional[str] = None
    timed_region: Optional[str] = None
    # V5 fields are appended so legacy positional construction keeps the
    # exact V4 field order. Framework code constructs metrics by keyword.
    stabilization_repeats: int = 0
    stabilization_repeat_samples_ms: List[float] = field(
        default_factory=list
    )
    input_output_storage_disjoint: bool = False
    stabilization_operator_calls: int = 0
    dispatch_mode: str = "eager_direct"
    graph_capture_width: int = 0
    graph_replays: int = 0
    capture_timed: bool = False
    mutable_inputs_restored: bool = False
    profiler_is_diagnostic: bool = False
    
    def __post_init__(self):
        """初始化后处理，确保throughput_ops_per_sec有值"""
        if self.throughput_ops_per_sec is None and self.throughput is not None:
            self.throughput_ops_per_sec = self.throughput
    
    def __str__(self):
        result = (f"性能指标 ({self.operator_name} - {self.device_type} - {self.precision_type}):\n"
                 f"  平均时间: {self.avg_time_ms:.3f}ms (批量执行模式)\n")
        
        if self.tops is not None:
            result += f"  计算性能: {self.tops:.2f} TOPS\n"
        
        if self.bandwidth_gb_s is not None:
            result += f"  内存带宽: {self.bandwidth_gb_s:.2f} GB/s\n"
        
        if self.throughput is not None:
            result += f"  吞吐量: {self.throughput:.2f} ops/s\n"
        
        return result


@dataclass
class _CapturedPreparedChain:
    replay: Callable[[], None]
    retained_outputs: List[Any]
    logical_invocations: int


PERFORMANCE_PROVENANCE_FIELDS = (
    "framework_api",
    "protocol_version",
    "warmup",
    "iterations",
    "repeats",
    "stabilization_repeats",
    "stabilization_repeat_samples_ms",
    "repeat_samples_ms",
    "event_window_samples_ms",
    "event_window_min_ms",
    "event_window_median_ms",
    "event_window_max_ms",
    "repeat_min_ms",
    "repeat_median_ms",
    "repeat_max_ms",
    "repeat_p25_ms",
    "repeat_p75_ms",
    "repeat_iqr_pct",
    "repeat_spread_pct",
    "aggregation",
    "preallocated_invocations_per_repeat",
    "input_reuse_within_repeat",
    "input_storage_sets_verified",
    "input_storage_ptr_count",
    "input_output_storage_disjoint",
    "output_storage_sets_verified",
    "output_storage_ptr_count",
    "output_tensor_count",
    "output_unique_storages_per_set",
    "preallocated_output_aliases_verified",
    "preallocated_output_sets_verified",
    "output_tensors_per_set",
    "output_allocation_mode",
    "output_allocation_policy",
    "output_storage_policy",
    "timing_method",
    "timing_semantics",
    "timed_output_capture_policy",
    "preallocated_output_contract",
    "output_alias_verification_scope",
    "preallocated_output_contract_invocations_per_repeat",
    "output_verification_replay_invocations_per_repeat",
    "total_operator_calls_per_repeat",
    "workspace_allocation_policy",
    "dispatch_loop_policy",
    "device_stabilization_policy",
    "device_stabilization_timed",
    "stabilization_operator_calls",
    "task_queue_enable",
    "timed_region",
    "dispatch_mode",
    "graph_capture_width",
    "graph_replays",
    "capture_timed",
    "mutable_inputs_restored",
    "profiler_is_diagnostic",
)

FRESH_ITERATION_PLAN_FIELDS = (
    "iteration_selection_policy",
    "requested_iterations",
    "base_iterations",
    "effective_iterations",
    "adaptive_capacity_iterations",
    "adaptive_iterations_cap",
    "estimated_unique_bytes_per_invocation",
    "fresh_storage_soft_target_bytes",
    "estimated_fresh_storage_bytes_per_repeat",
    "fresh_storage_soft_target_overflow",
    "requested_warmup",
    "effective_warmup",
    "minimum_warmup",
    "minimum_iterations",
    "fresh_storage_hard_limit_bytes",
    "fresh_storage_hard_limit_overflow",
)

DEFAULT_FRESH_STORAGE_SOFT_TARGET_BYTES = 4 * 1024**3
DEFAULT_ADAPTIVE_ITERATIONS_CAP = 2048


def build_fresh_iteration_plan(
    *,
    num_warmup: int,
    requested_iterations: Optional[int],
    base_iterations: int,
    estimated_unique_bytes_per_invocation: int,
    fresh_storage_soft_target_bytes: int = (
        DEFAULT_FRESH_STORAGE_SOFT_TARGET_BYTES
    ),
    adaptive_iterations_cap: int = DEFAULT_ADAPTIVE_ITERATIONS_CAP,
) -> Dict[str, Any]:
    """Choose one cross-provider fresh-address iteration count.

    The storage target is deliberately soft: the historical base iteration
    count is never reduced, even when that base footprint exceeds the target.
    Callers must use a provider-independent peak-retained byte estimate so the
    same shape receives the same iteration count on every device.
    """
    integer_arguments = {
        "num_warmup": (num_warmup, 0),
        "base_iterations": (base_iterations, 1),
        "estimated_unique_bytes_per_invocation": (
            estimated_unique_bytes_per_invocation,
            1,
        ),
        "fresh_storage_soft_target_bytes": (
            fresh_storage_soft_target_bytes,
            1,
        ),
        "adaptive_iterations_cap": (adaptive_iterations_cap, 1),
    }
    for name, (value, minimum) in integer_arguments.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be a non-bool int")
        if value < minimum:
            comparator = ">=" if minimum == 0 else ">"
            threshold = minimum if minimum == 0 else minimum - 1
            raise ValueError(f"{name} must be {comparator} {threshold}")
    if adaptive_iterations_cap < base_iterations:
        raise ValueError(
            "adaptive_iterations_cap must be >= base_iterations"
        )
    if requested_iterations is not None:
        if (
            not isinstance(requested_iterations, int)
            or isinstance(requested_iterations, bool)
            or requested_iterations <= 0
        ):
            raise ValueError(
                "requested_iterations must be None or a positive non-bool int"
            )
        effective_iterations = requested_iterations
        selection_policy = "explicit_fixed"
    else:
        capacity_sets = (
            fresh_storage_soft_target_bytes
            // estimated_unique_bytes_per_invocation
        )
        adaptive_capacity_iterations = max(
            0,
            capacity_sets - num_warmup,
        )
        effective_iterations = max(
            base_iterations,
            min(
                adaptive_iterations_cap,
                adaptive_capacity_iterations,
            ),
        )
        selection_policy = "adaptive_unique_storage_soft_target"

    capacity_sets = (
        fresh_storage_soft_target_bytes
        // estimated_unique_bytes_per_invocation
    )
    adaptive_capacity_iterations = max(0, capacity_sets - num_warmup)
    estimated_total_bytes = (
        num_warmup + effective_iterations
    ) * estimated_unique_bytes_per_invocation
    return {
        "iteration_selection_policy": selection_policy,
        "requested_iterations": (
            requested_iterations
            if requested_iterations is not None
            else "auto"
        ),
        "base_iterations": base_iterations,
        "effective_iterations": effective_iterations,
        "adaptive_capacity_iterations": adaptive_capacity_iterations,
        "adaptive_iterations_cap": adaptive_iterations_cap,
        "estimated_unique_bytes_per_invocation": (
            estimated_unique_bytes_per_invocation
        ),
        "fresh_storage_soft_target_bytes": (
            fresh_storage_soft_target_bytes
        ),
        "estimated_fresh_storage_bytes_per_repeat": (
            estimated_total_bytes
        ),
        "fresh_storage_soft_target_overflow": (
            estimated_total_bytes > fresh_storage_soft_target_bytes
        ),
    }


def build_memory_bounded_fresh_invocation_plan(
    *,
    requested_warmup: int,
    requested_iterations: Optional[int],
    base_iterations: int,
    estimated_unique_bytes_per_invocation: int,
    fresh_storage_hard_limit_bytes: int,
    minimum_warmup: int = 2,
    minimum_iterations: int = 1,
) -> Dict[str, Any]:
    """Choose a fresh-address invocation plan within a hard byte limit."""
    integer_arguments = {
        "requested_warmup": requested_warmup,
        "base_iterations": base_iterations,
        "estimated_unique_bytes_per_invocation": (
            estimated_unique_bytes_per_invocation
        ),
        "fresh_storage_hard_limit_bytes": (
            fresh_storage_hard_limit_bytes
        ),
        "minimum_warmup": minimum_warmup,
        "minimum_iterations": minimum_iterations,
    }
    for name, value in integer_arguments.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be a positive non-bool int")
        if value <= 0:
            raise ValueError(f"{name} must be a positive non-bool int")
    if requested_iterations is not None and (
        not isinstance(requested_iterations, int)
        or isinstance(requested_iterations, bool)
        or requested_iterations <= 0
    ):
        raise ValueError(
            "requested_iterations must be None or a positive non-bool int"
        )

    capacity = (
        fresh_storage_hard_limit_bytes
        // estimated_unique_bytes_per_invocation
    )
    if requested_iterations is None:
        total = min(requested_warmup + base_iterations, capacity)
        effective_warmup = min(
            requested_warmup,
            max(minimum_warmup, total // 5),
        )
        effective_iterations = total - effective_warmup
        selection_policy = "adaptive_unique_storage_hard_limit"
    else:
        effective_warmup = requested_warmup
        effective_iterations = requested_iterations
        selection_policy = "explicit_fixed"

    estimated_total_bytes = (
        effective_warmup + effective_iterations
    ) * estimated_unique_bytes_per_invocation
    if (
        effective_warmup < minimum_warmup
        or effective_iterations < minimum_iterations
        or estimated_total_bytes > fresh_storage_hard_limit_bytes
    ):
        raise ValueError("fresh-storage hard limit cannot satisfy protocol")

    adaptive_capacity_iterations = max(
        0,
        capacity - effective_warmup,
    )
    return {
        "iteration_selection_policy": selection_policy,
        "requested_iterations": (
            requested_iterations
            if requested_iterations is not None
            else "auto"
        ),
        "base_iterations": base_iterations,
        "effective_iterations": effective_iterations,
        "adaptive_capacity_iterations": adaptive_capacity_iterations,
        "adaptive_iterations_cap": base_iterations,
        "estimated_unique_bytes_per_invocation": (
            estimated_unique_bytes_per_invocation
        ),
        "fresh_storage_soft_target_bytes": (
            fresh_storage_hard_limit_bytes
        ),
        "estimated_fresh_storage_bytes_per_repeat": (
            estimated_total_bytes
        ),
        "fresh_storage_soft_target_overflow": False,
        "requested_warmup": requested_warmup,
        "effective_warmup": effective_warmup,
        "minimum_warmup": minimum_warmup,
        "minimum_iterations": minimum_iterations,
        "fresh_storage_hard_limit_bytes": (
            fresh_storage_hard_limit_bytes
        ),
        "fresh_storage_hard_limit_overflow": False,
    }


def build_curve_selection_provenance(
    *,
    quick: bool,
    num_shards: int,
    total_formal_points: int,
    total_requested_points: int,
    selected_points: int,
    uses_formal_shape_matrix: bool,
    providers_complete: bool = True,
) -> Dict[str, Any]:
    """Describe shape selection without claiming measurements succeeded."""
    selection_covers_full_formal_matrix = (
        not quick
        and num_shards == 1
        and uses_formal_shape_matrix
        and providers_complete
        and selected_points == total_formal_points
    )
    if quick:
        selection_mode = "quick_shape_subset"
    elif selection_covers_full_formal_matrix:
        selection_mode = "full_formal_shape_matrix"
    elif uses_formal_shape_matrix and num_shards > 1:
        selection_mode = "formal_shape_shard"
    elif not providers_complete:
        selection_mode = "provider_subset"
    else:
        selection_mode = "custom_shape_matrix"
    return {
        "selection_mode": selection_mode,
        "shape_matrix_source": (
            "default_formal" if uses_formal_shape_matrix else "custom"
        ),
        "coverage_mode": (
            "sharded" if num_shards > 1 else "single_process"
        ),
        "coverage_total_formal_points": total_formal_points,
        "coverage_total_requested_points": total_requested_points,
        "coverage_selected_points": selected_points,
        "selection_covers_full_formal_matrix": (
            selection_covers_full_formal_matrix
        ),
        # This is deliberately false in checkpoints and becomes true only
        # after every canonical formal point has completed successfully.
        "coverage_complete": False,
    }


def finalize_curve_coverage(
    rows: List[Dict[str, Any]],
    *,
    success_statuses: Tuple[str, ...] = ("ok", "success"),
) -> bool:
    """Mark a curve complete only after all canonical points succeeded."""
    complete = bool(rows)
    if complete:
        total_formal_points = int(rows[0]["coverage_total_formal_points"])
        point_indices = [int(row["point_index"]) for row in rows]
        complete = (
            all(
                bool(row["selection_covers_full_formal_matrix"])
                for row in rows
            )
            and len(rows) == total_formal_points
            and len(set(point_indices)) == total_formal_points
            and all(row.get("status") in success_statuses for row in rows)
        )
    for row in rows:
        row["coverage_complete"] = complete
    return complete


class BaseOperatorTest(ABC):
    """算子测试基类"""
    
    def __init__(self, operator_name: str):
        self.operator_name = operator_name
        self.supported_precisions = [PrecisionType.FP16, PrecisionType.BF16, PrecisionType.FP32]
        self.supported_devices = [DeviceType.CPU, DeviceType.NPU]
    
    @abstractmethod
    def generate_test_data(self, **kwargs) -> Dict[str, Any]:
        """生成测试数据"""
        pass
    
    @abstractmethod
    def run_cpu_reference(self, data: Dict[str, Any]) -> torch.Tensor:
        """运行CPU参考实现"""
        pass
    
    @abstractmethod
    def run_device_implementation(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行设备实现"""
        pass
    
    @abstractmethod
    def get_available_implementations(self, device: str) -> List[str]:
        """获取可用的实现方式"""
        pass
    
    def run_core_operator(
        self, 
        data: Dict[str, Any], 
        device: str, 
        precision: PrecisionType,
        implementation: str = "default"
    ) -> torch.Tensor:
        """运行核心算子操作（用于精确性能测试）
        
        默认实现调用run_device_implementation，子类可以重写此方法
        来只执行核心算子操作，排除数据预处理和后处理的时间开销
        """
        return self.run_device_implementation(data, device, precision, implementation)

    def _declares_preallocated_output_contract(
        self,
        prepared_data: Dict[str, Any],
        implementation: str = "default",
    ) -> bool:
        """Declare a phase-invariant ``out=`` contract for direct timing.

        Subclasses may return ``True`` only when every invocation of
        ``_execute_core_operator`` with this prepared payload writes to and
        returns the explicitly designated output buffers.  The framework may
        then discard timed Python return objects and verify actual aliases on
        untimed warmup calls.
        """
        del prepared_data, implementation
        return False

    def _restore_mutable_graph_inputs(
        self,
        prepared_payloads: List[Any],
        implementation: str = "default",
    ) -> None:
        """Restore mutable captured inputs before graph replay, if needed."""
        del prepared_payloads, implementation

    
    def calculate_throughput(self, data: Dict[str, Any], time_ms: float) -> Optional[float]:
        """计算吞吐量
        
        Args:
            data: 测试数据
            time_ms: 执行时间（毫秒）
            
        Returns:
            Optional[float]: 吞吐量，如果无法计算则返回None
        """
        # 默认实现返回None，子类应该重写此方法
        return None
    

class OperatorTestFramework:
    """算子测试框架"""

    _PREPARED_OUTPUT_KEYS = frozenset({
        "out",
        "output",
        "outputs",
        "output_buffers",
        "expert_outputs",
    })

    def __init__(self, result_dir: str = "test_results"):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        self.operators: Dict[str, BaseOperatorTest] = {}

    @staticmethod
    def _device_storage_ptrs(value: Any, device_type: str) -> set[int]:
        """Collect device storage identities from a nested prepared payload."""
        pointers: set[int] = set()
        if isinstance(value, torch.Tensor):
            if value.numel() and value.device.type == device_type:
                try:
                    pointers.add(value.untyped_storage().data_ptr())
                except (AttributeError, RuntimeError):
                    pointers.add(value.data_ptr())
            return pointers
        if isinstance(value, dict):
            for child in value.values():
                pointers.update(
                    OperatorTestFramework._device_storage_ptrs(
                        child, device_type
                    )
                )
            return pointers
        if isinstance(value, (list, tuple)):
            for child in value:
                pointers.update(
                    OperatorTestFramework._device_storage_ptrs(
                        child, device_type
                    )
                )
        return pointers

    @staticmethod
    def _device_tensor_count(value: Any, device_type: str) -> int:
        """Count tensors, including empty tensors and storage-sharing views."""
        if isinstance(value, torch.Tensor):
            return int(value.device.type == device_type)
        if isinstance(value, dict):
            return sum(
                OperatorTestFramework._device_tensor_count(
                    child, device_type
                )
                for child in value.values()
            )
        if isinstance(value, (list, tuple)):
            return sum(
                OperatorTestFramework._device_tensor_count(
                    child, device_type
                )
                for child in value
            )
        return 0

    @classmethod
    def _prepared_output_values(
        cls,
        prepared: Any,
    ) -> List[Any]:
        """Collect values explicitly designated as prepared outputs."""
        outputs: List[Any] = []
        if isinstance(prepared, dict):
            for key, value in prepared.items():
                if key in cls._PREPARED_OUTPUT_KEYS:
                    outputs.append(value)
                else:
                    outputs.extend(
                        cls._prepared_output_values(value)
                    )
        elif isinstance(prepared, (list, tuple)):
            for value in prepared:
                outputs.extend(
                    cls._prepared_output_values(value)
                )
        return outputs

    @classmethod
    def _prepared_input_values(cls, prepared: Any) -> Any:
        """Return a nested view with explicitly designated outputs removed."""
        if isinstance(prepared, dict):
            return {
                key: cls._prepared_input_values(value)
                for key, value in prepared.items()
                if key not in cls._PREPARED_OUTPUT_KEYS
            }
        if isinstance(prepared, list):
            return [
                cls._prepared_input_values(value)
                for value in prepared
            ]
        if isinstance(prepared, tuple):
            return tuple(
                cls._prepared_input_values(value)
                for value in prepared
            )
        return prepared

    @classmethod
    def _prepared_output_storage_ptrs(
        cls,
        prepared: Any,
        device_type: str,
    ) -> set[int]:
        """Collect only buffers explicitly designated as prepared outputs."""
        pointers: set[int] = set()
        for value in cls._prepared_output_values(prepared):
            pointers.update(cls._device_storage_ptrs(value, device_type))
        return pointers

    @classmethod
    def _verify_independent_storage_sets(
        cls,
        values: List[Any],
        device: str,
        label: str,
    ) -> Tuple[int, int]:
        """Fail when two V2 invocations reuse any device tensor storage."""
        device_type = "npu" if "npu" in device else (
            "cuda" if "cuda" in device else "cpu"
        )
        seen: set[int] = set()
        verified = 0
        for index, value in enumerate(values):
            current = cls._device_storage_ptrs(value, device_type)
            if not current:
                raise RuntimeError(
                    f"{label} {index} contains no non-empty {device_type} tensors"
                )
            overlap = seen.intersection(current)
            if overlap:
                raise RuntimeError(
                    f"{label} reuse device storage at invocation {index}; "
                    f"overlap_count={len(overlap)}"
                )
            seen.update(current)
            verified += 1
        return verified, len(seen)

    @classmethod
    def _verify_disjoint_storage_domains(
        cls,
        left_values: Any,
        right_values: Any,
        device: str,
        label: str,
    ) -> None:
        """Fail when two prepared/retained storage domains overlap."""
        device_type = "npu" if "npu" in device else (
            "cuda" if "cuda" in device else "cpu"
        )
        left_ptrs = cls._device_storage_ptrs(
            left_values,
            device_type,
        )
        right_ptrs = cls._device_storage_ptrs(
            right_values,
            device_type,
        )
        if not left_ptrs or not right_ptrs:
            raise RuntimeError(
                f"{label} requires non-empty {device_type} storage domains"
            )
        overlap = left_ptrs.intersection(right_ptrs)
        if overlap:
            raise RuntimeError(
                f"{label} overlap_count={len(overlap)}"
            )

    @classmethod
    def _count_preallocated_output_aliases(
        cls,
        prepared_values: List[Any],
        outputs: List[Any],
        device: str,
    ) -> int:
        """Count calls returning explicitly designated output buffers."""
        if len(prepared_values) != len(outputs):
            raise RuntimeError(
                "prepared/output count mismatch: "
                f"{len(prepared_values)} != {len(outputs)}"
            )

        device_type = "npu" if "npu" in device else (
            "cuda" if "cuda" in device else "cpu"
        )
        verified = 0
        for prepared, output in zip(prepared_values, outputs):
            prepared_output_ptrs = cls._prepared_output_storage_ptrs(
                prepared, device_type
            )
            output_ptrs = cls._device_storage_ptrs(output, device_type)
            if (
                output_ptrs
                and output_ptrs.issubset(prepared_output_ptrs)
            ):
                verified += 1
        return verified

    @staticmethod
    def performance_provenance_fields() -> List[str]:
        """Return the one canonical CSV schema for V2 protocol evidence."""
        return list(PERFORMANCE_PROVENANCE_FIELDS)

    @staticmethod
    def performance_provenance(metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Return the stable, flat protocol fields written to curve CSVs."""
        repeat_min_ms = (
            min(metrics.repeat_samples_ms)
            if metrics.repeat_samples_ms else None
        )
        repeat_median_ms = (
            statistics.median(metrics.repeat_samples_ms)
            if metrics.repeat_samples_ms else None
        )
        repeat_max_ms = (
            max(metrics.repeat_samples_ms)
            if metrics.repeat_samples_ms else None
        )
        if len(metrics.repeat_samples_ms) >= 2:
            repeat_p25_ms, _, repeat_p75_ms = statistics.quantiles(
                metrics.repeat_samples_ms,
                n=4,
                method="inclusive",
            )
        elif metrics.repeat_samples_ms:
            repeat_p25_ms = metrics.repeat_samples_ms[0]
            repeat_p75_ms = metrics.repeat_samples_ms[0]
        else:
            repeat_p25_ms = None
            repeat_p75_ms = None
        event_window_samples_ms = [
            sample * metrics.iterations
            for sample in metrics.repeat_samples_ms
        ]
        event_window_min_ms = (
            min(event_window_samples_ms)
            if event_window_samples_ms else None
        )
        event_window_median_ms = (
            statistics.median(event_window_samples_ms)
            if event_window_samples_ms else None
        )
        event_window_max_ms = (
            max(event_window_samples_ms)
            if event_window_samples_ms else None
        )
        repeat_spread_pct = (
            (repeat_max_ms / repeat_min_ms - 1.0) * 100.0
            if repeat_min_ms is not None
            and repeat_max_ms is not None
            and repeat_min_ms > 0
            else None
        )
        repeat_iqr_pct = (
            (repeat_p75_ms - repeat_p25_ms)
            / repeat_median_ms
            * 100.0
            if repeat_p25_ms is not None
            and repeat_p75_ms is not None
            and repeat_median_ms is not None
            and repeat_median_ms > 0
            else None
        )
        output_tensors_per_set = (
            metrics.output_tensor_count
            // metrics.output_storage_sets_verified
            if metrics.output_storage_sets_verified > 0
            and metrics.output_tensor_count
            % metrics.output_storage_sets_verified == 0
            else None
        )
        output_unique_storages_per_set = (
            metrics.output_storage_ptr_count
            // metrics.output_storage_sets_verified
            if metrics.output_storage_sets_verified > 0
            and metrics.output_storage_ptr_count
            % metrics.output_storage_sets_verified == 0
            else None
        )
        return {
            "framework_api": metrics.framework_api,
            "protocol_version": metrics.protocol_version,
            "warmup": metrics.warmup_iterations,
            "iterations": metrics.iterations,
            "repeats": metrics.repeats,
            "stabilization_repeats": metrics.stabilization_repeats,
            "stabilization_repeat_samples_ms": json.dumps(
                metrics.stabilization_repeat_samples_ms
            ),
            "repeat_samples_ms": json.dumps(metrics.repeat_samples_ms),
            "event_window_samples_ms": json.dumps(
                event_window_samples_ms
            ),
            "event_window_min_ms": event_window_min_ms,
            "event_window_median_ms": event_window_median_ms,
            "event_window_max_ms": event_window_max_ms,
            "repeat_min_ms": repeat_min_ms,
            "repeat_median_ms": repeat_median_ms,
            "repeat_max_ms": repeat_max_ms,
            "repeat_p25_ms": repeat_p25_ms,
            "repeat_p75_ms": repeat_p75_ms,
            "repeat_iqr_pct": repeat_iqr_pct,
            "repeat_spread_pct": repeat_spread_pct,
            "aggregation": metrics.aggregation,
            "preallocated_invocations_per_repeat": (
                metrics.preallocated_invocations_per_repeat
            ),
            "input_reuse_within_repeat": metrics.input_reuse_within_repeat,
            "input_storage_sets_verified": (
                metrics.input_storage_sets_verified
            ),
            "input_storage_ptr_count": metrics.input_storage_ptr_count,
            "input_output_storage_disjoint": (
                metrics.input_output_storage_disjoint
            ),
            "output_storage_sets_verified": (
                metrics.output_storage_sets_verified
            ),
            "output_storage_ptr_count": metrics.output_storage_ptr_count,
            "output_tensor_count": metrics.output_tensor_count,
            "output_unique_storages_per_set": (
                output_unique_storages_per_set
            ),
            "preallocated_output_aliases_verified": (
                metrics.preallocated_output_aliases_verified
            ),
            "preallocated_output_sets_verified": (
                metrics.preallocated_output_aliases_verified
            ),
            "output_tensors_per_set": output_tensors_per_set,
            "output_allocation_mode": metrics.output_allocation_mode,
            "output_allocation_policy": metrics.output_allocation_mode,
            "output_storage_policy": metrics.output_storage_policy,
            "timing_method": metrics.timing_method,
            "timing_semantics": metrics.timing_semantics,
            "timed_output_capture_policy": (
                metrics.timed_output_capture_policy
            ),
            "preallocated_output_contract": (
                metrics.preallocated_output_contract
            ),
            "output_alias_verification_scope": (
                metrics.output_alias_verification_scope
            ),
            "preallocated_output_contract_invocations_per_repeat": (
                metrics
                .preallocated_output_contract_invocations_per_repeat
            ),
            "output_verification_replay_invocations_per_repeat": (
                metrics.output_verification_replay_invocations_per_repeat
            ),
            "total_operator_calls_per_repeat": (
                metrics.total_operator_calls_per_repeat
            ),
            "workspace_allocation_policy": (
                metrics.workspace_allocation_policy
            ),
            "dispatch_loop_policy": metrics.dispatch_loop_policy,
            "device_stabilization_policy": (
                metrics.device_stabilization_policy
            ),
            "device_stabilization_timed": (
                metrics.device_stabilization_timed
            ),
            "stabilization_operator_calls": (
                metrics.stabilization_operator_calls
            ),
            "task_queue_enable": metrics.task_queue_enable,
            "timed_region": metrics.timed_region,
            "dispatch_mode": metrics.dispatch_mode,
            "graph_capture_width": metrics.graph_capture_width,
            "graph_replays": metrics.graph_replays,
            "capture_timed": metrics.capture_timed,
            "mutable_inputs_restored": metrics.mutable_inputs_restored,
            "profiler_is_diagnostic": metrics.profiler_is_diagnostic,
        }
    
    def _measure_execution_time(self, func, device: str, num_iterations: int = 1) -> float:
        """使用设备事件精确测量执行时间
        
        Args:
            func: 要测量的函数
            device: 设备类型
            num_iterations: 测量次数
            
        Returns:
            List[float]: 每次执行的时间（毫秒）
        """
        times = []
        
        try:
            if "npu" in device:
                # NPU事件计时 - 批量执行版本
                start_event = torch_npu.npu.Event(enable_timing=True)
                end_event = torch_npu.npu.Event(enable_timing=True)
                
                # 初始同步确保设备就绪
                torch_npu.npu.synchronize()
                
                # 记录开始时间
                start_event.record()
                
                # 执行所有迭代
                for i in range(num_iterations):
                    func()
                
                # 记录结束时间
                end_event.record()
                
                # 等待结束事件完成并计算总时间
                end_event.synchronize()
                total_time = start_event.elapsed_time(end_event)  # 毫秒
                
                # 计算平均时间
                avg_time = total_time / num_iterations
                return avg_time
                    
            elif "cuda" in device:
                # CUDA事件计时 - 批量执行版本
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # 初始同步确保设备就绪
                torch.cuda.synchronize()
                
                # 记录开始时间
                start_event.record()
                
                # 执行所有迭代
                for i in range(num_iterations):
                    func()
                
                # 记录结束时间
                end_event.record()
                
                # 等待结束事件完成并计算总时间
                end_event.synchronize()
                total_time = start_event.elapsed_time(end_event)  # 毫秒
                
                # 计算平均时间
                avg_time = total_time / num_iterations
                return avg_time
                    
            else:
                # CPU计时 - 批量执行版本
                import time as time_module
                
                # 记录开始时间
                start_time = time_module.perf_counter()
                
                # 执行所有迭代
                for i in range(num_iterations):
                    func()
                
                # 记录结束时间
                end_time = time_module.perf_counter()
                
                # 计算总时间和平均时间
                total_time = (end_time - start_time) * 1000  # 转换为毫秒
                avg_time = total_time / num_iterations
                return avg_time
                        
        except Exception as e:
            print(f"    ❌ 计时过程中发生错误: {str(e)}")
            raise
    
    @staticmethod
    def _dispatch_prepared_payloads(
        prepared_payloads: List[Any],
        execute_core_operator: Callable[[Any, str], Any],
        implementation: str,
        retained_outputs: Optional[List[Any]],
    ) -> None:
        """Launch prepared payloads without one Python closure per payload."""
        if retained_outputs is None:
            for payload in prepared_payloads:
                execute_core_operator(payload, implementation)
            return
        output_slot_offset = (
            len(retained_outputs) - len(prepared_payloads)
        )
        if output_slot_offset < 0:
            raise RuntimeError(
                "retained output slots must be preallocated before timing"
            )
        for output_index, payload in enumerate(prepared_payloads):
            retained_outputs[output_slot_offset + output_index] = (
                execute_core_operator(payload, implementation)
            )

    def _measure_execution_time_v2(
        self,
        prepared_payloads: List[Any],
        execute_core_operator: Callable[[Any, str], Any],
        implementation: str,
        device: str,
        retained_outputs: Optional[List[Any]] = None,
    ) -> float:
        """Measure one direct loop over already-prepared payloads."""
        num_iterations = len(prepared_payloads)
        if num_iterations <= 0:
            raise ValueError("prepared_payloads must not be empty")

        try:
            if "npu" in device:
                with torch_npu.npu.device(device):
                    start_event = torch_npu.npu.Event(enable_timing=True)
                    end_event = torch_npu.npu.Event(enable_timing=True)
                    torch_npu.npu.synchronize()
                    start_event.record()
                    self._dispatch_prepared_payloads(
                        prepared_payloads,
                        execute_core_operator,
                        implementation,
                        retained_outputs,
                    )
                    end_event.record()
                    end_event.synchronize()
                    total_time = start_event.elapsed_time(end_event)
                return total_time / num_iterations

            if "cuda" in device:
                cuda_device = torch.device(device)
                with torch.cuda.device(cuda_device):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(cuda_device)
                    start_event.record()
                    self._dispatch_prepared_payloads(
                        prepared_payloads,
                        execute_core_operator,
                        implementation,
                        retained_outputs,
                    )
                    end_event.record()
                    end_event.synchronize()
                    total_time = start_event.elapsed_time(end_event)
                return total_time / num_iterations

            start_time = time.perf_counter()
            self._dispatch_prepared_payloads(
                prepared_payloads,
                execute_core_operator,
                implementation,
                retained_outputs,
            )
            return (time.perf_counter() - start_time) * 1000 / num_iterations
        except Exception as e:
            print(f"    ❌ V2计时过程中发生错误: {str(e)}")
            raise

    def _capture_prepared_payload_chain_v2(
        self,
        prepared_payloads: List[Any],
        execute_core_operator: Callable[[Any, str], Any],
        implementation: str,
        device: str,
    ) -> _CapturedPreparedChain:
        """Capture one independent-address chain without eager fallback."""
        if not prepared_payloads:
            raise ValueError("prepared_payloads must not be empty")

        retained_outputs: List[Any] = [None] * len(prepared_payloads)
        if "npu" in device:
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph):
                self._dispatch_prepared_payloads(
                    prepared_payloads,
                    execute_core_operator,
                    implementation,
                    retained_outputs,
                )
        elif "cuda" in device:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._dispatch_prepared_payloads(
                    prepared_payloads,
                    execute_core_operator,
                    implementation,
                    retained_outputs,
                )
        else:
            raise ValueError(
                "captured_chain requires a CUDA or NPU device"
            )

        return _CapturedPreparedChain(
            replay=graph.replay,
            retained_outputs=retained_outputs,
            logical_invocations=len(prepared_payloads),
        )

    @staticmethod
    def _measure_captured_chain_replay_v2(
        captured_chain: _CapturedPreparedChain,
        device: str,
    ) -> float:
        """Time one graph replay and return latency per logical invocation."""
        if captured_chain.logical_invocations <= 0:
            raise ValueError("captured chain must contain an invocation")

        if "npu" in device:
            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)
        elif "cuda" in device:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        else:
            raise ValueError(
                "captured_chain requires a CUDA or NPU device"
            )

        start_event.record()
        captured_chain.replay()
        end_event.record()
        end_event.synchronize()
        return (
            start_event.elapsed_time(end_event)
            / captured_chain.logical_invocations
        )
    
    def run_unified_profile_test(
        self,
        operator_name: str,
        test_data: Dict[str, Any],
        device: str = "npu:0",
        precision: PrecisionType = PrecisionType.BF16,
        num_warmup: int = 5,
        num_iterations: int = 10,
        enable_profile: bool = True,
        test_name: str = "unified_profile",
        profile_level: str = "Level1",
        aic_metrics: str = "PipeUtilization",
        export_type: str = "Text",
        **kwargs
    ) -> Dict[str, Any]:
        """运行统一profile测试 
        
        Args:
            operator_name: 算子名称
            test_data: 测试数据
            device: 设备
            precision: 精度类型
            num_warmup: 预热次数
            num_iterations: 迭代次数
            enable_profile: 是否启用profile
            test_name: 测试名称
            profile_level: Profile级别
            aic_metrics: AIC指标类型
            export_type: 导出类型
            **kwargs: 其他参数
        """
        
        if operator_name not in self.operators:
            raise ValueError(f"算子 {operator_name} 未注册")
        
        operator_test = self.operators[operator_name]
        
        print(f"\n{'='*60}")
        print(f"统一Profile测试: {operator_name} - {test_name}")
        print(f"设备: {device}, 精度: {precision.name}, 迭代次数: {num_iterations}")
        print(f"所有实现的profile将合并到一个文件中")
        print(f"{'='*60}")
        
        implementations = operator_test.get_available_implementations(device)
        comparison_results = {}
        
        # 为每个实现分别进行profile测试
        for impl_name in implementations:
            print(f"\n--- 测试实现: {impl_name} ---")
            
            try:
                # 为当前实现创建独立的profile保存路径
                prof_save_path = None
                prof = None
                
                if enable_profile:
                    prof_save_path = self.result_dir / f"{operator_name}_{impl_name}_{test_name}_{precision.name}_profile"
                    prof_save_path.mkdir(exist_ok=True)
                    
                    # 使用 ProfilerFactory 创建 profiler
                    backend = ProfilerBackend.NPU if device.startswith("npu") else ProfilerBackend.CUDA
                    
                    if device.startswith("npu"):
                        # NPU 配置
                        config = ProfilerConfig(
                            backend=backend,
                            trace_file_path=str(prof_save_path),
                            record_shapes=False,
                            profile_memory=False,
                            with_stack=False,
                            schedule_wait=0,
                            schedule_warmup=0,
                            schedule_active=num_iterations,
                            schedule_repeat=1,
                            schedule_skip_first=1,
                            experimental_config={
                                'profile_level': profile_level,
                                'aic_metrics': aic_metrics,
                                'export_type': export_type
                            }
                        )
                    else:
                        # CUDA 配置
                        config = ProfilerConfig(
                            backend=backend,
                            trace_file_path=str(prof_save_path),
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True,
                            schedule_wait=1,
                            schedule_warmup=1,
                            schedule_active=num_iterations,
                            schedule_repeat=2
                        )
                    
                    prof = ProfilerFactory.create_profiler(config)
                    
                # 执行profile测试
                print(f"开始执行 {num_iterations} 次迭代...")
                for i in range(num_warmup):
                    operator_test.run_device_implementation(
                        test_data, device, precision, impl_name
                    )

                prof.start()
                for i in range(num_iterations):
                    operator_test.run_device_implementation(
                        test_data, device, precision, impl_name
                    )
                    prof.step()
                
                # 设备同步
                if device.startswith("npu"):
                    torch_npu.npu.synchronize()
                elif device.startswith("cuda"):
                    torch.cuda.synchronize()
                
                # 停止当前实现的profile
                if prof is not None:
                    prof.stop()
                    print(f"✅ Profile已保存到: {prof_save_path}")
                
                # 记录结果
                comparison_results[impl_name] = {
                    "profile_path": prof_save_path if enable_profile else None,
                    "iterations": num_iterations,
                    "status": "success"
                }
                
                print(f"✅ 完成 {num_iterations} 次迭代")
                
            except Exception as e:
                print(f"❌ 实现 {impl_name} 测试失败: {str(e)}")
                comparison_results[impl_name] = {
                    "profile_path": None,
                    "iterations": 0,
                    "status": "failed",
                    "error": str(e)
                }
        
        # 保存对比结果
        results = {
            "operator_name": operator_name,
            "test_name": test_name,
            "device": device,
            "precision": precision.name,
            "num_iterations": num_iterations,
            "num_warmup": num_warmup,
            "profile_config": {
                "enable_profile": enable_profile,
                "profile_level": profile_level,
                "aic_metrics": aic_metrics,
                "export_type": export_type
            },
            "comparison_results": comparison_results,
            "test_data_info": test_data.get('metadata', {}),
            "profile_enabled": enable_profile
        }
        
        self.save_results(results, f"{operator_name}_{test_name}_unified_profile")
        
        print(f"\n{'='*60}")
        print("分别Profile测试完成")
        if enable_profile:
            print("每个实现的Profile文件:")
            for impl_name, result in comparison_results.items():
                if result.get("profile_path"):
                    print(f"  {impl_name}: {result['profile_path']}")
        print(f"{'='*60}")
        
        return results

    def run_core_operator_profile_test_v2(
        self,
        operator_test: BaseOperatorTest,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
        *,
        num_warmup: int = 10,
        num_iterations: int = 20,
        trace_file_path: str,
        profile_level: str = "Level1",
        aic_metrics: str = "PipeUtilization",
        verify_independent_storage: bool = True,
    ) -> Dict[str, Any]:
        """Profile prepared core-operator dispatch without full API calls."""
        for count_name, count_value, minimum in (
            ("num_warmup", num_warmup, 0),
            ("num_iterations", num_iterations, 1),
        ):
            if not isinstance(count_value, int) or isinstance(
                count_value,
                bool,
            ):
                raise ValueError(f"{count_name} must be a non-bool int")
            if count_value < minimum:
                if minimum == 0:
                    raise ValueError(f"{count_name} must be >= 0")
                raise ValueError(f"{count_name} must be > {minimum - 1}")
        if "npu" not in device and "cuda" not in device:
            raise ValueError(
                "prepared core profiling requires a CUDA or NPU device"
            )

        invocations = num_warmup + num_iterations
        prepared_payloads: List[Any] = []
        retained_outputs: List[Any] = []
        prepared_input_values: Optional[List[Any]] = None
        profiler = None
        try:
            with torch.inference_mode():
                for _ in range(invocations):
                    prepared_payloads.append(
                        operator_test._prepare_data_for_core_operator(
                            data,
                            device,
                            precision,
                            implementation,
                        )
                    )

            input_storage_sets_verified = 0
            if verify_independent_storage:
                prepared_input_values = [
                    self._prepared_input_values(payload)
                    for payload in prepared_payloads
                ]
                (
                    input_storage_sets_verified,
                    _,
                ) = self._verify_independent_storage_sets(
                    prepared_input_values,
                    device,
                    "prepared core profile input/workspace sets",
                )

            if "npu" in device:
                device_context = torch_npu.npu.device(device)
                backend = ProfilerBackend.NPU
            else:
                device_context = torch.cuda.device(torch.device(device))
                backend = ProfilerBackend.CUDA
            context_factory = getattr(
                operator_test,
                "_core_operator_benchmark_context",
                None,
            )
            provider_context = (
                context_factory(device, precision, implementation)
                if context_factory is not None
                else nullcontext()
            )
            profiler = ProfilerFactory.create_profiler(
                ProfilerConfig(
                    backend=backend,
                    trace_file_path=trace_file_path,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    experimental_config={
                        "profile_level": profile_level,
                        "aic_metrics": aic_metrics,
                    },
                    schedule_wait=0,
                    schedule_warmup=0,
                    schedule_active=num_iterations,
                    schedule_repeat=1,
                    schedule_skip_first=0,
                )
            )
            execute = operator_test._execute_core_operator

            with device_context, provider_context, torch.inference_mode():
                for payload in prepared_payloads[:num_warmup]:
                    retained_outputs.append(execute(payload, implementation))
                if "npu" in device:
                    torch_npu.npu.synchronize()
                else:
                    torch.cuda.synchronize(torch.device(device))

                profiler.start()
                try:
                    for payload in prepared_payloads[num_warmup:]:
                        retained_outputs.append(execute(payload, implementation))
                        profiler.step()
                    if "npu" in device:
                        torch_npu.npu.synchronize()
                    else:
                        torch.cuda.synchronize(torch.device(device))
                finally:
                    profiler.stop()

            output_storage_sets_verified = 0
            if verify_independent_storage:
                (
                    output_storage_sets_verified,
                    _,
                ) = self._verify_independent_storage_sets(
                    retained_outputs,
                    device,
                    "prepared core profile output sets",
                )
                self._verify_disjoint_storage_domains(
                    prepared_input_values,
                    retained_outputs,
                    device,
                    "prepared core profile input/workspace and output domains",
                )

            return {
                "profile_path": trace_file_path,
                "provider": implementation,
                "device": device,
                "precision": precision.name,
                "warmup_iterations": num_warmup,
                "profiled_invocations": num_iterations,
                "preallocated_invocations": invocations,
                "input_storage_sets_verified": input_storage_sets_verified,
                "output_storage_sets_verified": output_storage_sets_verified,
                "dispatch_mode": "eager_prepared_core",
                "profiler_is_diagnostic": True,
            }
        finally:
            retained_outputs.clear()
            prepared_payloads.clear()
            prepared_input_values = None
            gc.collect()

    def register_operator(self, operator_test: BaseOperatorTest):
        """注册算子测试"""
        self.operators[operator_test.operator_name] = operator_test
        print(f"✓ 已注册算子: {operator_test.operator_name}")
    
    def calculate_accuracy_metrics(
        self, 
        reference: torch.Tensor, 
        test_output: torch.Tensor,
        operator_name: str,
        precision_type: str
    ) -> AccuracyMetrics:
        """计算精度指标"""
        
        # 确保tensor在CPU上
        reference = reference.cpu().float()
        test_output = test_output.cpu().float()
        
        # 计算绝对误差
        abs_error = torch.abs(reference - test_output)
        max_abs_error = torch.max(abs_error).item()
        mean_abs_error = torch.mean(abs_error).item()
        
        # 计算相对误差
        rel_error = abs_error / (torch.abs(reference) + 1e-8)
        max_rel_error = torch.max(rel_error).item()
        mean_rel_error = torch.mean(rel_error).item()
        
        # 计算余弦相似度
        reference_flat = reference.flatten()
        test_output_flat = test_output.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            reference_flat.unsqueeze(0), 
            test_output_flat.unsqueeze(0)
        ).item()
        
        # 计算MSE和RMSE
        mse = torch.mean((reference - test_output) ** 2).item()
        rmse = math.sqrt(mse)
        
        return AccuracyMetrics(
            max_abs_error=max_abs_error,
            mean_abs_error=mean_abs_error,
            max_rel_error=max_rel_error,
            mean_rel_error=mean_rel_error,
            cosine_similarity=cosine_sim,
            mse=mse,
            rmse=rmse,
            precision_type=precision_type,
            operator_name=operator_name
        )
    
    def run_performance_test(
        self,
        operator_test: BaseOperatorTest,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
        num_warmup: int = 5,
        num_iterations: int = 20
    ) -> PerformanceMetrics:
        """运行性能测试"""
        
        print(f"  性能测试: {device} - {precision.name} - {implementation}")
        
        # 预热
        for _ in range(num_warmup):
            _ = operator_test.run_device_implementation(data, device, precision, implementation)
        
        # 同步设备
        if "npu" in device:
            torch_npu.npu.synchronize()
        elif "cuda" in device:
            torch.cuda.synchronize()
        
        # 性能测试 - 使用设备事件计时
        def run_once():
            return operator_test.run_device_implementation(data, device, precision, implementation)
        
        avg_time = self._measure_execution_time(run_once, device, num_iterations)
        
        # 计算吞吐量
        throughput = operator_test.calculate_throughput(data, avg_time)
        
        # 计算TOPS和带宽
        tops = None
        bandwidth_gb_s = None
        
        # 尝试计算TOPS
        if hasattr(operator_test, 'calculate_flops'):
            try:
                flops = operator_test.calculate_flops(data)
                if flops is not None and avg_time > 0:
                    # avg_time是毫秒，转换为秒后计算TOPS
                    avg_time_s = avg_time / 1000.0
                    tops = flops / (avg_time_s * 1e12)  # 转换为TOPS
            except Exception as e:
                print(f"警告: 计算TOPS时出错: {e}")
                pass
        
        # 尝试计算带宽
        if hasattr(operator_test, 'calculate_bandwidth'):
            try:
                bandwidth_gb_s = operator_test.calculate_bandwidth(data, avg_time)
            except Exception as e:
                print(f"警告: 计算带宽时出错: {e}")
                pass
        
        # 性能指标将在汇总时显示，这里不重复输出
        
        return PerformanceMetrics(
            avg_time_ms=avg_time,
            throughput=throughput,
            precision_type=precision.name,
            device_type=device,
            operator_name=operator_test.operator_name,
            iterations=num_iterations,
            throughput_ops_per_sec=throughput,
            tops=tops,
            bandwidth_gb_s=bandwidth_gb_s
        )
    
    def _run_core_operator_performance_repeat_v2(
        self,
        operator_test: BaseOperatorTest,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str,
        num_warmup: int,
        num_iterations: int,
        retain_outputs: bool,
        verify_independent_storage: bool,
        dispatch_mode: str,
    ) -> Tuple[
        float, int, int, int, int, int, int, str, str, str
    ]:
        """Run one fresh-storage V2 repeat and return only scalar results."""
        invocations_per_repeat = num_warmup + num_iterations
        prepared_data_list: List[Any] = []
        retained_outputs: List[Any] = []
        result: Optional[
            Tuple[
                float, int, int, int, int, int, int, str, str, str
            ]
        ] = None
        prepared_data = None
        prepared_input_values = None
        prepared_output_values = None
        alias_prepared_data = None
        alias_outputs = None
        output_domain_values = None
        timed_retained_outputs = None
        output = None
        execute_core_operator = None
        provider_context = None
        device_context = None
        captured_chain = None
        measured_payloads = None
        input_audit_payloads = None
        timed_output_capture_policy = "not_retained"
        preallocated_output_contract = "not_declared"
        output_alias_verification_scope = "not_retained"

        try:
            with torch.inference_mode():
                for invocation_index in range(invocations_per_repeat):
                    prepared_data_list.append(
                        operator_test._prepare_data_for_core_operator(
                            data,
                            device,
                            precision,
                            implementation,
                        )
                    )
                    if (
                        (invocation_index + 1) % 10 == 0
                        or invocation_index == 0
                    ):
                        print(
                            "      预分配进度: "
                            f"{invocation_index + 1}/"
                            f"{invocations_per_repeat}"
                        )

            verified_input_sets = 0
            verified_input_ptrs = 0
            measured_payloads = prepared_data_list[num_warmup:]
            if (
                verify_independent_storage
                or dispatch_mode == "captured_chain"
            ):
                input_audit_payloads = (
                    measured_payloads
                    if dispatch_mode == "captured_chain"
                    else prepared_data_list
                )
                prepared_input_values = [
                    self._prepared_input_values(prepared_value)
                    for prepared_value in input_audit_payloads
                ]
                (
                    verified_input_sets,
                    verified_input_ptrs,
                ) = self._verify_independent_storage_sets(
                    prepared_input_values,
                    device,
                    (
                        "V2 captured measured input/workspace sets"
                        if dispatch_mode == "captured_chain"
                        else "V2 prepared input/workspace sets"
                    ),
                )

            if "npu" in device:
                device_context = torch_npu.npu.device(device)
            elif "cuda" in device:
                device_context = torch.cuda.device(torch.device(device))
            else:
                device_context = nullcontext()

            context_factory = getattr(
                operator_test,
                "_core_operator_benchmark_context",
                None,
            )
            provider_context = (
                context_factory(device, precision, implementation)
                if context_factory is not None
                else nullcontext()
            )

            device_type = "npu" if "npu" in device else (
                "cuda" if "cuda" in device else "cpu"
            )
            prepared_output_presence = [
                bool(self._prepared_output_storage_ptrs(
                    prepared_value,
                    device_type,
                ))
                for prepared_value in prepared_data_list
            ]
            if any(prepared_output_presence) and not all(
                prepared_output_presence
            ):
                raise RuntimeError(
                    "prepared output buffers must be present for either "
                    "every invocation or none"
                )
            has_preallocated_outputs = bool(
                prepared_output_presence
                and all(prepared_output_presence)
            )
            contract_presence = [
                bool(
                    operator_test._declares_preallocated_output_contract(
                        prepared_value,
                        implementation,
                    )
                )
                for prepared_value in prepared_data_list
            ]
            if any(contract_presence) and not all(contract_presence):
                raise RuntimeError(
                    "preallocated output contract must be declared for "
                    "either every invocation or none"
                )
            has_preallocated_output_contract = bool(
                contract_presence and all(contract_presence)
            )
            if (
                has_preallocated_output_contract
                and not has_preallocated_outputs
            ):
                raise RuntimeError(
                    "preallocated output contract was declared without "
                    "explicit prepared output buffers"
                )
            direct_preallocated_timing = (
                retain_outputs
                and has_preallocated_outputs
                and has_preallocated_output_contract
                and num_warmup > 0
            )
            if has_preallocated_output_contract:
                preallocated_output_contract = (
                    "declared_phase_invariant_out"
                )
            if direct_preallocated_timing:
                timed_output_capture_policy = (
                    "preallocated_output_contract_no_timed_return_capture"
                )
                output_alias_verification_scope = "warmup_returns_only"
            elif retain_outputs:
                if dispatch_mode == "captured_chain":
                    timed_output_capture_policy = (
                        "capture_returns_retained_outside_timed_replay"
                    )
                    output_alias_verification_scope = (
                        "warmup_and_capture_returns"
                    )
                else:
                    timed_output_capture_policy = (
                        "retained_return_inside_timed_region"
                    )
                    output_alias_verification_scope = (
                        "warmup_and_measured_returns"
                    )

            execute_core_operator = operator_test._execute_core_operator
            verified_output_sets = 0
            verified_output_ptrs = 0
            verified_output_tensors = 0
            with device_context, provider_context, torch.inference_mode():
                for warmup_index in range(num_warmup):
                    output = execute_core_operator(
                        prepared_data_list[warmup_index],
                        implementation,
                    )
                    if retain_outputs:
                        retained_outputs.append(output)

                if "npu" in device:
                    torch_npu.npu.synchronize()
                elif "cuda" in device:
                    torch.cuda.synchronize(torch.device(device))

                if dispatch_mode == "captured_chain":
                    captured_chain = (
                        self._capture_prepared_payload_chain_v2(
                            measured_payloads,
                            execute_core_operator,
                            implementation,
                            device,
                        )
                    )
                    if (
                        captured_chain.logical_invocations
                        != num_iterations
                    ):
                        raise RuntimeError(
                            "captured chain invocation count mismatch: "
                            f"{captured_chain.logical_invocations} != "
                            f"{num_iterations}"
                        )
                    if (
                        len(captured_chain.retained_outputs)
                        != num_iterations
                    ):
                        raise RuntimeError(
                            "captured chain output count mismatch: "
                            f"{len(captured_chain.retained_outputs)} != "
                            f"{num_iterations}"
                        )
                    (
                        verified_output_sets,
                        verified_output_ptrs,
                    ) = self._verify_independent_storage_sets(
                        captured_chain.retained_outputs,
                        device,
                        "V2 captured functional output sets",
                    )
                    verified_output_tensors = sum(
                        self._device_tensor_count(value, device_type)
                        for value in captured_chain.retained_outputs
                    )
                    self._verify_disjoint_storage_domains(
                        prepared_input_values,
                        captured_chain.retained_outputs,
                        device,
                        (
                            "V2 captured measured input/workspace and "
                            "output storage domains"
                        ),
                    )
                    if (
                        not direct_preallocated_timing
                        and retain_outputs
                    ):
                        retained_outputs.extend(
                            captured_chain.retained_outputs
                        )
                    operator_test._restore_mutable_graph_inputs(
                        measured_payloads,
                        implementation,
                    )
                    if "npu" in device:
                        torch_npu.npu.synchronize()
                    else:
                        torch.cuda.synchronize(torch.device(device))
                    operator_test._restore_mutable_graph_inputs(
                        measured_payloads,
                        implementation,
                    )
                    repeat_time_ms = (
                        self._measure_captured_chain_replay_v2(
                            captured_chain,
                            device,
                        )
                    )
                else:
                    if not direct_preallocated_timing and retain_outputs:
                        retained_outputs.extend([None] * num_iterations)
                        timed_retained_outputs = retained_outputs
                    repeat_time_ms = self._measure_execution_time_v2(
                        measured_payloads,
                        execute_core_operator,
                        implementation,
                        device,
                        timed_retained_outputs,
                    )
                if (
                    not math.isfinite(repeat_time_ms)
                    or repeat_time_ms <= 0
                ):
                    raise RuntimeError(
                        f"invalid repeat latency: {repeat_time_ms} ms"
                    )

            alias_prepared_data = prepared_data_list
            alias_outputs = retained_outputs
            if direct_preallocated_timing:
                prepared_output_values = [
                    tuple(self._prepared_output_values(prepared_value))
                    for prepared_value in prepared_data_list
                ]
                if dispatch_mode != "captured_chain":
                    verified_output_tensors = sum(
                        self._device_tensor_count(value, device_type)
                        for value in prepared_output_values
                    )
                if (
                    verify_independent_storage
                    and dispatch_mode != "captured_chain"
                ):
                    (
                        verified_output_sets,
                        verified_output_ptrs,
                    ) = self._verify_independent_storage_sets(
                        prepared_output_values,
                        device,
                        "V2 prepared output sets",
                    )
                alias_prepared_data = prepared_data_list[:num_warmup]
                alias_outputs = retained_outputs
            elif (
                retain_outputs
                and dispatch_mode != "captured_chain"
            ):
                verified_output_tensors = sum(
                    self._device_tensor_count(value, device_type)
                    for value in retained_outputs
                )
            if (
                verify_independent_storage
                and retain_outputs
                and not direct_preallocated_timing
                and dispatch_mode != "captured_chain"
            ):
                (
                    verified_output_sets,
                    verified_output_ptrs,
                ) = self._verify_independent_storage_sets(
                    retained_outputs,
                    device,
                    "V2 retained output sets",
                )

            verified_output_aliases = 0
            if retain_outputs:
                verified_output_aliases = (
                    self._count_preallocated_output_aliases(
                        alias_prepared_data,
                        alias_outputs,
                        device,
                    )
                )
            if (
                direct_preallocated_timing
                and verified_output_aliases != num_warmup
            ):
                raise RuntimeError(
                    "direct preallocated timing contract failed its warmup "
                    "return-alias probe; got "
                    f"{verified_output_aliases}/{num_warmup}"
                )
            if retain_outputs and hasattr(
                operator_test, "_verify_preallocated_output_aliases"
            ):
                provider_verified_aliases = (
                    operator_test._verify_preallocated_output_aliases(
                        alias_prepared_data,
                        alias_outputs,
                        implementation,
                    )
                )
                if (
                    provider_verified_aliases
                    and provider_verified_aliases
                    != verified_output_aliases
                ):
                    raise RuntimeError(
                        "provider output-alias verification disagrees with "
                        "generic storage relation: "
                        f"{provider_verified_aliases} != "
                        f"{verified_output_aliases}"
                    )

            if (
                verify_independent_storage
                and retain_outputs
                and dispatch_mode != "captured_chain"
            ):
                output_domain_values = (
                    prepared_output_values
                    if direct_preallocated_timing
                    else retained_outputs
                )
                self._verify_disjoint_storage_domains(
                    prepared_input_values,
                    output_domain_values,
                    device,
                    "V2 input/workspace and output storage domains",
                )

            result = (
                float(repeat_time_ms),
                verified_input_sets,
                verified_input_ptrs,
                verified_output_sets,
                verified_output_ptrs,
                verified_output_tensors,
                verified_output_aliases,
                timed_output_capture_policy,
                preallocated_output_contract,
                output_alias_verification_scope,
            )
        finally:
            retained_outputs.clear()
            if captured_chain is not None:
                captured_chain.retained_outputs.clear()
            prepared_data_list.clear()
            output = None
            prepared_data = None
            prepared_input_values = None
            prepared_output_values = None
            alias_prepared_data = None
            alias_outputs = None
            output_domain_values = None
            timed_retained_outputs = None
            execute_core_operator = None
            provider_context = None
            device_context = None
            captured_chain = None
            measured_payloads = None
            input_audit_payloads = None
            gc.collect()
            try:
                if "npu" in device:
                    with torch_npu.npu.device(device):
                        torch_npu.npu.empty_cache()
                elif "cuda" in device:
                    with torch.cuda.device(torch.device(device)):
                        torch.cuda.empty_cache()
            except Exception:
                pass

        if result is None:
            raise RuntimeError("repeat completed without a latency result")
        return result

    def run_core_operator_performance_test_v2(
        self,
        operator_test: BaseOperatorTest,
        data: Dict[str, Any],
        device: str,
        precision: PrecisionType,
        implementation: str = "default",
        num_warmup: int = 10,
        num_iterations: int = 20,
        num_repeats: int = 1,
        retain_outputs: bool = True,
        verify_independent_storage: bool = False,
        num_stabilization_repeats: Optional[int] = None,
        dispatch_mode: str = "eager_direct",
    ) -> PerformanceMetrics:
        """Measure the core operator with preallocated fresh storage.

        Every repeat independently prepares ``num_warmup + num_iterations``
        payloads. Optional full-window stabilization repeats use the same
        fresh-storage contract but are excluded from the reported median.
        """
        if dispatch_mode not in {"eager_direct", "captured_chain"}:
            raise ValueError(
                "dispatch_mode must be exactly 'eager_direct' or "
                "'captured_chain'"
            )
        if dispatch_mode == "captured_chain" and not (
            "npu" in device or "cuda" in device
        ):
            raise ValueError(
                "captured_chain requires a CUDA or NPU device"
            )
        if num_stabilization_repeats is None:
            raw_stabilization_repeats = os.environ.get(
                "OPERATOR_TEST_STABILIZATION_REPEATS",
                "0",
            )
            try:
                num_stabilization_repeats = int(
                    raw_stabilization_repeats
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "OPERATOR_TEST_STABILIZATION_REPEATS must be a "
                    "non-negative integer"
                ) from exc
            if str(num_stabilization_repeats) != raw_stabilization_repeats:
                raise ValueError(
                    "OPERATOR_TEST_STABILIZATION_REPEATS must use canonical "
                    "base-10 integer syntax"
                )
        for count_name, count_value, minimum in (
            ("num_warmup", num_warmup, 0),
            ("num_iterations", num_iterations, 1),
            ("num_repeats", num_repeats, 1),
            (
                "num_stabilization_repeats",
                num_stabilization_repeats,
                0,
            ),
        ):
            if not isinstance(count_value, int) or isinstance(
                count_value,
                bool,
            ):
                raise ValueError(f"{count_name} must be a non-bool int")
            if count_value < minimum:
                if minimum == 0:
                    raise ValueError(f"{count_name} must be >= 0")
                raise ValueError(f"{count_name} must be > {minimum - 1}")

        print(
            f"  核心算子性能测试 V2: {device} - {precision.name} - "
            f"{implementation} - S{num_stabilization_repeats}/"
            f"W{num_warmup}/I{num_iterations}/R{num_repeats}"
        )

        has_prepare = hasattr(
            operator_test, "_prepare_data_for_core_operator"
        )
        has_execute = hasattr(operator_test, "_execute_core_operator")
        if not has_prepare or not has_execute:
            if (
                dispatch_mode == "captured_chain"
                or num_repeats != 1
                or num_stabilization_repeats
                or verify_independent_storage
            ):
                raise RuntimeError(
                    "strict Framework V2 requires "
                    "_prepare_data_for_core_operator and "
                    "_execute_core_operator"
                )
            print("    ⚠️  算子不支持分离的核心算子测试，使用完整方法")
            return self.run_performance_test(
                operator_test,
                data,
                device,
                precision,
                implementation,
                num_warmup,
                num_iterations,
            )

        invocations_per_repeat = num_warmup + num_iterations
        stabilization_repeat_samples_ms: List[float] = []
        repeat_samples_ms: List[float] = []
        input_set_counts: List[int] = []
        input_ptr_counts: List[int] = []
        output_set_counts: List[int] = []
        output_ptr_counts: List[int] = []
        output_tensor_counts: List[int] = []
        output_alias_counts: List[int] = []
        output_capture_policies: List[str] = []
        output_contracts: List[str] = []
        output_alias_scopes: List[str] = []

        total_repeats = num_stabilization_repeats + num_repeats
        for repeat_index in range(total_repeats):
            is_stabilization = (
                repeat_index < num_stabilization_repeats
            )
            phase_index = (
                repeat_index
                if is_stabilization
                else repeat_index - num_stabilization_repeats
            )
            phase_count = (
                num_stabilization_repeats
                if is_stabilization
                else num_repeats
            )
            phase_label = (
                "stabilization repeat"
                if is_stabilization
                else "measured repeat"
            )
            print(
                f"    🔁 {phase_label} {phase_index + 1}/{phase_count}: "
                f"预分配 {invocations_per_repeat} 份"
            )
            (
                repeat_time_ms,
                verified_input_sets,
                verified_input_ptrs,
                verified_output_sets,
                verified_output_ptrs,
                verified_output_tensors,
                verified_output_aliases,
                timed_output_capture_policy,
                preallocated_output_contract,
                output_alias_verification_scope,
            ) = self._run_core_operator_performance_repeat_v2(
                operator_test,
                data,
                device,
                precision,
                implementation,
                num_warmup,
                num_iterations,
                retain_outputs,
                verify_independent_storage,
                dispatch_mode,
            )
            if is_stabilization:
                stabilization_repeat_samples_ms.append(repeat_time_ms)
            else:
                repeat_samples_ms.append(repeat_time_ms)
            input_set_counts.append(verified_input_sets)
            input_ptr_counts.append(verified_input_ptrs)
            output_set_counts.append(verified_output_sets)
            output_ptr_counts.append(verified_output_ptrs)
            output_tensor_counts.append(verified_output_tensors)
            output_alias_counts.append(verified_output_aliases)
            output_capture_policies.append(timed_output_capture_policy)
            output_contracts.append(preallocated_output_contract)
            output_alias_scopes.append(output_alias_verification_scope)

        avg_time = float(statistics.median(repeat_samples_ms))
        print(
            f"    ✅ 核心算子中位延迟: {avg_time:.3f}ms; "
            f"repeat means={repeat_samples_ms}"
        )
        
        # 计算吞吐量
        throughput = operator_test.calculate_throughput(data, avg_time)
        
        # 计算TOPS和带宽
        tops = None
        bandwidth_gb_s = None
        
        # 尝试计算TOPS
        if hasattr(operator_test, 'calculate_flops'):
            try:
                flops = operator_test.calculate_flops(data)
                if flops is not None and avg_time > 0:
                    # avg_time是毫秒，转换为秒后计算TOPS
                    avg_time_s = avg_time / 1000.0
                    tops = flops / (avg_time_s * 1e12)  # 转换为TOPS
            except Exception as e:
                print(f"警告: 计算TOPS时出错: {e}")
                pass
        
        # 尝试计算带宽
        if hasattr(operator_test, 'calculate_bandwidth'):
            try:
                bandwidth_gb_s = operator_test.calculate_bandwidth(data, avg_time)
            except Exception as e:
                print(f"警告: 计算带宽时出错: {e}")
                pass
        
        input_sets_verified = min(input_set_counts)
        input_ptr_count = min(input_ptr_counts)
        output_sets_verified = min(output_set_counts)
        output_ptr_count = min(output_ptr_counts)
        output_tensor_count = min(output_tensor_counts)
        output_aliases_verified = min(output_alias_counts)
        if len(set(output_capture_policies)) != 1:
            raise RuntimeError(
                "timed output capture policy changed across repeats: "
                f"{output_capture_policies}"
            )
        timed_output_capture_policy = output_capture_policies[0]
        if len(set(output_contracts)) != 1:
            raise RuntimeError(
                "preallocated output contract changed across repeats: "
                f"{output_contracts}"
            )
        preallocated_output_contract = output_contracts[0]
        if len(set(output_alias_scopes)) != 1:
            raise RuntimeError(
                "output alias verification scope changed across repeats: "
                f"{output_alias_scopes}"
            )
        output_alias_verification_scope = output_alias_scopes[0]
        if dispatch_mode == "captured_chain" and not retain_outputs:
            output_allocation_mode = (
                "capture_outputs_verified_retained_through_replay_only"
            )
        elif not retain_outputs:
            output_allocation_mode = "not_retained_unverified"
        elif timed_output_capture_policy == (
            "preallocated_output_contract_no_timed_return_capture"
        ):
            output_allocation_mode = (
                "preallocated_output_contract_with_warmup_alias_probe"
            )
        elif output_aliases_verified == invocations_per_repeat:
            output_allocation_mode = (
                "preallocated_output_buffers_measured_returns_verified"
            )
        elif output_aliases_verified == 0:
            output_allocation_mode = (
                "no_preallocated_output_buffer_verified"
            )
        else:
            output_allocation_mode = (
                "partially_preallocated_output_buffers_"
                f"{output_aliases_verified}_of_{invocations_per_repeat}"
            )

        return PerformanceMetrics(
            avg_time_ms=avg_time,
            throughput=throughput,
            precision_type=precision.name,
            device_type=device,
            operator_name=f"{operator_test.operator_name}_core_v2",
            iterations=num_iterations,
            throughput_ops_per_sec=throughput,
            tops=tops,
            bandwidth_gb_s=bandwidth_gb_s,
            framework_api=(
                "OperatorTestFramework.run_core_operator_performance_test_v2"
            ),
            warmup_iterations=num_warmup,
            preallocated_input_sets=invocations_per_repeat,
            independent_storage_sets_verified=input_sets_verified,
            independent_output_storage_sets_verified=(
                output_sets_verified
            ),
            preallocated_output_aliases_verified=(
                output_aliases_verified
            ),
            output_allocation_mode=output_allocation_mode,
            output_storage_policy=(
                (
                    "capture_outputs_retained_until_repeat_end"
                    if retain_outputs
                    else "capture_outputs_retained_through_replay_only"
                )
                if dispatch_mode == "captured_chain"
                else (
                    "retained_until_repeat_end"
                    if retain_outputs
                    else "not_retained"
                )
            ),
            protocol_version="operator-test-framework-v2-fresh-v6",
            repeats=num_repeats,
            stabilization_repeats=num_stabilization_repeats,
            stabilization_repeat_samples_ms=(
                stabilization_repeat_samples_ms
            ),
            repeat_samples_ms=repeat_samples_ms,
            aggregation=(
                "median_of_post_stabilization_repeat_means"
                if num_repeats > 1 and num_stabilization_repeats
                else "median_of_repeat_means"
                if num_repeats > 1
                else "single_post_stabilization_repeat_mean"
                if num_stabilization_repeats
                else "single_repeat_mean"
            ),
            preallocated_invocations_per_repeat=invocations_per_repeat,
            input_storage_sets_verified=input_sets_verified,
            input_storage_ptr_count=input_ptr_count,
            input_output_storage_disjoint=bool(
                dispatch_mode == "captured_chain"
                or (
                    verify_independent_storage
                    and retain_outputs
                )
            ),
            output_storage_sets_verified=output_sets_verified,
            output_storage_ptr_count=output_ptr_count,
            output_tensor_count=output_tensor_count,
            input_reuse_within_repeat=False,
            timing_method=(
                "device_event_graph_replay"
                if dispatch_mode == "captured_chain"
                else "device_event"
                if "npu" in device or "cuda" in device
                else "host_perf_counter"
            ),
            timing_semantics=(
                "device elapsed time; capture and mutable restore excluded"
                if dispatch_mode == "captured_chain"
                else "device elapsed time; includes stream-idle gaps between "
                "start/end events caused by host dispatch"
                if "npu" in device or "cuda" in device
                else "host wall-clock elapsed time"
            ),
            timed_output_capture_policy=timed_output_capture_policy,
            preallocated_output_contract=preallocated_output_contract,
            output_alias_verification_scope=(
                output_alias_verification_scope
            ),
            preallocated_output_contract_invocations_per_repeat=(
                invocations_per_repeat
                if preallocated_output_contract == (
                    "declared_phase_invariant_out"
                )
                else 0
            ),
            output_verification_replay_invocations_per_repeat=(
                0
            ),
            total_operator_calls_per_repeat=(
                num_warmup + 2 * num_iterations
                if dispatch_mode == "captured_chain"
                else invocations_per_repeat
            ),
            workspace_allocation_policy="not_audited",
            dispatch_loop_policy=(
                "captured_prepared_payload_chain_single_replay"
                if dispatch_mode == "captured_chain"
                else "python_direct_prepared_payload_loop"
            ),
            device_stabilization_policy=(
                "fresh_storage_full_window_priming_repeats"
                if num_stabilization_repeats
                else "none"
            ),
            device_stabilization_timed=bool(
                num_stabilization_repeats
            ),
            stabilization_operator_calls=(
                num_stabilization_repeats
                * (
                    num_warmup + 2 * num_iterations
                    if dispatch_mode == "captured_chain"
                    else invocations_per_repeat
                )
            ),
            task_queue_enable=(
                os.environ.get("TASK_QUEUE_ENABLE", "unset")
                if "npu" in device
                else "not_applicable"
            ),
            timed_region=(
                (
                    "one graph replay containing I independent core "
                    "invocations"
                )
                if dispatch_mode == "captured_chain"
                else (
                    "Python direct prepared-payload loop of "
                    "_execute_core_operator; prepare excluded; timed Python "
                    "returns discarded under declared out contract"
                )
                if timed_output_capture_policy == (
                    "preallocated_output_contract_no_timed_return_capture"
                )
                else (
                    "Python direct prepared-payload loop of "
                    "_execute_core_operator plus return slot assignment; "
                    "prepare excluded"
                )
                if timed_output_capture_policy == (
                    "retained_return_inside_timed_region"
                )
                else (
                    "Python direct prepared-payload loop of "
                    "_execute_core_operator; prepare excluded"
                )
            ),
            dispatch_mode=dispatch_mode,
            graph_capture_width=(
                num_iterations
                if dispatch_mode == "captured_chain"
                else 0
            ),
            graph_replays=(
                1 if dispatch_mode == "captured_chain" else 0
            ),
            capture_timed=False,
            mutable_inputs_restored=(
                dispatch_mode == "captured_chain"
            ),
            profiler_is_diagnostic=False,
        )
    
    
    def run_accuracy_test(
        self,
        operator_name: str,
        test_data: Dict[str, Any],
        test_name: str = "default"
    ) -> Dict[str, Any]:
        """运行精度测试"""
        
        if operator_name not in self.operators:
            raise ValueError(f"算子 {operator_name} 未注册")
        
        operator_test = self.operators[operator_name]
        
        print(f"\n{'='*60}")
        print(f"精度测试: {operator_name} - {test_name}")
        print(f"{'='*60}")
        
        # 运行CPU参考实现
        print("运行CPU参考实现...")
        cpu_output = operator_test.run_cpu_reference(test_data)
        
        results = {
            'operator_name': operator_name,
            'test_name': test_name,
            'test_data_info': test_data.get('metadata', {}),
            'cpu_output_shape': list(cpu_output.shape),
            'precision_results': {}
        }
        
        # 对每种精度和设备进行测试
        for precision in operator_test.supported_precisions:
            precision_name = precision.name
            print(f"\n=== 测试精度: {precision_name} ===")
            
            precision_results = {}
            
            for device_type in operator_test.supported_devices:
                if device_type == DeviceType.CPU:
                    continue  # CPU已经作为参考实现
                
                device = device_type.value
                if device == "npu":
                    device = "npu:0"
                elif device == "cuda":
                    device = "cuda:0"
                
                device_results = {}
                implementations = operator_test.get_available_implementations(device)
                
                for impl in implementations:
                    try:
                        print(f"  运行 {device} - {impl} 实现...")
                        device_output = operator_test.run_device_implementation(
                            test_data, device, precision, impl
                        )
                        
                        # 计算精度指标
                        metrics = self.calculate_accuracy_metrics(
                            cpu_output, device_output, operator_name, f"{device}_{impl}_{precision_name}"
                        )
                        
                        device_results[impl] = {
                            'metrics': metrics,
                            'success': True
                        }
                        print(f"    ✓ 完成")
                        
                    except Exception as e:
                        print(f"    ❌ 失败: {str(e)}")
                        device_results[impl] = {
                            'error': str(e),
                            'status': 'failed'
                        }
                
                precision_results[device] = device_results
            
            results['precision_results'][precision_name] = precision_results
        
        return results
    
    def _generate_summary(
        self, 
        accuracy_results: List[Dict[str, Any]], 
        performance_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成测试汇总"""
        
        summary = {
            'total_test_cases': len(accuracy_results),
            'accuracy_summary': {},
            'performance_summary': {}
        }
        
        # 精度汇总
        if accuracy_results:
            precision_types = set()
            for result in accuracy_results:
                precision_types.update(result['precision_results'].keys())
            
            for precision in precision_types:
                precision_summary = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'avg_cosine_similarity': 0,
                    'avg_rmse': 0
                }
                
                total_cosine = 0
                total_rmse = 0
                success_count = 0
                
                for result in accuracy_results:
                    if precision in result['precision_results']:
                        for device, device_results in result['precision_results'][precision].items():
                            for impl, impl_result in device_results.items():
                                precision_summary['total_tests'] += 1
                                # 检查impl_result是否直接包含success键，或者是嵌套结构
                                if 'success' in impl_result:
                                    # 直接结构（精度测试）
                                    if impl_result['success']:
                                        precision_summary['successful_tests'] += 1
                                        metrics = impl_result['metrics']
                                        total_cosine += metrics.cosine_similarity
                                        total_rmse += metrics.rmse
                                        success_count += 1
                                elif 'full_function' in impl_result:
                                    # 嵌套结构（性能测试）
                                    if impl_result['full_function']['success']:
                                        precision_summary['successful_tests'] += 1
                                        metrics = impl_result['full_function']['metrics']
                                        total_cosine += metrics.cosine_similarity
                                        total_rmse += metrics.rmse
                                        success_count += 1
                
                if success_count > 0:
                    precision_summary['avg_cosine_similarity'] = total_cosine / success_count
                    precision_summary['avg_rmse'] = total_rmse / success_count
                
                summary['accuracy_summary'][precision] = precision_summary
        
        # 性能汇总
        if performance_results:
            precision_types = set()
            for result in performance_results:
                precision_types.update(result['performance_results'].keys())
            
            for precision in precision_types:
                precision_summary = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'avg_time_ms': 0,
                    'best_implementation': None,
                    'best_time_ms': float('inf')
                }
                
                total_time = 0
                success_count = 0
                
                for result in performance_results:
                    if precision in result['performance_results']:
                        for device, device_results in result['performance_results'][precision].items():
                            for impl, impl_result in device_results.items():
                                precision_summary['total_tests'] += 1
                                # 检查impl_result是否直接包含success键，或者是嵌套结构
                                if 'success' in impl_result:
                                    # 直接结构
                                    if impl_result['success']:
                                        precision_summary['successful_tests'] += 1
                                        metrics = impl_result['metrics']
                                        total_time += metrics.avg_time_ms
                                        success_count += 1
                                        
                                        # 记录最佳实现
                                        if metrics.avg_time_ms < precision_summary['best_time_ms']:
                                            precision_summary['best_time_ms'] = metrics.avg_time_ms
                                            precision_summary['best_implementation'] = f"{device}_{impl}"
                                elif 'full_function' in impl_result:
                                    # 嵌套结构（性能测试）
                                    if impl_result['full_function']['success']:
                                        precision_summary['successful_tests'] += 1
                                        metrics = impl_result['full_function']['metrics']
                                        total_time += metrics.avg_time_ms
                                        success_count += 1
                                        
                                        # 记录最佳实现
                                        if metrics.avg_time_ms < precision_summary['best_time_ms']:
                                            precision_summary['best_time_ms'] = metrics.avg_time_ms
                                            precision_summary['best_implementation'] = f"{device}_{impl}"
                
                if success_count > 0:
                    precision_summary['avg_time_ms'] = total_time / success_count
                
                summary['performance_summary'][precision] = precision_summary
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """保存测试结果"""
        
        # 转换结果为可序列化格式
        serializable_results = self._make_serializable(results)
        
        # 保存JSON格式
        json_file = self.result_dir / f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 测试结果已保存到: {json_file}")
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (AccuracyMetrics, PerformanceMetrics)):
            return obj.__dict__
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)  # 将 PosixPath 转换为字符串
        else:
            return obj
    
    def print_summary_report(self, results: Dict[str, Any]):
        """打印汇总报告"""
        
        print(f"\n{'='*80}")
        print(f"测试汇总报告: {results['operator_name']}")
        print(f"{'='*80}")
        
        summary = results['summary']
        
        print(f"总测试案例数: {summary['total_test_cases']}")
        
        # 精度汇总
        print(f"\n--- 精度测试汇总 ---")
        for precision, precision_summary in summary['accuracy_summary'].items():
            success_rate = precision_summary['successful_tests'] / precision_summary['total_tests'] * 100
            print(f"{precision}:")
            print(f"  成功率: {success_rate:.1f}% ({precision_summary['successful_tests']}/{precision_summary['total_tests']})")
            if precision_summary['successful_tests'] > 0:
                print(f"  平均余弦相似度: {precision_summary['avg_cosine_similarity']:.6f}")
                print(f"  平均RMSE: {precision_summary['avg_rmse']:.6e}")
        
        # 性能汇总
        print(f"\n--- 性能测试汇总 ---")
        for precision, precision_summary in summary['performance_summary'].items():
            success_rate = precision_summary['successful_tests'] / precision_summary['total_tests'] * 100
            print(f"{precision}:")
            print(f"  成功率: {success_rate:.1f}% ({precision_summary['successful_tests']}/{precision_summary['total_tests']})")
            if precision_summary['successful_tests'] > 0:
                print(f"  平均时间: {precision_summary['avg_time_ms']:.3f}ms")
                if precision_summary['best_implementation']:
                    print(f"  最佳实现: {precision_summary['best_implementation']} ({precision_summary['best_time_ms']:.3f}ms)")
    

    
    def list_registered_operators(self):
        """列出已注册的算子"""
        print(f"\n已注册的算子 (共 {len(self.operators)} 个):")
        for name, operator in self.operators.items():
            print(f"  - {name}")
            print(f"    支持精度: {[p.name for p in operator.supported_precisions]}")
            print(f"    支持设备: {[d.value for d in operator.supported_devices]}")
