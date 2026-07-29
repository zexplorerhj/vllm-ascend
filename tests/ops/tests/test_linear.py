"""
Linear算子测试套件 - 使用torch.nn.functional.linear
主要提供Profile和Performance测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional
from tests.base_test_suite import BaseTestSuite
from operator_test_framework import (
    FRESH_ITERATION_PLAN_FIELDS,
    PERFORMANCE_PROVENANCE_FIELDS,
    PrecisionType,
    build_fresh_iteration_plan,
    build_curve_selection_provenance,
    finalize_curve_coverage,
)
from linear.linear_fp8_operator import LinearFp8OperatorTest
from linear.linear_fp8_npu_operator import LinearFp8NpuOperatorTest
from linear.linear_mxfp8_npu_operator import LinearMxFp8NpuOperatorTest
from linear.linear_operator import LinearOperatorTest

LINEAR_BASE_ITERATIONS = 50
LINEAR_PRECISION_TYPES = {
    "fp16": PrecisionType.FP16,
    "bf16": PrecisionType.BF16,
    "fp8": PrecisionType.FP8,
    "mxfp8": PrecisionType.MXFP8,
}
LINEAR_QUANTIZATION_SEMANTICS = {
    "fp16": "none; FP16 activation/weight",
    "bf16": "none; BF16 activation/weight",
    "fp8": (
        "E4M3 activation/weight; per-token activation and "
        "per-output-channel weight FP32 scales"
    ),
    "mxfp8": (
        "E4M3 activation/weight; group32 pair-packed E8M0 "
        "activation/weight scales"
    ),
}
LINEAR_OUTPUT_SEMANTICS = {
    "fp16": "FP16,no_bias",
    "bf16": "BF16,no_bias",
    "fp8": "BF16,no_bias",
    "mxfp8": "BF16,no_bias",
}


def estimate_linear_fp8_fresh_bytes(m: int, n: int, k: int) -> int:
    """Return retained bytes for one FP8 W8A8/BF16 Linear payload."""
    if min(m, n, k) <= 0:
        raise ValueError("Linear FP8 dimensions must be positive")
    return m * k + k * n + 2 * m * n + 4 * m + 4 * n


def estimate_linear_mxfp8_fresh_bytes(m: int, n: int, k: int) -> int:
    """Return retained bytes for one group-32 MXFP8/BF16 payload."""
    if min(m, n, k) <= 0:
        raise ValueError("Linear MXFP8 dimensions must be positive")
    groups_per_row = (k + 31) // 32
    return (
        m * k
        + k * n
        + 2 * m * n
        + groups_per_row * (m + n)
    )


class LinearTestSuite(BaseTestSuite):
    """Linear算子测试套件 - 基于torch.nn.functional.linear"""
    
    def __init__(
        self,
        precision: str = "bf16",
        batch_size: int = 128,
        input_dim: int = 1024,
        output_dim: int = 4096,
        device: str = "auto",
    ):
        """
        初始化Linear测试套件
        
        Args:
            precision: 精度类型，"fp16", "bf16"
            batch_size: 批次大小
            input_dim: 输入维度
            output_dim: 输出维度
        """
        super().__init__("Linear")
        self.precision = precision.lower()
        if self.precision not in LINEAR_PRECISION_TYPES:
            raise ValueError(f"unsupported Linear precision: {self.precision}")
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.requested_device = device
        self.operator_test = self._operator_type_for_device(device)()

    def _operator_type_for_device(self, device: str):
        if self.precision == "fp8":
            if device and device.startswith("npu"):
                return LinearFp8NpuOperatorTest
            return LinearFp8OperatorTest
        if self.precision == "mxfp8":
            return LinearMxFp8NpuOperatorTest
        return LinearOperatorTest

    def _select_operator_for_device(self, device: str) -> None:
        managed_types = (
            LinearOperatorTest,
            LinearFp8OperatorTest,
            LinearFp8NpuOperatorTest,
            LinearMxFp8NpuOperatorTest,
        )
        if not isinstance(self.operator_test, managed_types):
            return
        operator_type = self._operator_type_for_device(device)
        if type(self.operator_test) is not operator_type:
            self.operator_test = operator_type()
    
    def register_operator(self):
        """注册Linear算子到测试框架"""
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建标准测试案例 - Linear主要使用Profile测试"""
        return self.create_profile_test_cases()
    
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例 - 使用较小的维度"""
        return [
            {
                'name': f'qwen3-32b_tp8_batch128_ffn_up_{self.precision}',
                'params': {
                    'batch_size': 128,
                    'input_dim': 5120,
                    'output_dim': 6400,  # FFN上投影
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
        ]
    
    def create_profile_test_cases(self) -> List[Dict[str, Any]]:
        """创建专门的Profile测试案例"""
        return [
            {
                'name': f'qwen3-32b_tp8_batch128_ffn_up_{self.precision}',
                'params': {
                    'batch_size': 128,
                    'input_dim': 5120,
                    'output_dim': 6400,  # FFN上投影
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch128_ffn_down_{self.precision}',
                'params': {
                    'batch_size': 128,
                    'input_dim': 3200,
                    'output_dim': 5120,  # FFN下投影
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch128_oproj_{self.precision}',
                'params': {
                    'batch_size': 128,
                    'input_dim': 1024,
                    'output_dim': 5120,
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch128_qkv_{self.precision}',
                'params': {
                    'batch_size': 128,
                    'input_dim': 5120,
                    'output_dim': 1280,  # 更宽的输出
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch256_ffn_up_{self.precision}',
                'params': {
                    'batch_size': 256,
                    'input_dim': 5120,
                    'output_dim': 6400,  # FFN上投影
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch256_ffn_down_{self.precision}',
                'params': {
                    'batch_size': 256,
                    'input_dim': 3200,
                    'output_dim': 5120,  # FFN下投影
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch256oproj_{self.precision}',
                'params': {
                    'batch_size': 256,
                    'input_dim': 1024,
                    'output_dim': 5120,
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch256_qkv_{self.precision}',
                'params': {
                    'batch_size': 256,
                    'input_dim': 5120,
                    'output_dim': 1280,  # 更宽的输出
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch512_ffn_up_{self.precision}',
                'params': {
                    'batch_size': 512,
                    'input_dim': 5120,
                    'output_dim': 6400,  # FFN上投影
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch512_ffn_down_{self.precision}',
                'params': {
                    'batch_size': 512,
                    'input_dim': 3200,
                    'output_dim': 5120,  # FFN下投影
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch512oproj_{self.precision}',
                'params': {
                    'batch_size': 512,
                    'input_dim': 1024,
                    'output_dim': 5120,
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': f'qwen3-32b_tp8_batch512_qkv_{self.precision}',
                'params': {
                    'batch_size': 512,
                    'input_dim': 5120,
                    'output_dim': 1280,  # 更宽的输出
                    'bias': False,
                    'value_range': (-1.0, 1.0)
                }
            }
        ]
    
    def run_profile_test(self, test_cases: List[Dict[str, Any]] = None, num_iterations: int = 10):
        """运行Profile测试"""
        if test_cases is None:
            test_cases = self.create_profile_test_cases()
        
        # 确定精度类型
        precision_type = LINEAR_PRECISION_TYPES[self.precision]
        
        return super().run_profile_test(
            test_cases=test_cases,
            num_iterations=num_iterations,
            precision_type=precision_type
        )
    
    def run_performance_test_v2(self, test_cases: List[Dict[str, Any]] = None, precision_type: PrecisionType = None, num_warmup: int = 20, num_iterations: int = 100):
        """运行Linear算子V2性能测试 - 专为矩阵乘法优化
        
        Args:
            test_cases: 测试案例列表
            precision_type: 精度类型
            num_warmup: 预热次数（矩阵乘法建议20+）
            num_iterations: 测试迭代次数（矩阵乘法建议200+）
        
        Returns:
            Dict: V2性能测试结果
        """
        if test_cases is None:
            test_cases = self.create_profile_test_cases()
        
        print(f"\n🚀 Linear算子V2性能测试")
        print(f"📐 使用torch.nn.functional.linear")
        print(f"🎯 预先准备数据，零开销性能测试")
        print(f"🔥 预热次数: {num_warmup}")
        print(f"🔄 测试迭代: {num_iterations}")
        print(f"📊 精度: {self.precision.upper()}")
        
        # 确定精度类型
        if precision_type is None:
            precision_type = LINEAR_PRECISION_TYPES[self.precision]
        
        # 调用父类的V2性能测试方法
        return super().run_performance_test_suite_v2(
            test_cases=test_cases,
            precision_type=precision_type,
            num_warmup=num_warmup,
            num_iterations=num_iterations
        )

    def run_tflops_test(
        self,
        start=256,
        end=4096,
        step=128,
        *,
        sizes=None,
        device="auto",
        num_warmup=10,
        num_iterations: Optional[int] = None,
        num_repeats=3,
        plot_results=True,
        quick=False,
        shard_index=0,
        num_shards=1,
    ):
        """Run the bias-free formal GEMM curve with Framework V2."""
        import csv
        import math
        import time
        import torch

        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(
                "shard index must satisfy 0 <= index < num_shards"
            )
        formal_sizes = list(range(256, 4096 + 1, 128))
        if formal_sizes[-1] != 4096:
            formal_sizes.append(4096)
        if sizes is None:
            if start <= 0 or end < start or step <= 0:
                raise ValueError("require 0 < start <= end and step > 0")
            sizes = list(range(start, end + 1, step))
            if sizes[-1] != end:
                sizes.append(end)
        else:
            sizes = list(sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("sizes must contain positive integers")
        indexed_sizes = list(enumerate(sizes))
        total_requested_points = len(indexed_sizes)
        if quick:
            indexed_sizes = indexed_sizes[:1]
        indexed_sizes = [
            (point_index, size)
            for point_index, size in indexed_sizes
            if point_index % num_shards == shard_index
        ]
        coverage_selected_points = len(indexed_sizes)
        selection_provenance = build_curve_selection_provenance(
            quick=quick,
            num_shards=num_shards,
            total_formal_points=len(formal_sizes),
            total_requested_points=total_requested_points,
            selected_points=coverage_selected_points,
            uses_formal_shape_matrix=sizes == formal_sizes,
        )

        if device == "auto":
            if self.precision == "fp8":
                if torch.cuda.is_available():
                    device = "cuda:0"
                else:
                    try:
                        import torch_npu
                        if torch_npu.npu.is_available():
                            device = "npu:0"
                    except (ImportError, AttributeError, RuntimeError):
                        pass
            elif self.precision == "mxfp8":
                try:
                    import torch_npu
                    if torch_npu.npu.is_available():
                        device = "npu:0"
                except (ImportError, AttributeError, RuntimeError):
                    pass
            else:
                try:
                    import torch_npu
                    if torch_npu.npu.is_available():
                        device = "npu:0"
                except (ImportError, AttributeError, RuntimeError):
                    pass
                if device == "auto" and torch.cuda.is_available():
                    device = "cuda:0"
        elif device == "cuda":
            device = "cuda:0"
        elif device == "npu":
            device = "npu:0"
        if device == "auto":
            if self.precision == "fp8":
                raise RuntimeError(
                    "formal FP8 Linear curve requires CUDA SM90/H20 or "
                    "NPU Ascend 950PR"
                )
            if self.precision == "mxfp8":
                raise RuntimeError(
                    "formal MXFP8 Linear curve requires NPU Ascend 950PR"
                )
            raise RuntimeError("formal Linear curve requires CUDA or NPU")
        if self.precision == "mxfp8" and not device.startswith("npu"):
            raise RuntimeError(
                "formal MXFP8 Linear curve requires NPU Ascend 950PR"
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        self._select_operator_for_device(device)
        implementations = self.operator_test.get_formal_implementations(device)
        if len(implementations) != 1:
            if self.precision == "fp8":
                raise RuntimeError(
                    "formal FP8 Linear curve requires CUDA SM90/H20 or "
                    "NPU Ascend 950PR; "
                    f"device={device}, providers={implementations}"
                )
            if self.precision == "mxfp8":
                raise RuntimeError(
                    "formal MXFP8 Linear curve requires NPU Ascend 950PR; "
                    f"device={device}, providers={implementations}"
                )
            raise RuntimeError(
                f"expected one formal Linear provider for {device}, got "
                f"{implementations}"
            )
        implementation = implementations[0]
        precision_type = LINEAR_PRECISION_TYPES[self.precision]
        result_dir = self.framework.result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_file = result_dir / (
            f"linear_tflops_{self.precision}_"
            f"{device.replace(':', '_')}_{timestamp}.csv"
        )
        plot_file = result_dir / (
            f"linear_tflops_curve_{self.precision}_"
            f"{device.replace(':', '_')}_{timestamp}.png"
        )
        provenance_fields = list(PERFORMANCE_PROVENANCE_FIELDS)
        fieldnames = [
            "point_index", "shard_index", "num_shards", "selection_mode",
            "shape_matrix_source", "coverage_mode",
            "coverage_total_formal_points", "coverage_selected_points",
            "coverage_total_requested_points",
            "selection_covers_full_formal_matrix", "coverage_complete",
            "M", "N", "K",
            "bias", "provider", "device", "precision",
            "quantization_semantics", "output_semantics", "avg_time_ms",
            "TFLOPS", "status", "error",
            *FRESH_ITERATION_PLAN_FIELDS,
            *provenance_fields,
        ]
        rows = []
        failures = []
        for point_index, size in indexed_sizes:
            iteration_plan = build_fresh_iteration_plan(
                num_warmup=num_warmup,
                requested_iterations=num_iterations,
                base_iterations=LINEAR_BASE_ITERATIONS,
                estimated_unique_bytes_per_invocation=(
                    estimate_linear_fp8_fresh_bytes(size, size, size)
                    if self.precision == "fp8"
                    else estimate_linear_mxfp8_fresh_bytes(
                        size, size, size
                    )
                    if self.precision == "mxfp8"
                    else 6 * size * size
                ),
            )
            effective_iterations = int(
                iteration_plan["effective_iterations"]
            )
            row = {
                "point_index": point_index,
                "shard_index": shard_index,
                "num_shards": num_shards,
                **selection_provenance,
                "M": size,
                "N": size,
                "K": size,
                "bias": False,
                "provider": implementation,
                "device": device,
                "precision": self.precision.upper(),
                "quantization_semantics": (
                    LINEAR_QUANTIZATION_SEMANTICS[self.precision]
                ),
                "output_semantics": (
                    LINEAR_OUTPUT_SEMANTICS[self.precision]
                ),
                "avg_time_ms": "",
                "TFLOPS": "",
                "status": "pending",
                "error": "",
                "warmup": num_warmup,
                "iterations": effective_iterations,
                "repeats": num_repeats,
                **iteration_plan,
            }
            try:
                test_data = self.operator_test.generate_test_data(
                    batch_size=size,
                    input_dim=size,
                    output_dim=size,
                    bias=False,
                )
                metrics = (
                    self.framework.run_core_operator_performance_test_v2(
                        operator_test=self.operator_test,
                        data=test_data,
                        device=device,
                        precision=precision_type,
                        implementation=implementation,
                        num_warmup=num_warmup,
                        num_iterations=effective_iterations,
                        num_repeats=num_repeats,
                        retain_outputs=True,
                        verify_independent_storage=True,
                    )
                )
                tflops = (
                    metrics.throughput / 1000.0
                    if metrics.throughput is not None else None
                )
                if (
                    tflops is None
                    or not math.isfinite(tflops)
                    or tflops <= 0
                ):
                    raise RuntimeError(f"invalid TFLOPS result: {tflops}")
                row.update(self.framework.performance_provenance(metrics))
                row.update(
                    avg_time_ms=metrics.avg_time_ms,
                    TFLOPS=tflops,
                    status="ok",
                )
            except Exception as exc:
                failures.append((size, exc))
                row.update(
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
            with csv_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        if finalize_curve_coverage(rows):
            with csv_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        successful_rows = [row for row in rows if row["status"] == "ok"]
        if plot_results and successful_rows:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(
                [row["M"] for row in successful_rows],
                [row["TFLOPS"] for row in successful_rows],
                "bo-",
            )
            plt.xlabel("Matrix Size (M=N=K)")
            plt.ylabel("TFLOPS")
            plt.title(
                f"Linear TFLOPS ({device}, {self.precision.upper()})"
            )
            plt.grid(True, alpha=0.5)
            plt.savefig(plot_file, dpi=160)
            plt.close()

        if failures:
            raise RuntimeError(
                f"{len(failures)} formal Linear point(s) failed; "
                f"checkpoint retained at {csv_file}"
            )
        return {
            "csv_file": str(csv_file),
            "plot_file": str(plot_file) if plot_results else None,
            "rows": rows,
        }

def main():
    """主函数 - 支持独立运行Linear算子Profile测试"""
    import argparse
    from operator_test_framework import OperatorTestFramework
    
    parser = argparse.ArgumentParser(description='Linear 算子 Profile 测试')
    parser.add_argument(
        '--precision',
        choices=['fp16', 'bf16', 'fp8', 'mxfp8'],
        default='bf16',
        help='精度类型',
    )
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--input-dim', type=int, default=1024, help='输入维度')
    parser.add_argument('--output-dim', type=int, default=4096, help='输出维度')
    parser.add_argument('--device', default='auto', help='测试设备')
    parser.add_argument('--iterations', type=int, default=10, help='迭代次数')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['profile', 'performance', 'tflops'],
        default='profile',
        help='测试模式: profile(Profile测试), performance(V2性能测试-矩阵乘法优化), tflops(TFLOPS测试)'
    )
    
    parser.add_argument('--tflops-start', type=int, default=256, help='TFLOPS测试起始大小')
    parser.add_argument('--tflops-end', type=int, default=4096, help='TFLOPS测试结束大小')
    parser.add_argument('--tflops-step', type=int, default=128, help='TFLOPS测试步长')
    parser.add_argument('--tflops-sizes', type=int, nargs='+')
    parser.add_argument('--tflops-warmup', type=int, default=10)
    parser.add_argument(
        '--tflops-iterations',
        type=int,
        default=None,
        help=(
            'fixed measured iterations; omitted selects deterministic '
            'fresh-storage adaptive iterations'
        ),
    )
    parser.add_argument('--tflops-repeats', type=int, default=3)
    parser.add_argument('--result-dir', default='test_results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--no-plot', action='store_true')

    parser.add_argument('--custom-dims', type=str, help='自定义维度列表，格式: batch,input,output (例如: 64,512,2048)')
    
    args = parser.parse_args()
    
    # 创建测试框架
    framework = OperatorTestFramework(result_dir=args.result_dir)
    
    # 创建测试套件
    test_suite = LinearTestSuite(
        precision=args.precision,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        device=args.device,
    )
    
    # 设置框架并注册算子
    test_suite.setup(framework)
    
    print(f"🔧 使用 torch.nn.functional.linear 进行 {args.precision.upper()} 精度测试")
    print(f"📐 默认维度: batch={args.batch_size}, input={args.input_dim}, output={args.output_dim}")
    
    try:
        # 创建自定义测试案例（如果指定了自定义维度）
        test_cases = None
        if args.custom_dims:
            dims = [int(x.strip()) for x in args.custom_dims.split(',')]
            if len(dims) != 3:
                print("❌ 自定义维度格式错误，应为: batch,input,output")
                return 1
            
            batch, input_dim, output_dim = dims
            test_cases = [{
                'name': f'custom_{batch}x{input_dim}x{output_dim}_{args.precision}',
                'params': {
                    'batch_size': batch,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'bias': True,
                    'value_range': (-1.0, 1.0)
                }
            }]
            print(f"📋 使用自定义维度: {batch}×{input_dim}×{output_dim}")
        else:
            print(f"📋 使用默认Profile测试案例")
        
        # 根据模式运行相应测试
        if args.mode == "profile":
            print("🚀 运行 Linear Profile 测试...")
            results = test_suite.run_profile_test(
                test_cases=test_cases,
                num_iterations=args.iterations
            )
            
            print(f"\n{'='*60}")
            print("✅ Profile 测试完成！")
            print(f"📁 结果已保存到 test_results 目录")
            print(f"📊 成功测试: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
            print(f"{'='*60}")
            
        elif args.mode == "performance":
            print("🚀 运行 Linear V2 性能测试 (矩阵乘法优化)...")
            # 确定精度类型
            precision_type = LINEAR_PRECISION_TYPES[args.precision]
            results = test_suite.run_performance_test_v2(
                test_cases=test_cases,
                precision_type=precision_type,
                num_warmup=20,
                num_iterations=100
            )
            
            print(f"\n{'='*60}")
            print("✅ V2 性能测试完成！")
            print(f"📁 结果已保存到 test_results 目录")
            print(f"📊 成功测试: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
            print(f"🚀 测试版本: {results['summary']['version'].upper()}")
            print(f"{'='*60}")
        
        elif args.mode == "tflops":
            print("📈 运行 Linear TFLOPS 测试...")
            test_suite.run_tflops_test(
                start=args.tflops_start,
                end=args.tflops_end,
                step=args.tflops_step,
                sizes=args.tflops_sizes,
                device=args.device,
                num_warmup=args.tflops_warmup,
                num_iterations=args.tflops_iterations,
                num_repeats=args.tflops_repeats,
                plot_results=not args.no_plot,
                quick=args.quick,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        return 0
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 强制清理NPU资源，防止Segmentation fault
        try:
            import torch_npu
            import gc
            
            # 清空NPU缓存
            torch_npu.npu.empty_cache()
            
            # 强制垃圾回收
            gc.collect()
            
            # 同步NPU
            torch_npu.npu.synchronize()
            
            print("🧹 NPU资源清理完成")
            
        except ImportError:
            pass
        except Exception as cleanup_error:
            print(f"⚠️ 资源清理时出错: {cleanup_error}")
        
if __name__ == "__main__":
    raise SystemExit(main())
