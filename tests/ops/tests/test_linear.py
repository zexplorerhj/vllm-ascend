"""
Linear算子测试套件 - 使用torch.nn.functional.linear
主要提供Profile和Performance测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from operator_test_framework import PrecisionType
from linear.linear_operator import LinearOperatorTest

class LinearTestSuite(BaseTestSuite):
    """Linear算子测试套件 - 基于torch.nn.functional.linear"""
    
    def __init__(self, precision: str = "bf16", batch_size: int = 128, input_dim: int = 1024, output_dim: int = 4096):
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
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.operator_test = LinearOperatorTest()
    
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
        precision_map = {
            "bf16": PrecisionType.BF16,
            "fp16": PrecisionType.FP16,
        }
        precision_type = precision_map[self.precision]
        
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
            precision_map = {
                "fp16": PrecisionType.FP16,
                "bf16": PrecisionType.BF16,
            }
            precision_type = precision_map.get(self.precision, PrecisionType.BF16)
        
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
        num_iterations=50,
        num_repeats=3,
        plot_results=True,
    ):
        """Run the bias-free formal GEMM curve with Framework V2."""
        import csv
        import math
        import time
        import torch

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

        if device == "auto":
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
            raise RuntimeError("formal Linear curve requires CUDA or NPU")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        implementations = self.operator_test.get_formal_implementations(device)
        if len(implementations) != 1:
            raise RuntimeError(
                f"expected one formal Linear provider for {device}, got "
                f"{implementations}"
            )
        implementation = implementations[0]
        precision_type = {
            "fp16": PrecisionType.FP16,
            "bf16": PrecisionType.BF16,
        }[self.precision]
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
        provenance_fields = [
            "framework_api", "protocol_version", "warmup", "iterations",
            "repeats", "repeat_samples_ms", "aggregation",
            "preallocated_invocations_per_repeat",
            "input_reuse_within_repeat", "input_storage_sets_verified",
            "input_storage_ptr_count", "output_storage_sets_verified",
            "output_storage_ptr_count", "output_storage_policy",
            "timed_region",
        ]
        fieldnames = [
            "M", "N", "K", "bias", "provider", "device", "precision",
            "avg_time_ms", "TFLOPS", "status", "error",
            *provenance_fields,
        ]
        rows = []
        failures = []
        for size in sizes:
            row = {
                "M": size,
                "N": size,
                "K": size,
                "bias": False,
                "provider": implementation,
                "device": device,
                "precision": self.precision.upper(),
                "avg_time_ms": "",
                "TFLOPS": "",
                "status": "pending",
                "error": "",
                "warmup": num_warmup,
                "iterations": num_iterations,
                "repeats": num_repeats,
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
                        num_iterations=num_iterations,
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
    parser.add_argument('--precision', choices=['fp16', 'bf16'], default='bf16', help='精度类型')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--input-dim', type=int, default=1024, help='输入维度')
    parser.add_argument('--output-dim', type=int, default=4096, help='输出维度')
    parser.add_argument('--device', default='npu:0', help='测试设备')
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
    parser.add_argument('--tflops-iterations', type=int, default=50)
    parser.add_argument('--tflops-repeats', type=int, default=3)
    parser.add_argument('--no-plot', action='store_true')

    parser.add_argument('--custom-dims', type=str, help='自定义维度列表，格式: batch,input,output (例如: 64,512,2048)')
    
    args = parser.parse_args()
    
    # 创建测试框架
    framework = OperatorTestFramework()
    
    # 创建测试套件
    test_suite = LinearTestSuite(
        precision=args.precision,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim
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
            precision_map = {
                "fp16": PrecisionType.FP16,
                "bf16": PrecisionType.BF16,
            }
            precision_type = precision_map[args.precision]
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
