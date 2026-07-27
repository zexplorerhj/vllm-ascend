"""
RMSNorm算子测试套件
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from rmsnorm.rmsnorm_operator import RMSNormOperatorTest
from operator_test_framework import OperatorTestFramework

class RMSNormTestSuite(BaseTestSuite):
    """RMSNorm算子测试套件"""
    
    def __init__(self):
        super().__init__("RMSNorm")
        self.operator_test = RMSNormOperatorTest()
    
    def register_operator(self):
        """注册RMSNorm算子到测试框架"""
        if self.framework:
            self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建RMSNorm算子测试案例"""
        return [
            {
                'name': 'standard_bert_base',
                'params': {
                    'shape': (32, 128, 768), # Batch, Seq, Hidden
                    'eps': 1e-6
                }
            },
            {
                'name': 'standard_llama_7b',
                'params': {
                    'shape': (1, 2048, 4096), # Batch, Seq, Hidden
                    'eps': 1e-5
                }
            },
             {
                'name': 'large_hidden',
                'params': {
                    'shape': (8, 512, 8192), 
                    'eps': 1e-6
                }
            }
        ]
        
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        return [
             {
                'name': 'quick_small',
                'params': {
                    'shape': (4, 32, 128),
                    'eps': 1e-6
                }
            }
        ]

    def run_performance_test(self, test_cases: List[Dict[str, Any]] = None, include_core_analysis: bool = False, precision_type: Any = None):
        """运行性能测试 - 覆盖基类方法以支持CPU，并使用V2测试"""
        if test_cases is None:
            test_cases = self.create_test_cases()
            
        print(f"\n{'='*80}")
        print(f"{self.operator_name}算子核心性能测试 (V2)")
        print(f"{'='*80}")
        
        # Check supported devices
        devices = []
        from operator_test_framework import DeviceType
        for dt in self.operator_test.supported_devices:
             if dt == DeviceType.CPU:
                 devices.append("cpu")
             elif dt == DeviceType.NPU:
                 devices.append("npu:0")
        
        for test_case in test_cases:
            test_name = test_case['name']
            params = test_case['params']
            print(f"\n🎯 测试用例: {test_name}")
            
            data = self.operator_test.generate_test_data(**params)
            
            for device in devices:
                implementations = self.operator_test.get_available_implementations(device)
                for impl in implementations:
                    for precision in self.operator_test.supported_precisions:
                        print(f"  运行 {device} - {precision.name} - {impl} ...")
                        try:
                            # Use run_core_operator_performance_test_v2 from framework
                            metrics = self.framework.run_core_operator_performance_test_v2(
                                self.operator_test, data, device, precision, impl, num_warmup=10, num_iterations=50
                            )
                            print(f"    平均时间: {metrics.avg_time_ms:.3f} ms")
                            if metrics.throughput:
                                print(f"    吞吐量: {metrics.throughput:.2f} GFLOPS")
                            if metrics.bandwidth_gb_s:
                                print(f"    带宽: {metrics.bandwidth_gb_s:.2f} GB/s")
                                
                        except Exception as e:
                            print(f"    ❌ 失败: {e}")
                            import traceback
                            traceback.print_exc()

    def run_bandwidth_test(
        self,
        *,
        sizes=None,
        hidden_sizes=None,
        target_total_elements=64 * 1024 * 1024,
        device="auto",
        num_warmup=10,
        num_iterations=50,
        num_repeats=3,
        plot_results=True,
        quick=False,
        shard_index=0,
        num_shards=1,
    ):
        """Run both formal BF16 RMSNorm bandwidth matrices."""
        import csv
        import math
        import time
        import torch
        from operator_test_framework import PrecisionType

        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(
                "shard index must satisfy 0 <= index < num_shards"
            )
        sizes = list(sizes) if sizes is not None else [
            2**i for i in range(12, 28)
        ]
        hidden_sizes = (
            list(hidden_sizes) if hidden_sizes is not None
            else [1024 * index for index in range(1, 17)]
        )
        if (
            not sizes
            or not hidden_sizes
            or any(value <= 0 for value in sizes + hidden_sizes)
            or target_total_elements <= 0
        ):
            raise ValueError("RMSNorm curve dimensions must be positive")
        indexed_sizes = list(enumerate(sizes))
        indexed_hidden_sizes = [
            (len(sizes) + index, hidden_size)
            for index, hidden_size in enumerate(hidden_sizes)
        ]
        if quick:
            indexed_sizes = indexed_sizes[:1]
            indexed_hidden_sizes = indexed_hidden_sizes[:1]
        indexed_sizes = [
            item for item in indexed_sizes
            if item[0] % num_shards == shard_index
        ]
        indexed_hidden_sizes = [
            item for item in indexed_hidden_sizes
            if item[0] % num_shards == shard_index
        ]
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
            raise RuntimeError("formal RMSNorm curve requires CUDA or NPU")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        implementations = self.operator_test.get_formal_implementations(device)
        if len(implementations) != 1:
            raise RuntimeError(
                f"expected one formal RMSNorm provider for {device}, got "
                f"{implementations}"
            )
        implementation = implementations[0]
        result_dir = self.framework.result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device_tag = device.replace(":", "_")
        size_csv = result_dir / (
            f"rmsnorm_bandwidth_size_{device_tag}_{timestamp}.csv"
        )
        hidden_csv = result_dir / (
            f"rmsnorm_bandwidth_hidden_{device_tag}_{timestamp}.csv"
        )
        plot_file = result_dir / (
            f"rmsnorm_bandwidth_curve_{device_tag}_{timestamp}.png"
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
            "point_index", "shard_index", "num_shards", "curve",
            "total_elements", "hidden_size", "provider", "device",
            "precision", "avg_time_ms", "bandwidth_gb_s", "status",
            "error", *provenance_fields,
        ]
        size_rows = []
        hidden_rows = []
        failures = []

        def measure_point(
            point_index, curve, total_elements, hidden_size
        ):
            row = {
                "point_index": point_index,
                "shard_index": shard_index,
                "num_shards": num_shards,
                "curve": curve,
                "total_elements": total_elements,
                "hidden_size": hidden_size,
                "provider": implementation,
                "device": device,
                "precision": "BF16",
                "avg_time_ms": "",
                "bandwidth_gb_s": "",
                "status": "pending",
                "error": "",
                "warmup": num_warmup,
                "iterations": num_iterations,
                "repeats": num_repeats,
            }
            try:
                data = self.operator_test.generate_test_data(
                    shape=(total_elements // hidden_size, hidden_size)
                )
                metrics = (
                    self.framework.run_core_operator_performance_test_v2(
                        operator_test=self.operator_test,
                        data=data,
                        device=device,
                        precision=PrecisionType.BF16,
                        implementation=implementation,
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                        num_repeats=num_repeats,
                        retain_outputs=True,
                        verify_independent_storage=True,
                    )
                )
                bandwidth = metrics.bandwidth_gb_s
                if (
                    bandwidth is None
                    or not math.isfinite(bandwidth)
                    or bandwidth <= 0
                ):
                    raise RuntimeError(
                        f"invalid bandwidth result: {bandwidth}"
                    )
                row.update(self.framework.performance_provenance(metrics))
                row.update(
                    avg_time_ms=metrics.avg_time_ms,
                    bandwidth_gb_s=bandwidth,
                    status="ok",
                )
            except Exception as exc:
                failures.append((curve, total_elements, hidden_size, exc))
                row.update(
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            return row

        for point_index, size in indexed_sizes:
            effective_size = max(4096, (size // 4096) * 4096)
            size_rows.append(
                measure_point(
                    point_index, "total_size", effective_size, 4096
                )
            )
            with size_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(size_rows)

        for point_index, hidden_size in indexed_hidden_sizes:
            total_elements = (
                target_total_elements // hidden_size
            ) * hidden_size
            hidden_rows.append(
                measure_point(
                    point_index, "hidden_size", total_elements, hidden_size
                )
            )
            with hidden_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(hidden_rows)

        successful_size = [
            row for row in size_rows if row["status"] == "ok"
        ]
        successful_hidden = [
            row for row in hidden_rows if row["status"] == "ok"
        ]
        if plot_results and (successful_size or successful_hidden):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _, axes = plt.subplots(1, 2, figsize=(20, 8))
            axes[0].plot(
                [row["total_elements"] for row in successful_size],
                [row["bandwidth_gb_s"] for row in successful_size],
                "bo-",
            )
            axes[0].set_xscale("log")
            axes[0].set_title("Bandwidth vs Total Size (Hidden=4096)")
            axes[1].plot(
                [row["hidden_size"] for row in successful_hidden],
                [row["bandwidth_gb_s"] for row in successful_hidden],
                "ro-",
            )
            axes[1].set_title("Bandwidth vs Hidden Size")
            for axis in axes:
                axis.set_ylabel("Bandwidth (GB/s)")
                axis.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.savefig(plot_file, dpi=160)
            plt.close()

        if failures:
            raise RuntimeError(
                f"{len(failures)} formal RMSNorm point(s) failed; "
                f"checkpoints retained at {size_csv} and {hidden_csv}"
            )
        return {
            "size_csv": str(size_csv),
            "hidden_csv": str(hidden_csv),
            "plot_file": str(plot_file) if plot_results else None,
            "size_rows": size_rows,
            "hidden_rows": hidden_rows,
        }

def main():
    """主函数 - 支持独立运行RMSNorm算子测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RMSNorm算子测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["accuracy", "performance", "comprehensive", "fulltest", "bandwidth"],
        default="comprehensive",
        help="测试模式 (默认: comprehensive)"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="test_results",
        help="测试结果保存目录 (默认: test_results)"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--hidden-sizes", type=int, nargs="+")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--target-total-elements",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument("--no-plot", action="store_true")
    
    args = parser.parse_args()
    
    # 初始化测试框架和套件
    framework = OperatorTestFramework(result_dir=args.result_dir)
    suite = RMSNormTestSuite()
    suite.setup(framework)
    
    print(f"开始RMSNorm算子测试 (模式: {args.mode})")
    
    if args.mode == "accuracy":
        suite.run_accuracy_test()
    elif args.mode == "performance":
        suite.run_performance_test()
    elif args.mode == "comprehensive":
        # 综合测试：精度 + 性能
        suite.run_accuracy_test()
        suite.run_performance_test()
    elif args.mode == "fulltest":
        # 全面测试
        # Note: create_full_test_cases is not implemented yet, so we use create_test_cases
        print("注意: create_full_test_cases 未实现，使用 create_test_cases")
        cases = suite.create_test_cases()
        suite.run_accuracy_test(cases)
        suite.run_performance_test(cases)
    elif args.mode == "bandwidth":
        suite.run_bandwidth_test(
            sizes=args.sizes,
            hidden_sizes=args.hidden_sizes,
            target_total_elements=args.target_total_elements,
            device=args.device,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
            num_repeats=args.repeats,
            plot_results=not args.no_plot,
            quick=args.quick,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
