"""Add operator tests and formal bandwidth curve entry point."""

import argparse
import csv
import math
import os
import random
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from add.add_operator import AddOperatorTest
from operator_test_framework import (
    PERFORMANCE_PROVENANCE_FIELDS,
    build_curve_selection_provenance,
    finalize_curve_coverage,
)


class AddTestSuite(BaseTestSuite):
    """Add算子测试套件"""
    
    def __init__(self):
        super().__init__("Add")
        self.operator_test = AddOperatorTest()
    
    def register_operator(self):
        """注册Add算子到测试框架"""
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建Add算子测试案例"""
        return [
            {
                'name': 'large_tensor_wide_range',
                'params': {
                    'shape': (1024, 1024),
                    'value_range': (-1000.0, 1000.0)
                }
            },
            {
                'name': 'large_rectangular',
                'params': {
                    'shape': (512, 2048),
                    'value_range': (-1.0, 1.0)
                }
            },
            {
                'name': 'xlarge_tensor',
                'params': {
                    'shape': (2048, 2048),
                    'value_range': (-1.0, 1.0)
                }
            }
        ]
    
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        return [
            {
                'name': 'quick_medium',
                'params': {
                    'shape': (512, 512),
                    'value_range': (-10.0, 10.0)
                }
            },
        ]
    
    def create_full_test_cases(self, num_cases: int = 50) -> List[Dict[str, Any]]:
        """创建全面测试案例（随机shape与数值范围）"""
        test_cases = []
        preset_shapes = [
            (256, 256), (512, 512), (1024, 512),
            (1024, 1024), (2048, 1024), (512, 2048), (2048, 2048)
        ]
        value_ranges = [(-1.0, 1.0), (-10.0, 10.0), (-1000.0, 1000.0)]
        
        for i in range(num_cases):
            if random.random() < 0.5:
                shape = random.choice(preset_shapes)
            else:
                # 随机二维shape，避免过大导致测试时间过长
                shape = (random.randint(128, 4096), random.randint(128, 4096))
            value_range = random.choice(value_ranges)
            test_cases.append({
                'name': f'full_test_case_{i+1}',
                'params': {
                    'shape': shape,
                    'value_range': value_range
                }
            })
        return test_cases

    def run_bandwidth_test(
        self,
        sizes=None,
        device: str = "auto",
        num_warmup: int = 5,
        num_iterations: int = 20,
        num_repeats: int = 3,
        plot_results: bool = True,
        quick: bool = False,
        shard_index: int = 0,
        num_shards: int = 1,
    ):
        """Run the formal BF16 bandwidth curve with Framework V2."""
        from operator_test_framework import PrecisionType

        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(
                "shard index must satisfy 0 <= index < num_shards"
            )
        formal_sizes = [2**i for i in range(12, 28)]
        if sizes is None:
            sizes = formal_sizes
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
            raise RuntimeError("formal Add curve requires CUDA or NPU")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        implementations = self.operator_test.get_formal_implementations(device)
        if len(implementations) != 1:
            raise RuntimeError(
                f"expected one formal Add provider for {device}, got "
                f"{implementations}"
            )
        implementation = implementations[0]
        result_dir = self.framework.result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_file = result_dir / (
            f"add_bandwidth_{device.replace(':', '_')}_{timestamp}.csv"
        )
        plot_file = result_dir / (
            f"add_bandwidth_curve_{device.replace(':', '_')}_{timestamp}.png"
        )
        provenance_fields = list(PERFORMANCE_PROVENANCE_FIELDS)
        fieldnames = [
            "point_index", "shard_index", "num_shards", "selection_mode",
            "shape_matrix_source", "coverage_mode",
            "coverage_total_formal_points", "coverage_selected_points",
            "coverage_total_requested_points",
            "selection_covers_full_formal_matrix", "coverage_complete",
            "size", "provider",
            "device", "precision", "avg_time_ms", "bandwidth_gb_s",
            "status", "error", *provenance_fields,
        ]
        rows = []
        failures = []

        for point_index, size in indexed_sizes:
            row = {
                "point_index": point_index,
                "shard_index": shard_index,
                "num_shards": num_shards,
                **selection_provenance,
                "size": size,
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
                test_data = self.operator_test.generate_test_data(
                    shape=(size,)
                )
                metrics = (
                    self.framework.run_core_operator_performance_test_v2(
                        operator_test=self.operator_test,
                        data=test_data,
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

            plt.figure(figsize=(12, 7))
            plt.plot(
                [row["size"] for row in successful_rows],
                [row["bandwidth_gb_s"] for row in successful_rows],
                "bo-",
            )
            plt.xscale("log")
            plt.xlabel("Tensor Size (elements)")
            plt.ylabel("Bandwidth (GB/s)")
            plt.title(f"Add Operator Bandwidth ({device})")
            plt.grid(True, which="both", alpha=0.5)
            plt.savefig(plot_file, dpi=160)
            plt.close()

        if failures:
            raise RuntimeError(
                f"{len(failures)} formal Add point(s) failed; "
                f"checkpoint retained at {csv_file}"
            )
        return {
            "csv_file": str(csv_file),
            "plot_file": str(plot_file) if plot_results else None,
            "rows": rows,
        }

def main():
    """主函数 - 支持独立运行Add算子测试"""
    from operator_test_framework import OperatorTestFramework
    
    parser = argparse.ArgumentParser(description="Add算子测试")
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
    parser.add_argument(
        "--num-cases",
        type=int,
        default=50,
        help="fulltest模式下生成的随机测试案例数量 (默认: 50)"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true")
    
    args = parser.parse_args()
    
    # 设置测试框架
    framework = OperatorTestFramework(result_dir=args.result_dir)
    
    # 创建并设置Add测试套件
    add_suite = AddTestSuite()
    add_suite.setup(framework)
    
    # 根据模式运行测试
    try:
        if args.mode == "accuracy":
            print("🧪 运行 Add 算子精度测试...")
            results = add_suite.run_accuracy_test()
        elif args.mode == "performance":
            print("🎯 运行 Add 算子性能测试（专注算子本身，忽略Python下发开销）...")
            from operator_test_framework import PrecisionType
            results = add_suite.run_performance_test(precision_type=PrecisionType.BF16)
        elif args.mode == "comprehensive":
            print("🔍 运行 Add 算子综合测试...")
            results = add_suite.run_comprehensive_test()
        elif args.mode == "fulltest":
            print("🎯 运行 Add 算子全面随机测试...")
            full_test_cases = add_suite.create_full_test_cases(num_cases=args.num_cases)
            results = add_suite.run_comprehensive_test(full_test_cases)
        elif args.mode == "bandwidth":
            print("📈 运行 Add 算子带宽测试...")
            results = add_suite.run_bandwidth_test(
                sizes=args.sizes,
                device=args.device,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                num_repeats=args.repeats,
                plot_results=not args.no_plot,
                quick=args.quick,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        else:
            print(f"❌ 不支持的测试模式: {args.mode}")
            print("支持的模式: accuracy, performance, comprehensive, fulltest, bandwidth")
            return 1
        
        print(f"\n✅ Add算子测试完成，结果已保存到 {args.result_dir}")
        return 0
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
