
"""
FlashAttention算子测试套件
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    import torch_npu
except ImportError:
    pass
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.base_test_suite import BaseTestSuite
from flashattention.base import FlashAttentionOperatorTest
from operator_test_framework import OperatorTestFramework

class FlashAttentionTestSuite(BaseTestSuite):
    """FlashAttention算子测试套件"""
    
    def __init__(self):
        super().__init__("flash_attention")
        self.operator_test = FlashAttentionOperatorTest()
    
    def register_operator(self):
        """注册FlashAttention算子到测试框架"""
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建FlashAttention算子测试案例"""
        return [
            {
                'name': 'standard_bnsd',
                'params': {
                    'batch_size': 4,
                    'num_heads': 32,
                    'seq_len': 128,
                    'head_size': 128,
                    'input_layout': 'BNSD',
                    'sparse_mode': 0
                }
            },
            {
                'name': 'gqa_bnsd',
                'params': {
                    'batch_size': 4,
                    'num_heads': 32,
                    'num_kv_heads': 8,
                    'seq_len': 128,
                    'head_size': 128,
                    'input_layout': 'BNSD',
                    'sparse_mode': 0
                }
            },
            {
                'name': 'causal_mask_bnsd',
                'params': {
                    'batch_size': 4,
                    'num_heads': 32,
                    'seq_len': 128,
                    'head_size': 128,
                    'input_layout': 'BNSD',
                    'sparse_mode': 3 # right down causal
                }
            },
            {
                'name': 'large_seq_bnsd',
                'params': {
                    'batch_size': 1,
                    'num_heads': 16,
                    'seq_len': 2048,
                    'head_size': 128,
                    'input_layout': 'BNSD',
                    'sparse_mode': 0
                }
            },
            {
                'name': 'tnd_layout_basic',
                'params': {
                    'batch_size': 4,
                    'num_heads': 32,
                    'seq_len': 128,
                    'head_size': 128,
                    'input_layout': 'TND',
                    'sparse_mode': 3
                }
            },
            {
                'name': 'tnd_layout_with_block_table',
                'params': {
                    'batch_size': 2,
                    'num_heads': 16,
                    'seq_len': 256,
                    'head_size': 128,
                    'input_layout': 'TND',
                    'sparse_mode': 3,
                    'use_block_table': True,
                    'block_size': 128
                }
            },
            {
                'name': 'tnd_layout_variable_seq_len',
                'params': {
                    'batch_size': 4,
                    'num_heads': 16,
                    'seq_len': 128, # Max seq len
                    'head_size': 128,
                    'input_layout': 'TND',
                    'sparse_mode': 3,
                    'variable_seq_lengths': True,
                    'min_seq_len': 32
                }
            }
        ]
        
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        return [
            {
                'name': 'quick_test_bnsd',
                'params': {
                    'batch_size': 2,
                    'num_heads': 8,
                    'seq_len': 64,
                    'head_size': 64,
                    'input_layout': 'BNSD',
                    'sparse_mode': 0
                }
            }
        ]

    def _select_device(self) -> str:
        # Preserve the existing NPU preference, then use CUDA when this suite
        # is run on a GPU-only host such as H20.
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return "npu:0"
        except (ImportError, RuntimeError, AttributeError):
            pass

        if torch.cuda.is_available():
            return "cuda:0"
        raise RuntimeError(
            "FlashAttention TFLOPS mode requires an available NPU or CUDA "
            "device; neither torch_npu nor torch.cuda is available"
        )

    @staticmethod
    def _device_name(device: str) -> str:
        if device.startswith("cuda"):
            return torch.cuda.get_device_name(torch.device(device))
        if device.startswith("npu"):
            try:
                import torch_npu
                return torch_npu.npu.get_device_name(int(device.split(":")[-1]))
            except (ImportError, RuntimeError, AttributeError, ValueError):
                return device
        return device

    @staticmethod
    def _cuda_benchmark_data(
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        causal: bool,
        device: str,
        precision_type,
    ) -> Dict[str, Any]:
        """Allocate the base Q/K/V set directly in the measured CUDA dtype.

        Generating directly on CUDA avoids a multi-GB FP32 host allocation at
        N_CTX=16384. Each formal provider still creates the framework-required
        fresh device copies for every warmup and measured invocation.
        """
        shape = (batch_size, num_heads, seq_len, head_dim)
        query = torch.randn(shape, dtype=precision_type.value, device=device)
        key = torch.randn(shape, dtype=precision_type.value, device=device)
        value = torch.randn(shape, dtype=precision_type.value, device=device)
        sparse_mode = 3 if causal else 0
        return {
            "query": query,
            "key": key,
            "value": value,
            "num_heads": num_heads,
            "num_kv_heads": num_heads,
            "head_size": head_dim,
            "input_layout": "BNSD",
            "sparse_mode": sparse_mode,
            "metadata": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "num_kv_heads": num_heads,
                "head_size": head_dim,
                "sparse_mode": sparse_mode,
                "test_name": "flash_attention_tflops",
            },
        }

    def run_tflops_test(
        self,
        precision: str = "fp16",
        num_warmup: int = 5,
        num_iterations: int = 10,
        num_repeats: int = 3,
        n_ctx_values: List[int] = None,
        head_dims: List[int] = None,
        causal_values: List[bool] = None,
        device: str = "auto",
        provider: str = None,
        plot_results: bool = True,
        quick: bool = False,
        shard_index: int = 0,
        num_shards: int = 1,
    ):
        from operator_test_framework import PrecisionType

        precision_map = {
            "fp16": PrecisionType.FP16,
            "bf16": PrecisionType.BF16,
        }
        if precision not in precision_map:
            raise ValueError(f"unsupported precision: {precision}")
        precision_type = precision_map[precision]

        for count_name, count_value, minimum in (
            ("num_warmup", num_warmup, 0),
            ("num_iterations", num_iterations, 1),
            ("num_repeats", num_repeats, 1),
        ):
            if (
                not isinstance(count_value, int)
                or isinstance(count_value, bool)
                or count_value < minimum
            ):
                comparator = ">=" if minimum == 0 else ">"
                boundary = minimum if minimum == 0 else minimum - 1
                raise ValueError(
                    f"{count_name} must be a non-bool int {comparator} "
                    f"{boundary}"
                )
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(
                "shard index must satisfy 0 <= index < num_shards"
            )

        if device == "auto":
            device = self._select_device()
        elif device == "cuda":
            device = "cuda:0"
        elif device == "npu":
            device = "npu:0"
        if not device.startswith(("cuda", "npu")):
            raise ValueError(f"unsupported FlashAttention device: {device}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")

        device_name = self._device_name(device)
        implementations = self.operator_test.get_formal_implementations(device)
        expected_provider_count = 2 if device.startswith("cuda") else 1
        if len(implementations) != expected_provider_count:
            raise RuntimeError(
                f"expected {expected_provider_count} formal FlashAttention "
                f"provider(s) for {device}, got {implementations}"
            )
        formal_implementations = tuple(implementations)
        if provider is not None:
            provider_aliases = {
                "pytorch_sdpa_flash_attention": (
                    "cuda_sdpa_flash_attention"
                ),
                "flash_attn.flash_attn_func": "cuda_flash_attn_func",
                "npu_fused_infer_attention_score": "npu_flash_attention",
            }
            selected_provider = provider_aliases.get(provider, provider)
            if selected_provider not in implementations:
                raise ValueError(
                    f"provider {provider!r} is not formal for {device}; "
                    f"available={implementations}"
                )
            implementations = [selected_provider]

        if n_ctx_values is None:
            n_ctx_values = [2 ** i for i in range(10, 15)]
        if head_dims is None:
            head_dims = [64, 128]
        if causal_values is None:
            causal_values = [True, False]
        n_ctx_values = list(n_ctx_values)
        head_dims = list(head_dims)
        causal_values = list(causal_values)
        if not n_ctx_values or any(
            not isinstance(n_ctx, int)
            or isinstance(n_ctx, bool)
            or n_ctx <= 0
            for n_ctx in n_ctx_values
        ):
            raise ValueError(f"N_CTX values must be positive: {n_ctx_values}")
        if not head_dims or any(
            not isinstance(head_dim, int)
            or isinstance(head_dim, bool)
            or head_dim <= 0
            for head_dim in head_dims
        ):
            raise ValueError(f"head dimensions must be positive: {head_dims}")
        if not causal_values or any(
            not isinstance(causal, bool) for causal in causal_values
        ):
            raise ValueError(
                f"causal values must be booleans: {causal_values}"
            )

        benchmark_cases = []
        for head_dim in head_dims:
            for causal in causal_values:
                benchmark_cases.append({
                    "batch_size": 4,
                    "num_heads": 32,
                    "head_dim": head_dim,
                    "causal": causal,
                })

        result_dir = Path(self.framework.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device_tag = device.replace(":", "_")
        csv_file = result_dir / (
            f"flashattention_tflops_{precision}_{device_tag}_{timestamp}.csv"
        )
        plot_file = result_dir / (
            f"flashattention_tflops_curve_{precision}_{device_tag}_"
            f"{timestamp}.png"
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
            "point_index", "shard_index", "num_shards", "head_dim",
            "causal", "n_ctx", "provider", "device", "device_name",
            "precision", "avg_time_ms", "TFLOPS", "status", "error",
            *provenance_fields,
        ]

        results = []
        failures = []
        print(f"\n{'='*80}")
        print(
            "FlashAttention Forward TFLOPS 测试 "
            f"({device_name}, {device}) - Precision: {precision.upper()}"
        )
        print(f"Providers: {', '.join(implementations)} (no fallback)")
        print(f"Protocol: W{num_warmup}/I{num_iterations}/R{num_repeats}")
        print(f"{'='*80}")
        print(
            f"{'D':>6} {'Causal':>8} {'N_CTX':>8} {'Provider':>28} "
            f"{'Time(ms)':>12} {'TFLOPS':>12}"
        )

        def checkpoint() -> None:
            with csv_file.open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        measured_curves = {
            (implementation, case["head_dim"], case["causal"]): {}
            for implementation in implementations
            for case in benchmark_cases
        }
        shape_points_per_provider = len(benchmark_cases) * len(n_ctx_values)
        for implementation in implementations:
            implementation_index = formal_implementations.index(
                implementation
            )
            for case_index, case in enumerate(benchmark_cases):
                head_dim = case["head_dim"]
                causal = case["causal"]
                curve_key = (implementation, head_dim, causal)
                for n_ctx_index, n_ctx in enumerate(n_ctx_values):
                    point_index = (
                        implementation_index * shape_points_per_provider
                        + case_index * len(n_ctx_values)
                        + n_ctx_index
                    )
                    if quick and n_ctx_index != 0:
                        continue
                    if point_index % num_shards != shard_index:
                        continue
                    sparse_mode = 3 if causal else 0
                    test_data = None
                    row = {
                        "point_index": point_index,
                        "shard_index": shard_index,
                        "num_shards": num_shards,
                        "head_dim": head_dim,
                        "causal": causal,
                        "n_ctx": n_ctx,
                        "provider": implementation,
                        "device": device,
                        "device_name": device_name,
                        "precision": precision.upper(),
                        "avg_time_ms": "",
                        "TFLOPS": "",
                        "status": "pending",
                        "error": "",
                        "warmup": num_warmup,
                        "iterations": num_iterations,
                        "repeats": num_repeats,
                    }
                    try:
                        if device.startswith("cuda"):
                            test_data = self._cuda_benchmark_data(
                                batch_size=case["batch_size"],
                                num_heads=case["num_heads"],
                                seq_len=n_ctx,
                                head_dim=head_dim,
                                causal=causal,
                                device=device,
                                precision_type=precision_type,
                            )
                        else:
                            test_data = self.operator_test.generate_test_data(
                                batch_size=case["batch_size"],
                                num_heads=case["num_heads"],
                                seq_len=n_ctx,
                                head_size=head_dim,
                                input_layout="BNSD",
                                sparse_mode=sparse_mode,
                            )

                        metrics = (
                            self.framework
                            .run_core_operator_performance_test_v2(
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
                        row.update(
                            self.framework.performance_provenance(metrics)
                        )
                        row["avg_time_ms"] = metrics.avg_time_ms
                        tflops = self.operator_test.calculate_tflops(
                            test_data, metrics.avg_time_ms, mode="fwd"
                        )
                        if (
                            tflops is None
                            or not math.isfinite(tflops)
                            or tflops <= 0
                        ):
                            raise RuntimeError(
                                f"invalid calculated TFLOPS: {tflops}"
                            )

                        measured_curves[curve_key][n_ctx] = tflops
                        row.update(
                            TFLOPS=tflops,
                            status="ok",
                        )
                        print(
                            f"{head_dim:6d} {str(causal):>8} {n_ctx:8d} "
                            f"{implementation[:28]:>28} "
                            f"{metrics.avg_time_ms:12.3f} {tflops:12.3f}"
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        failures.append(
                            (implementation, head_dim, causal, n_ctx, error)
                        )
                        row.update(status="error", error=error)
                        print(
                            f"{head_dim:6d} {str(causal):>8} {n_ctx:8d} "
                            f"{implementation[:28]:>28} "
                            f"{'FAILED':>12} {'-':>12} ({error})"
                        )
                    results.append(row)
                    checkpoint()

        print(f"\n✅ 测试结果已保存至 {csv_file}")

        successful_rows = [
            row for row in results if row["status"] == "ok"
        ]
        wrote_plot = False
        if plot_results and successful_rows:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            colors = {
                implementation: f"C{index}"
                for index, implementation in enumerate(implementations)
            }
            linestyles = {True: "-", False: "--"}
            plt.figure(figsize=(7 * len(head_dims), 6))
            for index, head_dim in enumerate(head_dims, start=1):
                plt.subplot(1, len(head_dims), index)
                for implementation in implementations:
                    for causal in causal_values:
                        measured = [
                            measured_curves[
                                (implementation, head_dim, causal)
                            ].get(n_ctx, float("nan"))
                            for n_ctx in n_ctx_values
                        ]
                        plt.plot(
                            n_ctx_values,
                            measured,
                            marker="o",
                            color=colors[implementation],
                            linestyle=linestyles[causal],
                            linewidth=2,
                            label=(
                                f"{implementation} causal={causal}"
                            ),
                        )

                plt.xscale("log", base=2)
                plt.xticks(
                    n_ctx_values,
                    [str(value) for value in n_ctx_values],
                    rotation=20,
                )
                plt.xlabel("N_CTX")
                plt.ylabel("TFLOPS")
                plt.title(
                    f"{device_name} FlashAttention FWD "
                    f"({precision.upper()}, B=4, H=32, D={head_dim})"
                )
                plt.grid(True, which="both", ls="-", alpha=0.4)
                plt.legend(fontsize=7)

            plt.tight_layout()
            plt.savefig(plot_file, dpi=160)
            plt.close()
            wrote_plot = True
            print(f"✅ TFLOPS曲线已保存至 {plot_file}")

        if failures:
            preview = "; ".join(
                f"provider={implementation}, D={head_dim}, "
                f"causal={causal}, N_CTX={n_ctx}: {error}"
                for implementation, head_dim, causal, n_ctx, error
                in failures[:3]
            )
            raise RuntimeError(
                f"{len(failures)}/{len(results)} FlashAttention measurements "
                f"failed ({len(successful_rows)} succeeded); checkpoint "
                f"retained at {csv_file}. First failures: {preview}"
            )

        return {
            "csv_file": str(csv_file),
            "plot_file": str(plot_file) if wrote_plot else None,
            "rows": results,
        }

def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean (true/false), got {value!r}"
    )


def main(argv=None):
    """主函数 - 支持独立运行FlashAttention算子测试"""
    
    parser = argparse.ArgumentParser(description="FlashAttention算子测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["accuracy", "performance", "profile", "comprehensive", "tflops"],
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
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="tflops模式使用的精度"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="tflops模式预热次数"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="tflops模式测量次数"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="tflops模式独立重复次数"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="tflops模式设备: auto, cuda, npu, cuda:N, npu:N"
    )
    parser.add_argument(
        "--provider",
        help="仅运行指定的formal provider（默认运行设备上的全部provider）"
    )
    parser.add_argument(
        "--n-ctx-values",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="tflops模式测试的序列长度列表"
    )
    parser.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        default=[64, 128],
        help="tflops模式测试的head dimension列表"
    )
    parser.add_argument(
        "--causal-values",
        type=_parse_bool,
        nargs="+",
        default=[True, False],
        help="tflops模式测试的causal矩阵，例如: true false"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不导入matplotlib或生成PNG"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    
    args = parser.parse_args(argv)
    
    # 设置测试框架
    framework = OperatorTestFramework(result_dir=args.result_dir)
    
    # 创建并设置FlashAttention测试套件
    suite = FlashAttentionTestSuite()
    suite.setup(framework)
    
    try:
        # 根据模式运行测试
        if args.mode == "accuracy":
            suite.run_accuracy_test()
        elif args.mode == "performance":
            suite.run_performance_test()
        elif args.mode == "profile":
            suite.run_profile_test()
        elif args.mode == "comprehensive":
            suite.run_comprehensive_test()
        elif args.mode == "tflops":
            suite.run_tflops_test(
                precision=args.precision,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                num_repeats=args.repeats,
                n_ctx_values=args.n_ctx_values,
                head_dims=args.head_dims,
                causal_values=args.causal_values,
                device=args.device,
                provider=args.provider,
                plot_results=not args.no_plot,
                quick=args.quick,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
    except Exception as exc:
        print(f"❌ FlashAttention算子测试失败: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n✓ FlashAttention算子测试完成，结果已保存到 {args.result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
