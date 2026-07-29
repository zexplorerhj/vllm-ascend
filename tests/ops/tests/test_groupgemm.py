"""
GroupGemm算子测试套件
通用的 GroupGemm 测试，主要测试 num_experts=8 的场景
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from operator_test_framework import (
    PERFORMANCE_PROVENANCE_FIELDS,
    PrecisionType,
    build_curve_selection_provenance,
    finalize_curve_coverage,
)
from groupgemm.groupgemm_int8 import GroupGemmOperatorTest
from groupgemm.groupgemm_bf16 import GroupGemmBF16OperatorTest
from groupgemm.groupgemm_fp8 import GroupGemmFp8OperatorTest
from groupgemm.groupgemm_fp8_npu import GroupGemmFp8NpuOperatorTest
from groupgemm.groupgemm_mxfp8_npu import (
    GroupGemmMxFp8NpuOperatorTest,
)

GROUPGEMM_PRECISION_TYPES = {
    "int8": PrecisionType.INT8,
    "bf16": PrecisionType.BF16,
    "fp8": PrecisionType.FP8,
    "mxfp8": PrecisionType.MXFP8,
}


class GroupGemmTestSuite(BaseTestSuite):
    """GroupGemm算子测试套件 - 支持INT8、BF16、FP8和MXFP8精度"""
    
    def __init__(
        self,
        precision: str = "int8",
        num_experts: int = 8,
        hidden_dim: int = 7168,
        out_channel: int = 4096,
        use_nz_format: bool = False,
        device: str = "auto",
    ):
        """
        初始化GroupGemm测试套件
        
        Args:
            precision: 精度类型，"int8"、"bf16"、"fp8" 或 "mxfp8"
            num_experts: 专家数量
            hidden_dim: 隐藏维度
            out_channel: 输出通道数
            use_nz_format: 是否使用NZ格式（仅对INT8有效）
        """
        precision = precision.lower()
        if precision not in GROUPGEMM_PRECISION_TYPES:
            raise ValueError(f"unsupported GroupGemm precision: {precision}")
        format_suffix = "_NZ" if use_nz_format else ""
        if precision == "bf16":
            precision_name = f"GroupGemm_BF16{format_suffix}"
        elif precision == "fp8":
            precision_name = "GroupGemm_FP8"
        elif precision == "mxfp8":
            precision_name = "GroupGemm_MXFP8"
        else:
            precision_name = f"GroupGemm{format_suffix}"
        super().__init__(precision_name)
        self.precision = precision
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.out_channel = out_channel
        self.use_nz_format = use_nz_format
        self.requested_device = device
        self.operator_test = self._new_operator_for_device(device)

    def _operator_type_for_device(self, device: str):
        if self.precision == "fp8":
            if device and device.startswith("npu"):
                return GroupGemmFp8NpuOperatorTest
            return GroupGemmFp8OperatorTest
        if self.precision == "mxfp8":
            return GroupGemmMxFp8NpuOperatorTest
        if self.precision == "bf16":
            return GroupGemmBF16OperatorTest
        return GroupGemmOperatorTest

    def _new_operator_for_device(self, device: str):
        operator_type = self._operator_type_for_device(device)
        return operator_type(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            out_channel=self.out_channel,
            use_nz_format=self.use_nz_format,
        )

    def _select_operator_for_device(self, device: str) -> None:
        managed_types = (
            GroupGemmOperatorTest,
            GroupGemmBF16OperatorTest,
            GroupGemmFp8OperatorTest,
            GroupGemmFp8NpuOperatorTest,
            GroupGemmMxFp8NpuOperatorTest,
        )
        if not isinstance(self.operator_test, managed_types):
            return
        operator_type = self._operator_type_for_device(device)
        if type(self.operator_test) is not operator_type:
            self.operator_test = self._new_operator_for_device(device)
    
    def register_operator(self):
        """注册GroupGemm算子到测试框架"""
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建标准测试案例 - GroupGemm主要使用Profile测试"""
        return self.create_profile_test_cases()
    
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例 - 使用较小的序列长度"""
        suffix = f"_{self.precision}" if self.precision == "bf16" else ""
        return [
            {
                'name': f'quick_profile2048{suffix}',
                'params': {
                    'seq_len': 2048,
                    'num_experts': self.num_experts
                }
            }
        ]
    
    def create_profile_test_cases(self) -> List[Dict[str, Any]]:
        """创建标准测试案例 - GroupGemm主要使用Profile测试"""
        return self.create_profile_test_cases()
    
    def create_profile_test_cases(self) -> List[Dict[str, Any]]:
        """创建专门的Profile测试案例"""
        suffix = f"_{self.precision}" if self.precision == "bf16" else ""
        return [
            {
                'name': f'profile2048{suffix}',
                'params': {
                    'seq_len': 2048,
                    'num_experts': self.num_experts
                }
            },
            {
                'name': f'profile4096{suffix}',
                'params': {
                    'seq_len': 4096,
                    'num_experts': self.num_experts
                }
            },
            {
                'name': f'profile8192{suffix}',
                'params': {
                    'seq_len': 8192,
                    'num_experts': self.num_experts
                }
            },
            {
                'name': f'profile16384{suffix}',
                'params': {
                    'seq_len': 16384,
                    'num_experts': self.num_experts
                }
            },
            {
                'name': f'profile32768{suffix}',
                'params': {
                    'seq_len': 32768,
                    'num_experts': self.num_experts
                }
            },
        ]
    
    def run_tflops_test(
        self,
        seq_lens=None,
        num_experts=None,
        hidden_dim=None,
        out_channel=None,
        device="auto",
        num_warmup=10,
        num_iterations=30,
        num_repeats=3,
        plot_results=True,
        quick=False,
        shard_index=0,
        num_shards=1,
    ):
        """Run the formal pure-GEMM throughput curve with Framework V2."""
        import csv
        import math
        import time

        import torch
        from operator_test_framework import PrecisionType

        # 参数默认值
        if num_experts is None:
            num_experts = self.num_experts
        if hidden_dim is None:
            hidden_dim = self.hidden_dim
        if out_channel is None:
            out_channel = self.out_channel
        if num_experts <= 0 or hidden_dim <= 0 or out_channel <= 0:
            raise ValueError(
                "num_experts, hidden_dim 和 out_channel 必须都大于 0"
            )
        if (
            num_warmup < 0
            or num_iterations <= 0
            or num_repeats <= 0
        ):
            raise ValueError("W >= 0 and I/R > 0 are required")
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(
                "shard index must satisfy 0 <= index < num_shards"
            )
        npu_available = False
        try:
            import torch_npu
            npu_api = getattr(torch, "npu", getattr(torch_npu, "npu", None))
            npu_available = bool(npu_api and npu_api.is_available())
        except ImportError:
            pass

        if device in (None, "auto"):
            if self.precision == "fp8" and torch.cuda.is_available():
                device = "cuda:0"
            elif npu_available:
                device = "npu:0"
            elif self.precision == "mxfp8":
                raise RuntimeError(
                    "MXFP8 GroupGemm TFLOPS 测试需要 NPU Ascend 950PR"
                )
            elif torch.cuda.is_available():
                device = "cuda:0"
            else:
                raise RuntimeError("GroupGemm TFLOPS 测试需要 NPU 或 CUDA GPU")
        elif self.precision == "mxfp8" and device.startswith("cuda"):
            raise RuntimeError(
                "MXFP8 GroupGemm TFLOPS 测试需要 NPU Ascend 950PR"
            )
        elif device.startswith("npu") and not npu_available:
            raise RuntimeError(f"请求了 {device}，但 NPU 不可用")
        elif device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"请求了 {device}，但 CUDA 不可用")
        elif not (device.startswith("npu") or device.startswith("cuda")):
            raise ValueError(f"GroupGemm TFLOPS 测试不支持设备 {device}")

        self._select_operator_for_device(device)
        implementations = self.operator_test.get_formal_implementations(device)
        if len(implementations) != 1:
            raise RuntimeError(
                f"expected one formal {self.precision.upper()} GroupGemm "
                f"provider for {device}, got {implementations}"
            )
        implementation = implementations[0]

        formal_seq_lens = [
            64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
        ]
        if seq_lens is None:
            seq_lens = formal_seq_lens
        else:
            seq_lens = list(seq_lens)
        if not seq_lens or any(value <= 0 for value in seq_lens):
            raise ValueError(f"seq_lens 必须是非空正整数列表: {seq_lens}")
        indexed_seq_lens = list(enumerate(seq_lens))
        total_requested_points = len(indexed_seq_lens)
        if quick:
            indexed_seq_lens = indexed_seq_lens[:1]
        indexed_seq_lens = [
            item for item in indexed_seq_lens
            if item[0] % num_shards == shard_index
        ]
        coverage_selected_points = len(indexed_seq_lens)
        selection_provenance = build_curve_selection_provenance(
            quick=quick,
            num_shards=num_shards,
            total_formal_points=len(formal_seq_lens),
            total_requested_points=total_requested_points,
            selected_points=coverage_selected_points,
            uses_formal_shape_matrix=(
                seq_lens == formal_seq_lens
                and num_experts == 8
                and hidden_dim == 7168
                and out_channel == 4096
                and not self.use_nz_format
            ),
        )

        metric_name = {
            "int8": "INT8_TOPS",
            "bf16": "BF16_TFLOPS",
            "fp8": "FP8_TFLOPS",
            "mxfp8": "MXFP8_TFLOPS",
        }[self.precision]

        # 确定精度类型
        precision_type = GROUPGEMM_PRECISION_TYPES[self.precision]

        results = []
        failures = []
        result_dir = self.framework.result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_file = result_dir / (
            f"groupgemm_tflops_{self.precision}_"
            f"{device.replace(':', '_')}_{timestamp}.csv"
        )
        plot_file = result_dir / (
            f"groupgemm_tflops_curve_{self.precision}_"
            f"{device.replace(':', '_')}_{timestamp}.png"
        )
        provenance_fields = list(PERFORMANCE_PROVENANCE_FIELDS)
        fieldnames = [
            "point_index", "shard_index", "num_shards", "selection_mode",
            "shape_matrix_source", "coverage_mode",
            "coverage_total_formal_points", "coverage_selected_points",
            "coverage_total_requested_points",
            "selection_covers_full_formal_matrix", "coverage_complete",
            "seq_len",
            "num_experts", "hidden_dim", "out_channel", "device", "precision",
            "implementation", "kernel", "output_semantics",
            "avg_time_ms", "metric", "throughput_trillion_ops_s",
            "status", "error", *provenance_fields,
        ]

        for point_index, seq_len in indexed_seq_lens:
            kernel = ""
            output_semantics = ""
            row = {
                "point_index": point_index,
                "shard_index": shard_index,
                "num_shards": num_shards,
                **selection_provenance,
                "seq_len": seq_len,
                "num_experts": num_experts,
                "hidden_dim": hidden_dim,
                "out_channel": out_channel,
                "device": device,
                "precision": self.precision.upper(),
                "implementation": implementation,
                "kernel": kernel,
                "output_semantics": output_semantics,
                "avg_time_ms": "",
                "metric": metric_name,
                "throughput_trillion_ops_s": "",
                "status": "pending",
                "error": "",
                "warmup": num_warmup,
                "iterations": num_iterations,
                "repeats": num_repeats,
            }
            try:
                test_data = self.operator_test.generate_test_data(
                    seq_len=seq_len,
                    num_experts=num_experts,
                    hidden_dim=hidden_dim,
                    out_channel=out_channel
                )
                test_data["benchmark_implementation"] = implementation

                if implementation == self.operator_test.CUDA_BF16_IMPLEMENTATION:
                    group_rows = test_data["group_list"].tolist()
                    kernel = (
                        "torch_bmm_cublas" if len(set(group_rows)) == 1
                        else "torch_grouped_mm_cutlass"
                    )
                    output_semantics = "BF16xBF16->BF16,no_bias"
                elif implementation == self.operator_test.CUDA_INT8_IMPLEMENTATION:
                    kernel = "vllm_cutlass_scaled_mm"
                    output_semantics = (
                        "INT8xINT8,per-token*per-channel-scale->BF16"
                    )
                elif (
                    self.precision == "fp8"
                    and implementation
                    == getattr(
                        self.operator_test,
                        "CUDA_IMPLEMENTATION",
                        None,
                    )
                ):
                    kernel = "vllm_cutlass_scaled_mm_expert_loop"
                    output_semantics = (
                        "FP8(E4M3)xFP8(E4M3),"
                        "per-token*per-channel-scale->BF16"
                    )
                elif (
                    self.precision == "fp8"
                    and implementation
                    == getattr(
                        self.operator_test,
                        "NPU_FP8_IMPLEMENTATION",
                        None,
                    )
                ):
                    kernel = "npu_grouped_matmul"
                    output_semantics = (
                        "FP8(E4M3)xFP8(E4M3),"
                        "per-token-FP32*per-channel-FP32-scale"
                        "->BF16,no_bias"
                    )
                elif (
                    self.precision == "mxfp8"
                    and implementation
                    == getattr(
                        self.operator_test,
                        "NPU_MXFP8_IMPLEMENTATION",
                        None,
                    )
                ):
                    kernel = "npu_grouped_matmul"
                    output_semantics = (
                        "MXFP8(E4M3,group32)xMXFP8(E4M3,group32),"
                        "per-group-E8M0-scale->BF16,no_bias,pure-GMM2"
                    )
                else:
                    kernel = "npu_grouped_matmul"
                    output_semantics = (
                        "INT8xINT8,scaled->BF16,no_bias"
                        if self.precision == "int8"
                        else "BF16xBF16->BF16,no_bias"
                    )
                row.update(
                    kernel=kernel,
                    output_semantics=output_semantics,
                )
                metrics = self.framework.run_core_operator_performance_test_v2(
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
                trillion_ops = (
                    metrics.throughput / 1000.0
                    if metrics.throughput is not None else None
                )
                if (trillion_ops is None or not math.isfinite(trillion_ops)
                        or trillion_ops <= 0):
                    raise RuntimeError(
                        f"得到无效 {metric_name}: {trillion_ops}"
                    )

                row.update(
                    self.framework.performance_provenance(metrics)
                )
                row.update(
                    avg_time_ms=metrics.avg_time_ms,
                    throughput_trillion_ops_s=trillion_ops,
                    status="success",
                )
            except Exception as exc:
                failures.append((seq_len, exc))
                row.update(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(row)
            with csv_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        if finalize_curve_coverage(results):
            with csv_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        successful_rows = [
            row for row in results if row["status"] == "success"
        ]
        if plot_results and successful_rows:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(figsize=(12, 7))
            ax.plot(
                [row["seq_len"] for row in successful_rows],
                [
                    row["throughput_trillion_ops_s"]
                    for row in successful_rows
                ],
                'bo-',
                linewidth=2,
                markersize=6,
                label=f'GroupGemm ({self.precision.upper()}, {implementation})',
            )

            ax.set_xlabel('Total Tokens (seq_len = sum of M per expert)', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(
                f'GroupGemm {metric_name} vs seq_len ({device})\n'
                f'num_experts={num_experts}, K={hidden_dim}, N={out_channel}',
                fontsize=13
            )
            ax.set_xscale('log', base=2)
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.legend(fontsize=11)

            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            plt.close()

        if failures:
            raise RuntimeError(
                f"{len(failures)} formal GroupGemm point(s) failed; "
                f"checkpoint retained at {csv_file}"
            )
        return {
            "device": device,
            "implementation": implementation,
            "csv_file": str(csv_file),
            "plot_file": str(plot_file) if plot_results else None,
            "results": results,
        }

    def run_profile_test(self, test_cases: List[Dict[str, Any]] = None, num_iterations: int = 10):
        """运行 GroupGemm Profile 测试（使用基类增强版本）"""
        if test_cases is None:
            test_cases = self.create_profile_test_cases()
        
        # 显示 GroupGemm 特有信息
        precision_display = self.precision.upper()
        data_types = {
            "int8": "x=INT8, weight=INT8, scales=FP32/BF16, output=BF16",
            "bf16": "x=BF16, weight=BF16, bias=FP32, output=BF16",
            "fp8": "x=E4M3, weight=E4M3, scales=FP32, output=BF16",
            "mxfp8": (
                "x=E4M3, weight=E4M3, group32 scales=E8M0, output=BF16"
            ),
        }
        
        print(f"GroupGemm {precision_display} 特有配置:")
        print(f"专家数量: {self.num_experts}")
        print(f"测试案例数: {len(test_cases)}")
        print(f"数据类型: {data_types.get(self.precision, 'Unknown')}")
        
        # 确定精度类型
        precision_type = GROUPGEMM_PRECISION_TYPES[self.precision]
        
        # 调用基类的增强版本，传入 GroupGemm 特有的参数
        return super().run_profile_test(
            test_cases=test_cases,
            num_iterations=num_iterations,
            precision_type=precision_type,
            filename_suffix=self.precision
        )
    

    
def main():
    """主函数 - 支持独立运行GroupGemm算子Profile测试"""
    import argparse
    from operator_test_framework import OperatorTestFramework
    
    parser = argparse.ArgumentParser(description='GroupGemm 算子 Profile 测试')
    parser.add_argument(
        '--precision',
        choices=['int8', 'bf16', 'fp8', 'mxfp8'],
        default='int8',
        help='精度类型',
    )
    parser.add_argument('--num-experts', type=int, default=8, help='专家数量')
    parser.add_argument('--hidden-dim', type=int, default=7168, help='隐藏维度')
    parser.add_argument('--out-channel', type=int, default=4096, help='输出通道')
    parser.add_argument(
        '--device',
        default='auto',
        help='TFLOPS 测试设备 (auto, npu:0, cuda:0)',
    )
    parser.add_argument('--iterations', type=int, default=10, help='迭代次数')
    parser.add_argument(
        '--tflops-warmup', type=int, default=10, help='TFLOPS 测试预热次数'
    )
    parser.add_argument(
        '--tflops-iterations', type=int, default=30, help='TFLOPS 测试计时次数'
    )
    parser.add_argument('--tflops-repeats', type=int, default=3)
    parser.add_argument('--result-dir', default='test_results')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--use-nz-format', action='store_true', help='使用NZ格式（仅对INT8有效）')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['profile', 'performance', 'tflops'],
        default='profile',
        help='测试模式: profile(Profile测试), performance(性能测试), tflops(TFLOPS曲线测试)'
    )
    parser.add_argument('--seq-lens', type=str, help='序列长度列表，用逗号分隔 (例如: 2048,4096,8192)')
    
    args = parser.parse_args()
    
    # 创建测试框架
    framework = OperatorTestFramework(result_dir=args.result_dir)
    
    # 根据精度类型创建测试套件
    test_suite = GroupGemmTestSuite(
        precision=args.precision,
        num_experts=args.num_experts,
        hidden_dim=args.hidden_dim,
        out_channel=args.out_channel,
        use_nz_format=args.use_nz_format,
        device=args.device,
    )
    
    # 设置框架并注册算子
    test_suite.setup(framework)
    
    if args.precision == 'fp8':
        print(
            "🔧 使用 FP8 E4M3 精度测试 "
            "(CUDA H20 使用 vLLM CUTLASS grouped MoE MM；NPU 950PR 使用 "
            "npu_grouped_matmul pure-GMM2；输出 BF16)"
        )
    elif args.precision == 'mxfp8':
        print(
            "🔧 使用 MXFP8 E4M3/E8M0 group32 精度测试 "
            "(NPU Ascend 950PR npu_grouped_matmul pure-GMM2，输出 BF16)"
        )
    elif args.precision == 'bf16':
        print(
            "🔧 使用 BF16 精度测试 "
            "(NPU grouped_matmul 与 CUDA balanced bmm 均为 no-bias "
            "pure-GEMM 语义)"
        )
    else:
        nz_info = " + NZ格式" if args.use_nz_format else ""
        print(
            f"🔧 使用 INT8 精度测试{nz_info} "
            "(NPU 路径含缩放；CUDA 优先 vLLM CUTLASS scaled-mm，"
            "输出 BF16)"
        )
    
    exit_code = 0
    try:
        # 创建自定义测试案例（如果指定了序列长度）
        test_cases = None
        if args.seq_lens:
            seq_lens = [int(x.strip()) for x in args.seq_lens.split(',')]
            suffix = f"_{args.precision}" if args.precision == "bf16" else ""
            test_cases = []
            for seq_len in seq_lens:
                test_cases.append({
                    'name': f'profile{seq_len}{suffix}',
                    'params': {
                        'seq_len': seq_len,
                        'num_experts': args.num_experts,
                        'hidden_dim': args.hidden_dim,
                        'out_channel': args.out_channel
                    }
                })
            print(f"📋 使用自定义序列长度: {seq_lens}")
        else:
            print(f"📋 使用默认Profile测试案例")
        
        # 根据模式运行相应测试
        if args.mode == "profile":
            print("🚀 运行 GroupGemm Profile 测试...")
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
            print("🚀 运行 GroupGemm 性能测试...")
            # 确定精度类型
            precision_type = GROUPGEMM_PRECISION_TYPES[args.precision]
            results = test_suite.run_performance_test(
                test_cases=test_cases,
                precision_type=precision_type
            )

            print(f"\n{'='*60}")
            print("✅ 性能测试完成！")
            print(f"📁 结果已保存到 test_results 目录")
            print(f"📊 成功测试: {results['summary']['successful_tests']}/{results['summary']['total_tests']}")
            print(f"{'='*60}")

        elif args.mode == "tflops":
            print("📈 运行 GroupGemm TFLOPS 测试...")
            # 如果指定了自定义序列长度，则使用自定义的
            custom_seq_lens = None
            if args.seq_lens:
                custom_seq_lens = [int(x.strip()) for x in args.seq_lens.split(',')]
            test_suite.run_tflops_test(
                seq_lens=custom_seq_lens,
                num_experts=args.num_experts,
                hidden_dim=args.hidden_dim,
                out_channel=args.out_channel,
                device=args.device,
                num_warmup=args.tflops_warmup,
                num_iterations=args.tflops_iterations,
                num_repeats=args.tflops_repeats,
                plot_results=not args.no_plot,
                quick=args.quick,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    finally:
        # 清理当前可用的 accelerator 资源。
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                print("🧹 CUDA 资源清理完成")

            try:
                import torch_npu
                npu_api = getattr(torch, "npu", getattr(torch_npu, "npu", None))
                if npu_api and npu_api.is_available():
                    npu_api.synchronize()
                    npu_api.empty_cache()
                    print("🧹 NPU 资源清理完成")
            except ImportError:
                pass
        except Exception as cleanup_error:
            print(f"⚠️ 资源清理时出错: {cleanup_error}")

    # 不再用 finally 中的 sys.exit(0) 覆盖性能测试失败。
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
