"""
PagedAttention算子测试套件
"""

import sys
import os
import random
import math
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from tests.base_test_suite import BaseTestSuite
from operator_test_framework import (
    PERFORMANCE_PROVENANCE_FIELDS,
    build_curve_selection_provenance,
    finalize_curve_coverage,
)
from paged_attention.base import PagedAttentionOperatorTest


class PagedAttentionTestSuite(BaseTestSuite):
    """PagedAttention算子测试套件"""
    
    def __init__(self):
        super().__init__("paged_attention")
        self.operator_test = PagedAttentionOperatorTest()
    
    def register_operator(self):
        """注册PagedAttention算子到测试框架"""
        self.framework.register_operator(self.operator_test)
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """创建PagedAttention算子测试案例"""
        return [
            # qwen3-32b tp8
            {
                'name': 'qwen3-32b_tp8_1k',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 1024,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-32b_tp8_10k',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 10240,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-32b_tp8_30k',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 32768,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-235b-tp4dpx_8k',
                'params': {
                    'batch_size': 12,
                    'num_heads': 16,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 8192,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-235b-tp1dpx_8k',
                'params': {
                    'batch_size': 12,
                    'num_heads': 64,
                    'num_kv_heads': 4,
                    'head_size': 128,
                    'max_seq_len': 8192,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen2.5 72b-tp8_4k',
                'params': {
                    'batch_size': 256,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 4096,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            }
        ]
    
    def create_full_test_cases(self, num_cases: int = 50) -> List[Dict[str, Any]]:
        """创建全面测试案例，使用随机参数生成
        
        参数范围：
        - batch_size: 1-512
        - num_heads: [8, 16, 64]
        - num_kv_heads: [1, 4]
        - head_size: 128 (固定)
        - max_seq_len: 1-65536 (64k)
        - num_blocks: 9000-13000
        - block_size: 128 (固定)
        """
        test_cases = []
        
        # 可选的num_heads和num_kv_heads值
        num_heads_options = [8, 16, 64]
        num_kv_heads_options = [1, 4]
        
        for i in range(num_cases):
            # 随机生成参数
            batch_size = random.randint(1, 512)
            num_heads = random.choice(num_heads_options)
            num_kv_heads = random.choice(num_kv_heads_options)
            head_size = 128  # 固定值
            max_seq_len = random.randint(1, 65536)  # 1到64k
            num_blocks = random.randint(9000, 13000)
            block_size = 128  # 固定值
            
            test_case = {
                'name': f'full_test_case_{i+1}',
                'params': {
                    'batch_size': batch_size,
                    'num_heads': num_heads,
                    'num_kv_heads': num_kv_heads,
                    'head_size': head_size,
                    'max_seq_len': max_seq_len,
                    'num_blocks': num_blocks,
                    'block_size': block_size
                }
            }
            test_cases.append(test_case)
        
        return test_cases
    
    def create_quick_test_cases(self) -> List[Dict[str, Any]]:
        """创建快速测试案例"""
        return [
            {
                'name': 'quick_medium',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 4096,
                    'num_blocks': 10000,
                    'block_size': 128
                }
            }
        ]
    
    def print_test_cases_summary(self, test_cases: List[Dict[str, Any]]):
        """打印测试案例的参数统计摘要"""
        if not test_cases:
            return
        
        print(f"\n📊 测试案例参数统计 (共 {len(test_cases)} 个案例):")
        print("-" * 60)
        
        # 收集所有参数值
        batch_sizes = [case['params']['batch_size'] for case in test_cases]
        num_heads_list = [case['params']['num_heads'] for case in test_cases]
        num_kv_heads_list = [case['params']['num_kv_heads'] for case in test_cases]
        max_seq_lens = [case['params']['max_seq_len'] for case in test_cases]
        num_blocks_list = [case['params']['num_blocks'] for case in test_cases]
        
        print(f"batch_size    : {min(batch_sizes):4d} - {max(batch_sizes):4d}")
        print(f"num_heads     : {sorted(set(num_heads_list))}")
        print(f"num_kv_heads  : {sorted(set(num_kv_heads_list))}")
        print(f"head_size     : 128 (固定)")
        print(f"max_seq_len   : {min(max_seq_lens):5d} - {max(max_seq_lens):5d}")
        print(f"num_blocks    : {min(num_blocks_list):5d} - {max(num_blocks_list):5d}")
        print(f"block_size    : 128 (固定)")
        print("-" * 60)
    
    def create_profile_test_cases(self) -> List[Dict[str, Any]]:
        """创建专门的 Profile 测试案例"""
        return [
            {
                'name': 'qwen3-32b_tp8_10k_profile',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 10019,
                    'num_blocks': 9695,
                    'block_size': 128
                }
            },
            {
                'name': 'qwen3-32b_tp8_30k_profile',
                'params': {
                    'batch_size': 128,
                    'num_heads': 8,
                    'num_kv_heads': 1,
                    'head_size': 128,
                    'max_seq_len': 32768,
                    'num_blocks': 9695,
                    'block_size': 128
                }
            }
        ]

    def run_latency_plot_test(
        self,
        device: str = "auto",
        seqlen_start: int = 1024,
        seqlen_end: int = 32768,
        seqlen_step: int = 1024,
        seqlen_batch_size: int = 128,
        batch_min: int = 1,
        batch_max: int = 128,
        batch_step: int = 1,
        batch_curve_seq_lens: List[int] = None,
        num_warmup: int = 5,
        num_iterations: int = 20,
        repeats: int = 3,
        num_blocks: int = 10000,
        block_size: int = 128,
        seed: int = 0,
        allow_fallback: bool = False,
        providers: List[str] = None,
        plot_results: bool = True,
        quick: bool = False,
        shard_index: int = 0,
        num_shards: int = 1,
    ):
        import csv
        import time

        plt = None
        if plot_results:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError as exc:
                raise RuntimeError(
                    "未找到 matplotlib，无法生成延迟曲线；"
                    "可传 --no-plot 仅生成 CSV"
                ) from exc

        from operator_test_framework import PrecisionType

        if seqlen_start <= 0 or seqlen_end < seqlen_start:
            raise ValueError("seqlen range must satisfy 0 < start <= end")
        if seqlen_step <= 0:
            raise ValueError("seqlen_step must be positive")
        if seqlen_batch_size <= 0:
            raise ValueError("seqlen_batch_size must be positive")
        if batch_min <= 0 or batch_max < batch_min or batch_step <= 0:
            raise ValueError(
                "batch range must satisfy 0 < min <= max and step > 0"
            )
        if num_warmup < 0 or num_iterations <= 0 or repeats <= 0:
            raise ValueError(
                "num_warmup must be >= 0 and num_iterations/repeats > 0"
            )
        if num_shards <= 0 or not 0 <= shard_index < num_shards:
            raise ValueError(
                "shard index must satisfy 0 <= index < num_shards"
            )
        if num_blocks != 10000:
            raise ValueError(
                "formal PagedAttention curve requires num_blocks=10000"
            )
        if block_size != 128:
            raise ValueError(
                "formal PagedAttention curve requires block_size=128"
            )
        if allow_fallback:
            raise ValueError(
                "formal PagedAttention curve does not allow fallback providers"
            )

        if device == "auto":
            try:
                import torch_npu
                if torch_npu.npu.is_available():
                    device = "npu:0"
            except (ImportError, AttributeError):
                pass
            if device == "auto" and torch.cuda.is_available():
                device = "cuda:0"
            if device == "auto":
                raise RuntimeError("No available NPU or CUDA device was found")
        elif device == "cuda":
            device = "cuda:0"
        elif device == "npu":
            device = "npu:0"

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        if batch_curve_seq_lens is None:
            batch_curve_seq_lens = [10000, 30000]
        if not batch_curve_seq_lens or any(x <= 0 for x in batch_curve_seq_lens):
            raise ValueError("batch_curve_seq_lens must contain positive values")

        precision_type = PrecisionType.BF16
        formal_implementations = (
            self.operator_test.get_formal_implementations(device)
        )
        if not formal_implementations:
            raise RuntimeError(
                f"设备 {device} 上未找到 formal PagedAttention provider"
            )

        requested_providers = providers or ["preferred"]
        if requested_providers == ["preferred"]:
            implementations = formal_implementations
        elif requested_providers == ["all"]:
            implementations = formal_implementations
        else:
            unknown = sorted(
                set(requested_providers) - set(formal_implementations)
            )
            if unknown:
                raise ValueError(
                    f"设备 {device} 不支持 formal provider {unknown}; "
                    f"formal={formal_implementations}"
                )
            implementations = requested_providers

        result_dir = self.framework.result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device_tag = device.replace(":", "_")
        seqlen_csv = result_dir / (
            f"paged_attention_latency_vs_seqlen_{device_tag}_{timestamp}.csv"
        )
        batch_csv = result_dir / (
            f"paged_attention_latency_vs_batch_{device_tag}_{timestamp}.csv"
        )
        seqlen_plot = result_dir / (
            f"paged_attention_latency_vs_seqlen_{device_tag}_{timestamp}.png"
        )
        batch_plot = result_dir / (
            f"paged_attention_latency_vs_batch_{device_tag}_{timestamp}.png"
        )
        if device.startswith("cuda"):
            device_name = torch.cuda.get_device_name(torch.device(device))
        else:
            try:
                import torch_npu
                device_index = int(device.split(":", 1)[1])
                device_name = torch_npu.npu.get_device_name(device_index)
            except (ImportError, AttributeError, IndexError, ValueError):
                device_name = device
        if device.startswith("cuda"):
            try:
                import flashinfer
                provider_version = getattr(flashinfer, "__version__", "unknown")
            except (ImportError, OSError, RuntimeError):
                provider_version = "unavailable"
        else:
            try:
                import torch_npu
                provider_version = getattr(torch_npu, "__version__", "unknown")
            except (ImportError, OSError, RuntimeError):
                provider_version = "unavailable"

        seqlens = list(range(seqlen_start, seqlen_end + 1, seqlen_step))
        if seqlens[-1] != seqlen_end:
            seqlens.append(seqlen_end)
        batch_sizes = list(range(batch_min, batch_max + 1, batch_step))
        if batch_sizes[-1] != batch_max:
            batch_sizes.append(batch_max)

        print(f"\n{'='*80}")
        print("PagedAttention 延迟画图测试")
        print(
            f"设备: {device}, provider: {', '.join(implementations)}, "
            f"精度: {precision_type.name}, block_size: {block_size}, "
            f"num_blocks: {'auto' if num_blocks == 0 else num_blocks}, seed: {seed}, "
            f"W{num_warmup}/I{num_iterations}/R{repeats}"
        )
        print(f"{'='*80}")

        seqlen_latency_ms = {
            impl: [float("nan")] * len(seqlens)
            for impl in implementations
        }
        seqlen_rows = []

        def measure_one(data, implementation):
            metrics = self.framework.run_core_operator_performance_test_v2(
                operator_test=self.operator_test,
                data=data,
                device=device,
                precision=precision_type,
                implementation=implementation,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
                num_repeats=repeats,
                retain_outputs=True,
                verify_independent_storage=True,
            )
            latency = float(metrics.avg_time_ms)
            if not math.isfinite(latency) or latency <= 0:
                raise RuntimeError(f"invalid device latency: {latency} ms")
            return metrics, self.framework.performance_provenance(metrics)

        print(
            f"\n📈 曲线1: batch={seqlen_batch_size}, "
            f"seqlen={seqlen_start}..{seqlen_end}"
        )
        provenance_fields = list(PERFORMANCE_PROVENANCE_FIELDS)
        common_fields = [
            "provider", "device", "device_name", "torch_version",
            "provider_version", "precision", "num_heads", "num_kv_heads",
            "head_size", "batch_size", "block_size", "num_blocks",
            "page_pool_policy", "seed", "latency_ms", "status", "error",
        ]
        seqlen_fields = [
            *common_fields[:9], "seq_len", *common_fields[9:],
            "point_index", "shard_index", "num_shards",
            "selection_mode", "shape_matrix_source", "coverage_mode",
            "coverage_total_formal_points", "coverage_selected_points",
            "coverage_total_requested_points",
            "selection_covers_full_formal_matrix", "coverage_complete",
            "provider_selection_mode",
            *provenance_fields,
        ]
        failures = []
        seqlen_point_count = len(implementations) * len(seqlens)
        seqlen_total_formal_points = (
            len(formal_implementations) * 32
        )
        seqlen_selected_points = sum(
            1
            for implementation_index, _ in enumerate(implementations)
            for seqlen_index, _ in enumerate(seqlens)
            if (not quick or seqlen_index == 0)
            and (
                implementation_index * len(seqlens) + seqlen_index
            ) % num_shards == shard_index
        )
        providers_complete = (
            set(implementations) == set(formal_implementations)
        )
        seqlen_selection_provenance = build_curve_selection_provenance(
            quick=quick,
            num_shards=num_shards,
            total_formal_points=seqlen_total_formal_points,
            total_requested_points=len(implementations) * len(seqlens),
            selected_points=seqlen_selected_points,
            uses_formal_shape_matrix=(
                seqlens == list(range(1024, 32768 + 1, 1024))
                and seqlen_batch_size == 128
                and block_size == 128
                and num_blocks == 10000
            ),
            providers_complete=providers_complete,
        )
        for implementation_index, impl in enumerate(implementations):
            print(f"\n  provider: {impl}")
            for seqlen_index, seq_len in enumerate(seqlens):
                point_index = (
                    implementation_index * len(seqlens) + seqlen_index
                )
                if quick and seqlen_index != 0:
                    continue
                if point_index % num_shards != shard_index:
                    continue
                row = {
                    "provider": impl,
                    "device": device,
                    "device_name": device_name,
                    "torch_version": torch.__version__,
                    "provider_version": provider_version,
                    "precision": precision_type.name,
                    "num_heads": 8,
                    "num_kv_heads": 1,
                    "head_size": 128,
                    "batch_size": seqlen_batch_size,
                    "seq_len": seq_len,
                    "block_size": block_size,
                    "num_blocks": num_blocks,
                    "page_pool_policy": "",
                    "seed": seed,
                    "latency_ms": "",
                    "status": "pending",
                    "error": "",
                    "point_index": point_index,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    **seqlen_selection_provenance,
                    "provider_selection_mode": "fixed_formal_provider",
                    "warmup": num_warmup,
                    "iterations": num_iterations,
                    "repeats": repeats,
                }
                try:
                    test_data = self.operator_test.generate_latency_test_data(
                        batch_size=seqlen_batch_size,
                        num_heads=8,
                        num_kv_heads=1,
                        head_size=128,
                        max_seq_len=seq_len,
                        num_blocks=num_blocks,
                        block_size=block_size,
                        seed=seed,
                        storage_device="cpu",
                    )
                    metrics, provenance = measure_one(test_data, impl)
                    latency = float(metrics.avg_time_ms)
                    row.update(provenance)
                    row.update(
                        num_blocks=test_data["metadata"]["num_blocks"],
                        page_pool_policy=test_data["metadata"][
                            "page_pool_policy"
                        ],
                        latency_ms=latency,
                        status="ok",
                    )
                    seqlen_latency_ms[impl][seqlen_index] = latency
                except Exception as exc:
                    failures.append(("seqlen", impl, seq_len, exc))
                    row.update(
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                seqlen_rows.append(row)
                with seqlen_csv.open(
                    "w", newline="", encoding="utf-8"
                ) as file_obj:
                    writer = csv.DictWriter(
                        file_obj, fieldnames=seqlen_fields
                    )
                    writer.writeheader()
                    writer.writerows(seqlen_rows)
                if row["status"] == "ok":
                    print(
                        f"  {impl} seqlen={seq_len:5d}, "
                        f"latency={row['latency_ms']:.4f} ms"
                    )
                else:
                    print(
                        f"  {impl} seqlen={seq_len:5d} failed: "
                        f"{row['error']}"
                    )

        batch_latency_ms = {
            impl: {
                fixed_seq: [float("nan")] * len(batch_sizes)
                for fixed_seq in batch_curve_seq_lens
            }
            for impl in implementations
        }
        batch_rows = []
        batch_fields = [
            *common_fields[:9], "seq_len_fixed", *common_fields[9:],
            "point_index", "shard_index", "num_shards",
            "selection_mode", "shape_matrix_source", "coverage_mode",
            "coverage_total_formal_points", "coverage_selected_points",
            "coverage_total_requested_points",
            "selection_covers_full_formal_matrix", "coverage_complete",
            "provider_selection_mode",
            *provenance_fields,
        ]

        print(
            f"\n📈 曲线2: batch={batch_min}..{batch_max}, "
            f"seqlen固定 {batch_curve_seq_lens}"
        )
        batch_points_per_provider = (
            len(batch_curve_seq_lens) * len(batch_sizes)
        )
        batch_total_formal_points = (
            len(formal_implementations)
            * 2
            * 128
        )
        batch_selected_points = sum(
            1
            for implementation_index, _ in enumerate(implementations)
            for fixed_seq_index, _ in enumerate(batch_curve_seq_lens)
            for batch_index, _ in enumerate(batch_sizes)
            if (not quick or batch_index == 0)
            and (
                seqlen_point_count
                + implementation_index * batch_points_per_provider
                + fixed_seq_index * len(batch_sizes)
                + batch_index
            ) % num_shards == shard_index
        )
        batch_selection_provenance = build_curve_selection_provenance(
            quick=quick,
            num_shards=num_shards,
            total_formal_points=batch_total_formal_points,
            total_requested_points=(
                len(implementations)
                * len(batch_curve_seq_lens)
                * len(batch_sizes)
            ),
            selected_points=batch_selected_points,
            uses_formal_shape_matrix=(
                batch_sizes == list(range(1, 129))
                and batch_curve_seq_lens == [10000, 30000]
                and block_size == 128
                and num_blocks == 10000
            ),
            providers_complete=providers_complete,
        )
        for implementation_index, impl in enumerate(implementations):
            print(f"\n  provider: {impl}")
            for fixed_seq_index, fixed_seq in enumerate(
                batch_curve_seq_lens
            ):
                print(f"\n  固定 seqlen={fixed_seq}")
                for batch_index, batch_size in enumerate(batch_sizes):
                    point_index = (
                        seqlen_point_count
                        + implementation_index * batch_points_per_provider
                        + fixed_seq_index * len(batch_sizes)
                        + batch_index
                    )
                    if quick and batch_index != 0:
                        continue
                    if point_index % num_shards != shard_index:
                        continue
                    row = {
                        "provider": impl,
                        "device": device,
                        "device_name": device_name,
                        "torch_version": torch.__version__,
                        "provider_version": provider_version,
                        "precision": precision_type.name,
                        "num_heads": 8,
                        "num_kv_heads": 1,
                        "head_size": 128,
                        "seq_len_fixed": fixed_seq,
                        "batch_size": batch_size,
                        "block_size": block_size,
                        "num_blocks": num_blocks,
                        "page_pool_policy": "",
                        "seed": seed,
                        "latency_ms": "",
                        "status": "pending",
                        "error": "",
                        "point_index": point_index,
                        "shard_index": shard_index,
                        "num_shards": num_shards,
                        **batch_selection_provenance,
                        "provider_selection_mode": "fixed_formal_provider",
                        "warmup": num_warmup,
                        "iterations": num_iterations,
                        "repeats": repeats,
                    }
                    try:
                        test_data = self.operator_test.generate_latency_test_data(
                            batch_size=batch_size,
                            num_heads=8,
                            num_kv_heads=1,
                            head_size=128,
                            max_seq_len=fixed_seq,
                            num_blocks=num_blocks,
                            block_size=block_size,
                            seed=seed,
                            storage_device="cpu",
                        )
                        metrics, provenance = measure_one(test_data, impl)
                        latency = float(metrics.avg_time_ms)
                        row.update(provenance)
                        row.update(
                            num_blocks=test_data["metadata"]["num_blocks"],
                            page_pool_policy=test_data["metadata"][
                                "page_pool_policy"
                            ],
                            latency_ms=latency,
                            status="ok",
                        )
                        batch_latency_ms[impl][fixed_seq][
                            batch_index
                        ] = latency
                    except Exception as exc:
                        failures.append(("batch", impl, fixed_seq, batch_size, exc))
                        row.update(
                            status="error",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    batch_rows.append(row)
                    with batch_csv.open(
                        "w", newline="", encoding="utf-8"
                    ) as file_obj:
                        writer = csv.DictWriter(
                            file_obj, fieldnames=batch_fields
                        )
                        writer.writeheader()
                        writer.writerows(batch_rows)
                    if row["status"] == "ok":
                        print(
                            f"    {impl} batch={batch_size:3d}, "
                            f"latency={row['latency_ms']:.4f} ms"
                        )
                    else:
                        print(
                            f"    {impl} batch={batch_size:3d} failed: "
                            f"{row['error']}"
                        )

        print(f"\n✅ seqlen曲线数据已保存: {seqlen_csv}")

        print(f"✅ batch曲线数据已保存: {batch_csv}")

        for path, curve_rows, curve_fields, complete in (
            (
                seqlen_csv,
                seqlen_rows,
                seqlen_fields,
                finalize_curve_coverage(seqlen_rows),
            ),
            (
                batch_csv,
                batch_rows,
                batch_fields,
                finalize_curve_coverage(batch_rows),
            ),
        ):
            if complete:
                with path.open("w", newline="", encoding="utf-8") as file_obj:
                    writer = csv.DictWriter(
                        file_obj, fieldnames=curve_fields
                    )
                    writer.writeheader()
                    writer.writerows(curve_rows)

        if plot_results:
            plt.figure(figsize=(12, 7))
            markers = ['o', 's', '^', 'd', 'x', '*']
            for idx, impl in enumerate(implementations):
                marker = markers[idx % len(markers)]
                plt.plot(
                    seqlens,
                    seqlen_latency_ms[impl],
                    marker=marker,
                    linewidth=2,
                    markersize=4,
                    label=impl,
                )
            plt.xlabel('Sequence Length')
            plt.ylabel('Latency (ms)')
            plt.title(
                f'PagedAttention Latency vs Sequence Length '
                f'(batch={seqlen_batch_size}, {device})'
            )
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(seqlen_plot, dpi=160)
            plt.close()
            print(f"✅ seqlen曲线图已保存: {seqlen_plot}")

            plt.figure(figsize=(12, 7))
            line_styles = ['-', '--', '-.', ':']
            for impl_idx, impl in enumerate(implementations):
                marker = markers[impl_idx % len(markers)]
                for seq_idx, fixed_seq in enumerate(batch_curve_seq_lens):
                    plt.plot(
                        batch_sizes,
                        batch_latency_ms[impl][fixed_seq],
                        marker=marker,
                        linestyle=line_styles[seq_idx % len(line_styles)],
                        linewidth=2,
                        markersize=4,
                        label=f'{impl}, seqlen={fixed_seq}',
                    )
            plt.xlabel('Batch Size')
            plt.ylabel('Latency (ms)')
            plt.title(f'PagedAttention Latency vs Batch Size ({device})')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(batch_plot, dpi=160)
            plt.close()
            print(f"✅ batch曲线图已保存: {batch_plot}")
        else:
            print("ℹ️  --no-plot: 已跳过远端绘图，CSV 数据保持完整")

        if failures:
            raise RuntimeError(
                f"{len(failures)} formal PagedAttention point(s) failed; "
                f"checkpoints retained at {seqlen_csv} and {batch_csv}"
            )
        return {
            "device": device,
            "providers": implementations,
            "seqlen_csv": str(seqlen_csv),
            "batch_csv": str(batch_csv),
            "seqlen_plot": str(seqlen_plot),
            "batch_plot": str(batch_plot),
            "seqlen_rows": seqlen_rows,
            "batch_rows": batch_rows,
        }

def main():
    """主函数 - 支持独立运行PagedAttention算子测试"""
    import argparse
    import sys
    import os
    
    # 添加父目录到Python路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from operator_test_framework import OperatorTestFramework
    
    parser = argparse.ArgumentParser(description="PagedAttention算子测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["accuracy", "performance", "profile", "comprehensive", "fulltest", "latency"],
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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="latency 模式设备: auto, npu[:index], cuda[:index]",
    )
    parser.add_argument("--seqlen-start", type=int, default=1024)
    parser.add_argument("--seqlen-end", type=int, default=32768)
    parser.add_argument("--seqlen-step", type=int, default=1024)
    parser.add_argument("--seqlen-batch-size", type=int, default=128)
    parser.add_argument("--batch-min", type=int, default=1)
    parser.add_argument("--batch-max", type=int, default=128)
    parser.add_argument("--batch-step", type=int, default=1)
    parser.add_argument(
        "--batch-seq-lens",
        type=lambda value: [int(item) for item in value.split(",") if item],
        default=[10000, 30000],
        help="batch 曲线的固定 seqlen，逗号分隔 (默认: 10000,30000)",
    )
    parser.add_argument(
        "--warmup", "--num-warmup", dest="num_warmup", type=int, default=5
    )
    parser.add_argument(
        "--iterations", "--num-iterations", dest="num_iterations",
        type=int, default=20,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Framework V2 内部 repeat 数 (默认: 3)",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=10000,
        help="formal 物理 KV blocks 数，必须为 10000",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="formal KV page size，必须为 128",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="允许非融合 PyTorch SDPA 语义 fallback（仅调试，非默认性能曲线）",
    )
    parser.add_argument(
        "--providers",
        type=lambda value: [item for item in value.split(",") if item],
        default=["preferred"],
        help=(
            "PA provider，逗号分隔；preferred 表示 CUDA FlashInfer / "
            "NPU fused_infer_attention_score，all 表示设备全部实现"
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="仅生成 CSV；适用于没有 matplotlib 的远端运行环境",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    
    args = parser.parse_args()
    
    # 设置测试框架
    framework = OperatorTestFramework(result_dir=args.result_dir)
    
    # 创建并设置PagedAttention测试套件
    pa_suite = PagedAttentionTestSuite()
    pa_suite.setup(framework)
    
    # 根据模式运行测试
    if args.mode == "accuracy":
        results = pa_suite.run_accuracy_test()
    elif args.mode == "performance":
        results = pa_suite.run_performance_test()
    elif args.mode == "profile":
        results = pa_suite.run_profile_test()
    elif args.mode == "comprehensive":
        results = pa_suite.run_comprehensive_test()
    elif args.mode == "fulltest":
        # 生成随机测试案例并仅运行精度测试
        full_test_cases = pa_suite.create_full_test_cases(num_cases=args.num_cases)
        print(f"\n🎯 运行fulltest模式，生成 {len(full_test_cases)} 个随机测试案例")
        pa_suite.print_test_cases_summary(full_test_cases)
        results = pa_suite.run_accuracy_test(full_test_cases)
    elif args.mode == "latency":
        results = pa_suite.run_latency_plot_test(
            device=args.device,
            seqlen_start=args.seqlen_start,
            seqlen_end=args.seqlen_end,
            seqlen_step=args.seqlen_step,
            seqlen_batch_size=args.seqlen_batch_size,
            batch_min=args.batch_min,
            batch_max=args.batch_max,
            batch_step=args.batch_step,
            batch_curve_seq_lens=args.batch_seq_lens,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            repeats=args.repeats,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            seed=args.seed,
            allow_fallback=args.allow_fallback,
            providers=args.providers,
            plot_results=not args.no_plot,
            quick=args.quick,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    
    print(f"\n✓ PagedAttention算子测试完成，结果已保存到 {args.result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
