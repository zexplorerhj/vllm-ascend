# 自定义算子测试框架 (OpTests)

一个专业的深度学习算子精度测试和性能基准测试框架，支持Ascend NPU和精度类型，提供统一的测试接口和详细的性能分析。

## 🎯 项目目标

- **精度控制**: 确保自定义算子在不同精度下的计算准确性
- **性能基准**: 对重要算子的不同实现进行全面的性能对比测试
- **多设备支持**: 统一的测试框架支持CPU、NPU等多种计算设备
- **标准化测试**: 提供标准化的测试流程和结果输出格式

## ✨ 主要特性

### 🔍 精度测试
- 支持多种精度类型：FP16、BF16、FP32、INT8
- 自动计算精度指标：绝对误差、相对误差、余弦相似度
- 与CPU参考实现对比验证
- 详细的精度分析报告

### ⚡ 性能测试
- 多种性能指标：延迟、吞吐量、内存使用
- 支持不同算子实现的性能对比
- 集成Profiler进行深度性能分析
- 自动生成性能基准报告

### 🔧 框架特性
- 统一的算子测试接口
- 可扩展的算子注册机制
- 灵活的测试配置
- 结构化的结果输出（JSON、CSV）
- 自动化测试脚本

## 📁 项目结构

```
OpTests/
├── README.md                    # 项目说明文档
├── operator_test_framework.py   # 核心测试框架
├── run_tests.sh                 # 测试运行脚本
│
├── tests/                       # 测试套件
│   ├── __init__.py
│   ├── base_test_suite.py      # 测试套件基类
│   ├── test_add.py             # Add算子测试
│   ├── test_groupgemm.py       # GroupGemm算子测试
│   ├── test_linear.py          # Linear算子测试
│   └── test_paged_attention.py # PagedAttention算子测试
│
├── add/                         # Add算子实现
│   └── add_operator.py
│
├── linear/                      # Linear算子实现
│   └── linear_operator.py
│
├── groupgemm/                   # GroupGemm算子实现
│   ├── base_groupgemm.py
│   ├── groupgemm_bf16.py
│   └── groupgemm_int8.py
│
└── paged_attention/             # PagedAttention算子实现
    ├── __init__.py
    ├── base.py
    ├── fused_bnsd_impl.py
    ├── fused_bsh_impl.py
    └── original_impl.py
```

## 🚀 快速开始

### 运行测试

#### 1. 使用交互式脚本

```bash
./run_tests.sh
```

脚本会提供交互式菜单，选择要测试的算子和测试模式。

#### 2. 直接运行特定算子测试

```bash
# Add算子综合测试
python3 tests/test_add.py --mode comprehensive

# 只测试精度
python3 tests/test_add.py --mode accuracy

# 只测试性能
python3 tests/test_add.py --mode performance

# 快速测试
python3 tests/test_add.py --mode quick
```

#### 3. 其他算子测试

```bash
# PagedAttention算子测试
python3 tests/test_paged_attention.py --mode comprehensive

# GroupGemm算子测试
python3 tests/test_groupgemm.py --mode performance

# Linear算子测试
python3 tests/test_linear.py --mode accuracy
```

## 📊 测试结果

测试结果会保存在 `test_results/` 目录下，包含：

- **JSON格式**: 详细的测试数据和指标
- **CSV格式**: 便于分析的表格数据
- **Profiler输出**: 性能分析文件（如果启用）

## Formal curve protocol

`run_tests.sh --formal` dispatches the canonical curve entries through
`OperatorTestFramework.run_core_operator_performance_test_v2`.  Each repeat
prepares `warmup + iterations` independent input/workspace-storage sets
before timing; explicitly designated outputs are audited separately.  The CSV
reports the median of the measured repeat Event means and retains every
repeat sample.

Protocol `operator-test-framework-v2-fresh-v6` first runs two complete
fresh-storage stabilization repeats, excluded from aggregation, then runs
five measured repeats by default.  Every stabilization and measured repeat
independently prepares `W+I` payloads and never reuses an address within that
repeat; one repeat's lifetime ends before the next begins.  The already
prepared payloads are dispatched with one direct Python loop; there is no
implicit clock-ramp kernel.  Device Event elapsed time still includes any
stream-idle gap caused by Python/ATen launch dispatch, so the reported number
is provider service-path latency, not a claim of profiler-isolated kernel
duration.
CSV provenance records the excluded stabilization samples, measured samples,
max/min spread, and measured-repeat interquartile spread.

Add, Linear, RMSNorm, and recurrent GDN use deterministic adaptive measured
iterations when the iteration flag is omitted.  All devices use the same
provider-independent byte formula for a shape:

- Add BF16: `6 * elements`
- square Linear FP16/BF16: `6 * size^2`
- RMSNorm BF16:
  `4*elements + 2*hidden + 4*(elements/hidden)`
- recurrent GDN:
  `534628*T + 524288 + 4*(B+1) + (4*B for MTP3)`

The selected count is
`max(base_I, min(cap, floor(4 GiB / bytes_per_invocation) - W))`, clamped at
zero before the outer maximum.  The cap is 8192 for Add and 2048 for Linear,
RMSNorm, and recurrent GDN.  Four GiB is a *soft target*: historical base
iterations are never reduced, so large shapes can exceed it.  PagedAttention
uses fixed `W5/I64`; GroupGemm uses fixed `W10/I30`.  CSV rows record the
requested/base/effective counts, canonical byte estimate, target, estimated
repeat footprint, overflow flag, and actual Event-window samples.  This is
stability-depth adaptation; it does not reuse addresses.

When an operator has an explicit preallocated `out=` buffer and declares a
phase-invariant output contract, all `W+I` output-buffer addresses are checked
for independence, the complete input/workspace and output address domains
must be disjoint, and untimed warmup returns probe the alias contract.  Timed
Python return objects are then discarded.  Providers without that contract
retain every return in Event-external preallocated Python list slots until the
repeat ends; their slot assignment remains part of timed dispatch.
Provider-internal workspace allocation remains `not_audited`.
Only PagedAttention currently has a strict preallocated-output contract on
both CUDA and NPU.  RMSNorm, FlashAttention, and GroupGemm have asymmetric
native output-allocation APIs, so comparisons must preserve and disclose the
per-row `output_allocation_mode`; they are service-path comparisons, not a
claim of identical allocator-free kernel contracts.

Formal NPU runs unset `TASK_QUEUE_ENABLE` unless
`--task-queue 0|1|2` is supplied.  The selected value is written to each NPU
result row.  Device-specific tuned curves may choose a different explicit
mode, but the plot must disclose it; A3/A5 queue modes must not be assumed
equivalent.

`--quick` only selects the first shape of each canonical sub-curve.  It does
not change warmup, iteration, or repeat counts, and its rows can never claim
full formal coverage.

### H20 FP8 formal curves

FP8 curves are an explicit CUDA-only opt-in for H20/SM90; they are never part
of the default NPU or `all` formal matrix.  Select them with:

```bash
./run_tests.sh --formal --operator all --precision fp8 --device cuda:0 \
    --output-dir h20-fp8-results
```

This dispatches exactly the FP8 Linear and FP8 GroupGemm entries.  Both use
W8A8 E4M3: activations have one FP32 dequantization scale per token
(`[M, 1]`), and weights have one FP32 scale per output channel (`[1, N]` for
Linear and `[E, N]` for GroupGemm).  Outputs are BF16 and bias is disabled.

Each fresh payload performs quantization and builds CUTLASS metadata before
the Event-timed region.  The timed providers only invoke the low-level
caller-owned-output CUTLASS operation, and the framework audits strict output
aliasing and storage independence.  The H20 formal GroupGemm provider is the
`cuda_vllm_cutlass_scaled_mm_fp8_bf16_expert_loop`; the grouped CUTLASS
provider is diagnostic-only and is not mixed into formal curves.
