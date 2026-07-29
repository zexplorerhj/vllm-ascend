#!/bin/bash

set -o pipefail

# Always resolve entry-point imports relative to this script, even when another
# source tree also contains a top-level ``tests`` package. Formal output paths
# remain owned by the caller and are resolved against its original directory.
CALLER_DIR=$PWD
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# 算子测试框架运行脚本

formal_usage() {
    cat >&2 <<'EOF'
Usage:
  run_tests.sh --formal --operator OP --device DEVICE --output-dir DIR
      [--precision fp8|mxfp8|mxfp4]
      [--warmup W] [--iterations I] [--repeats R]
      [--stabilization-repeats S]
      [--task-queue unset|0|1|2]
      [--shard-index N] [--num-shards M] [--quick] [--dry-run]

OP: add | linear | rmsnorm | flashattention | groupgemm |
    paged_attention | recurrent | all
EOF
}

formal_error() {
    echo "错误: $*" >&2
    formal_usage
    exit 2
}

is_formal_device() {
    case "$1" in
        auto|cuda|npu)
            return 0
            ;;
        cuda:*|npu:*)
            local device_index=${1#*:}
            case "$device_index" in
                ''|*[!0-9]*) return 1 ;;
                *) return 0 ;;
            esac
            ;;
        *)
            return 1
            ;;
    esac
}

run_formal_command() {
    if [ "$formal_dry_run" -eq 1 ]; then
        printf 'DRY-RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

build_protocol_args() {
    local protocol_kind=$1
    local warmup_flag=--warmup
    local iterations_flag=--iterations
    local repeats_flag=--repeats
    if [ "$protocol_kind" = "tflops" ]; then
        warmup_flag=--tflops-warmup
        iterations_flag=--tflops-iterations
        repeats_flag=--tflops-repeats
    fi
    formal_protocol_args=()
    if [ -n "$formal_warmup" ]; then
        formal_protocol_args+=("$warmup_flag" "$formal_warmup")
    fi
    if [ -n "$formal_iterations" ]; then
        formal_protocol_args+=("$iterations_flag" "$formal_iterations")
    fi
    if [ -n "$formal_repeats" ]; then
        formal_protocol_args+=("$repeats_flag" "$formal_repeats")
    fi
}

build_selection_args() {
    formal_selection_args=(
        --shard-index "$formal_shard_index"
        --num-shards "$formal_num_shards"
    )
    if [ "$formal_quick" -eq 1 ]; then
        formal_selection_args+=(--quick)
    fi
}

run_formal_entry() {
    local protocol_kind=$1
    local entry_path=$2
    shift 2
    local -a command
    build_protocol_args "$protocol_kind"
    build_selection_args
    command=(
        python3 "$entry_path" "$@"
        --device "$formal_device"
        --result-dir "$formal_output_dir"
        --no-plot
        "${formal_protocol_args[@]}"
        "${formal_selection_args[@]}"
    )
    run_formal_command "${command[@]}"
}

run_recurrent_entry() {
    local -a command
    build_protocol_args general
    build_selection_args
    command=(
        python3 tests/test_recurrent_gated_delta_rule.py
        --device "$formal_device"
        --hardware auto
        --output "$formal_output_dir/recurrent.csv"
        "${formal_protocol_args[@]}"
        "${formal_selection_args[@]}"
    )
    run_formal_command "${command[@]}"
}

dispatch_formal_operator() {
    local requested_operator=$1
    local command_status=0
    local precision

    case "$requested_operator" in
        add)
            run_formal_entry general tests/test_add.py \
                --mode bandwidth || command_status=$?
            ;;
        linear)
            # The Linear entry owns its precision-specific formal grid.
            if [ -n "$formal_precision" ]; then
                run_formal_entry tflops tests/test_linear.py \
                    --mode tflops --precision "$formal_precision" \
                    || command_status=$?
            else
                for precision in fp16 bf16; do
                    run_formal_entry tflops tests/test_linear.py \
                        --mode tflops --precision "$precision" \
                        || command_status=$?
                done
            fi
            ;;
        rmsnorm)
            run_formal_entry general tests/test_rmsnorm.py \
                --mode bandwidth || command_status=$?
            ;;
        flashattention)
            for precision in fp16 bf16; do
                run_formal_entry general tests/test_flash_attention.py \
                    --mode tflops --precision "$precision" \
                    || command_status=$?
            done
            ;;
        groupgemm)
            if [ -n "$formal_precision" ]; then
                run_formal_entry tflops tests/test_groupgemm.py \
                    --mode tflops --precision "$formal_precision" \
                    || command_status=$?
            else
                for precision in bf16 int8; do
                    run_formal_entry tflops tests/test_groupgemm.py \
                        --mode tflops --precision "$precision" \
                        || command_status=$?
                done
            fi
            ;;
        paged_attention)
            run_formal_entry general tests/test_paged_attention.py \
                --mode latency || command_status=$?
            ;;
        recurrent)
            run_recurrent_entry || command_status=$?
            ;;
        *)
            echo "错误: 未知 formal operator: $requested_operator" >&2
            return 2
            ;;
    esac
    return "$command_status"
}

if [ "$#" -gt 0 ]; then
    [ "$1" = "--formal" ] || formal_error "有参数调用时必须以 --formal 开始"
    shift

    formal_operator=""
    formal_device=""
    formal_output_dir=""
    formal_precision=""
    formal_warmup=""
    formal_iterations=""
    formal_repeats=5
    formal_stabilization_repeats=2
    formal_task_queue=unset
    formal_shard_index=0
    formal_num_shards=1
    formal_quick=0
    formal_dry_run=0

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --operator|--device|--output-dir|--precision|--warmup|--iterations|--repeats|--stabilization-repeats|--task-queue|--shard-index|--num-shards)
                [ "$#" -ge 2 ] || formal_error "$1 缺少值"
                case "$1" in
                    --operator) formal_operator=$2 ;;
                    --device) formal_device=$2 ;;
                    --output-dir) formal_output_dir=$2 ;;
                    --precision) formal_precision=$2 ;;
                    --warmup) formal_warmup=$2 ;;
                    --iterations) formal_iterations=$2 ;;
                    --repeats) formal_repeats=$2 ;;
                    --stabilization-repeats) formal_stabilization_repeats=$2 ;;
                    --task-queue) formal_task_queue=$2 ;;
                    --shard-index) formal_shard_index=$2 ;;
                    --num-shards) formal_num_shards=$2 ;;
                esac
                shift 2
                ;;
            --quick)
                formal_quick=1
                shift
                ;;
            --dry-run)
                formal_dry_run=1
                shift
                ;;
            --help|-h)
                formal_usage
                exit 0
                ;;
            *)
                formal_error "未知参数: $1"
                ;;
        esac
    done

    [ -n "$formal_operator" ] || formal_error "必须指定 --operator"
    case "$formal_operator" in
        add|linear|rmsnorm|flashattention|groupgemm|paged_attention|recurrent|all) ;;
        *) formal_error "不支持的 operator: $formal_operator" ;;
    esac
    case "$formal_precision" in
        ''|fp8|mxfp8|mxfp4) ;;
        *) formal_error "--precision 仅支持 fp8、mxfp8 或 mxfp4" ;;
    esac
    [ -n "$formal_device" ] || formal_error "必须指定 --device"
    is_formal_device "$formal_device" || \
        formal_error "不支持的 device: $formal_device"
    if [ "$formal_precision" = "fp8" ]; then
        case "$formal_operator" in
            linear|groupgemm|all) ;;
            *) formal_error "FP8 formal 仅支持 linear、groupgemm 或 all" ;;
        esac
        case "$formal_device" in
            cuda|cuda:[0-9]*|npu|npu:[0-9]*) ;;
            *)
                formal_error \
                    "FP8 formal 仅支持 CUDA SM90/H20 或 NPU Ascend 950PR"
                ;;
        esac
    elif [ "$formal_precision" = "mxfp8" ]; then
        case "$formal_operator" in
            linear|groupgemm|all) ;;
            *) formal_error "MXFP8 formal 仅支持 linear、groupgemm 或 all" ;;
        esac
        case "$formal_device" in
            npu|npu:[0-9]*) ;;
            *) formal_error "MXFP8 formal 仅支持 NPU Ascend 950PR" ;;
        esac
    elif [ "$formal_precision" = "mxfp4" ]; then
        case "$formal_operator" in
            linear|groupgemm|all) ;;
            *) formal_error "MXFP4 formal 仅支持 linear、groupgemm 或 all" ;;
        esac
        case "$formal_device" in
            npu|npu:[0-9]*) ;;
            *) formal_error "MXFP4 formal 仅支持 NPU Ascend 950PR" ;;
        esac
    fi
    [ -n "$formal_output_dir" ] || formal_error "必须指定 --output-dir"
    case "$formal_output_dir" in
        /*) ;;
        *) formal_output_dir="$CALLER_DIR/$formal_output_dir" ;;
    esac

    case "$formal_shard_index" in
        ''|*[!0-9]*) formal_error "--shard-index 必须是非负整数" ;;
    esac
    case "$formal_num_shards" in
        ''|*[!0-9]*) formal_error "--num-shards 必须是正整数" ;;
    esac
    [ "$formal_num_shards" -gt 0 ] || formal_error "--num-shards 必须大于 0"
    [ "$formal_shard_index" -lt "$formal_num_shards" ] || \
        formal_error "shard index 必须小于 num shards"

    if [ -n "$formal_warmup" ]; then
        case "$formal_warmup" in
            ''|*[!0-9]*) formal_error "--warmup 必须是非负整数" ;;
        esac
    fi
    for positive_count in "$formal_iterations" "$formal_repeats"; do
        if [ -n "$positive_count" ]; then
            case "$positive_count" in
                *[!0-9]*|0) formal_error "iterations/repeats 必须是正整数" ;;
            esac
        fi
    done
    case "$formal_stabilization_repeats" in
        ''|*[!0-9]*)
            formal_error "--stabilization-repeats 必须是非负整数"
            ;;
    esac
    case "$formal_stabilization_repeats" in
        0|[1-9]*) ;;
        *)
            formal_error \
                "--stabilization-repeats 必须使用规范十进制整数写法"
            ;;
    esac
    export OPERATOR_TEST_STABILIZATION_REPEATS=$formal_stabilization_repeats
    case "$formal_task_queue" in
        unset)
            unset TASK_QUEUE_ENABLE
            ;;
        0|1|2)
            export TASK_QUEUE_ENABLE=$formal_task_queue
            ;;
        *)
            formal_error "--task-queue 必须是 unset、0、1 或 2"
            ;;
    esac

    if ! command -v python3 &> /dev/null; then
        formal_error "未找到 python3"
    fi
    if [ "$formal_dry_run" -eq 0 ]; then
        mkdir -p "$formal_output_dir"
    fi

    formal_status=0
    if [ "$formal_operator" = "all" ]; then
        if [ -n "$formal_precision" ]; then
            formal_families=(linear groupgemm)
        else
            formal_families=(
                add linear rmsnorm flashattention groupgemm
                paged_attention recurrent
            )
        fi
        for formal_family in "${formal_families[@]}"; do
            dispatch_formal_operator "$formal_family" || formal_status=1
        done
    else
        dispatch_formal_operator "$formal_operator" || formal_status=$?
    fi
    exit "$formal_status"
fi

# Preserve the historical interactive default while keeping the formal path
# explicit and recorded in every NPU result row.
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}

echo "算子测试框架"
echo "============"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 创建结果目录
mkdir -p test_results

echo ""
echo "可用的测试选项:"
echo "1. 只测试Add算子"
echo "2. 只测试PagedAttention算子"
echo "3. 只测试GroupGemm算子"
echo "4. 只测试Linear算子"
echo "5. 只测试RMSNorm算子"
echo "6. 只测试FlashAttention算子"
echo "7. 列出所有已注册的算子"
echo "8. 测试RecurrentGatedDeltaRule算子"

read -r -p "请选择测试选项 (1-8): " choice

case $choice in
    1)
        echo "选择Add算子测试模式:"
        echo "a. 综合测试"
        echo "b. 只测试精度"
        echo "c. 只测试性能"
        echo "d. 全面随机测试(fulltest)"
        echo "e. 带宽测试"
        read -r -p "请选择模式 (a-e): " add_mode
        
        case $add_mode in
            a) python3 tests/test_add.py --mode comprehensive ;;
            b) python3 tests/test_add.py --mode accuracy ;;
            c) python3 tests/test_add.py --mode performance ;;
            d) python3 tests/test_add.py --mode fulltest ;;
            e) python3 tests/test_add.py --mode bandwidth ;;
            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    2)
        echo "选择PagedAttention算子测试模式:"
        echo "a. 综合测试"
        echo "b. 只测试精度"
        echo "c. 只测试性能"
        echo "d. Profile测试"
        echo "e. 全面随机测试(fulltest)"
        echo "f. 延迟画图测试(latency)"
        read -r -p "请选择模式 (a-f): " pa_mode
        
        case $pa_mode in
            a) python3 tests/test_paged_attention.py --mode comprehensive ;;
            b) python3 tests/test_paged_attention.py --mode accuracy ;;
            c) python3 tests/test_paged_attention.py --mode performance ;;
            d) python3 tests/test_paged_attention.py --mode profile ;;
            e) python3 tests/test_paged_attention.py --mode fulltest ;;
            f) python3 tests/test_paged_attention.py --mode latency ;;
            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    3)
        echo "选择GroupGemm算子测试模式:"
        echo "a. INT8 Profile测试"
        echo "b. INT8 性能测试"
        echo "c. BF16 Profile测试"
        echo "d. BF16 性能测试"
        echo "e. INT8 + NZ格式 Profile测试"
        echo "f. INT8 + NZ格式 性能测试"
        echo "g. TFLOPS测试 - INT8"
        echo "h. TFLOPS测试 - BF16"
        read -r -p "请选择模式 (a-h): " gg_mode

        case $gg_mode in
            a) python3 tests/test_groupgemm.py --precision int8 --mode profile ;;
            b) python3 tests/test_groupgemm.py --precision int8 --mode performance ;;
            c) python3 tests/test_groupgemm.py --precision bf16 --mode profile ;;
            d) python3 tests/test_groupgemm.py --precision bf16 --mode performance ;;
            e) python3 tests/test_groupgemm.py --precision int8 --use-nz-format --mode profile ;;
            f) python3 tests/test_groupgemm.py --precision int8 --use-nz-format --mode performance ;;
            g) python3 tests/test_groupgemm.py --precision int8 --mode tflops ;;
            h) python3 tests/test_groupgemm.py --precision bf16 --mode tflops ;;
            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    4)
        echo "选择Linear算子测试模式:"
        echo "a. Profile测试 - FP16"
        echo "b. Profile测试 - BF16"
        echo "c. 性能测试 V2 - FP16"
        echo "d. 性能测试 V2 - BF16"
        echo "e. TFLOPS测试 - FP16"
        echo "f. TFLOPS测试 - BF16"

        read -r -p "请选择模式 (a-f): " linear_mode
        
        case $linear_mode in
            a) python3 tests/test_linear.py --precision fp16 --mode profile ;;
            b) python3 tests/test_linear.py --precision bf16 --mode profile ;;
            c) python3 tests/test_linear.py --precision fp16 --mode performance ;;
            d) python3 tests/test_linear.py --precision bf16 --mode performance ;;
            e) python3 tests/test_linear.py --precision fp16 --mode tflops ;;
            f) python3 tests/test_linear.py --precision bf16 --mode tflops ;;

            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    5)
        echo "选择RMSNorm算子测试模式:"
        echo "a. 综合测试"
        echo "b. 只测试精度"
         echo "c. 只测试性能"
         echo "d. 全面随机测试(fulltest)"
         echo "e. 带宽测试"
         read -r -p "请选择模式 (a-e): " rms_mode
         
         case $rms_mode in
             a) python3 tests/test_rmsnorm.py --mode comprehensive ;;
             b) python3 tests/test_rmsnorm.py --mode accuracy ;;
             c) python3 tests/test_rmsnorm.py --mode performance ;;
             d) python3 tests/test_rmsnorm.py --mode fulltest ;;
             e) python3 tests/test_rmsnorm.py --mode bandwidth ;;
             *) echo "无效选择"; exit 1 ;;
         esac
         ;;
    6)
        echo "选择FlashAttention算子测试模式:"
        echo "a. 综合测试"
        echo "b. 只测试精度"
        echo "c. 只测试性能"
        echo "d. Profile测试"
        echo "e. TFLOPS测试 - FP16"
        echo "f. TFLOPS测试 - BF16"
        read -r -p "请选择模式 (a-f): " fa_mode
        
        case $fa_mode in
            a) python3 tests/test_flash_attention.py --mode comprehensive ;;
            b) python3 tests/test_flash_attention.py --mode accuracy ;;
            c) python3 tests/test_flash_attention.py --mode performance ;;
            d) python3 tests/test_flash_attention.py --mode profile ;;
            e) python3 tests/test_flash_attention.py --mode tflops --precision fp16 ;;
            f) python3 tests/test_flash_attention.py --mode tflops --precision bf16 ;;
            *) echo "无效选择"; exit 1 ;;
        esac
        ;;
    8)
        recurrent_timestamp=$(date +%Y%m%d_%H%M%S)
        python3 tests/test_recurrent_gated_delta_rule.py \
            --device auto \
            --hardware auto \
            --modes decode,mtp3 \
            --batches 1,4,8,16,32,64,128 \
            --warmup 5 --iterations 20 --repeats 3 \
            --output "test_results/recurrent_gated_delta_rule_${recurrent_timestamp}.csv"
        ;;
    7)
        echo "列出所有已注册的算子..."
        python3 test_main.py --list
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

test_status=$?
if [ "$test_status" -ne 0 ]; then
    echo ""
    echo "测试失败（退出码: $test_status），请检查上面的错误和 test_results/ 诊断文件。"
    exit "$test_status"
fi

echo ""
echo "测试完成！结果保存在 test_results/ 目录中"
