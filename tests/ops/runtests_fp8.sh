#!/usr/bin/env bash

set -o pipefail

# Run the canonical quantized curve matrix through run_tests.sh.  This script
# only orchestrates precision/family passes; shape grids, fresh-storage planning,
# Event timing, correctness, providers, and CSV provenance remain owned by the
# existing operator entries and OperatorTestFramework.

CALLER_DIR=$PWD
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_TESTS="$SCRIPT_DIR/run_tests.sh"
PLOT_NORMQUANT="$SCRIPT_DIR/plot_normquant_captured.py"
PLOT_SUITE="$SCRIPT_DIR/plot_fp8_suite.py"

device=""
output_dir=""
precisions_csv=""
warmup=""
iterations=""
repeats=5
stabilization_repeats=2
task_queue="unset"
shard_index=0
num_shards=1
quick=0
dry_run=0
plot_results=1

usage() {
    cat <<'EOF'
Usage:
  runtests_fp8.sh --device DEVICE [--output-dir DIR]
      [--precisions fp8,mxfp8,mxfp4]
      [--warmup W] [--iterations I] [--repeats R]
      [--stabilization-repeats S]
      [--task-queue unset|0|1|2]
      [--shard-index N] [--num-shards M]
      [--quick] [--dry-run] [--no-plot]

Runs the formal curve entries for all three operator families:
  Linear, GroupGemm, and NormQuant.

Default precision matrix:
  npu[:N]   fp8,mxfp8,mxfp4  (Ascend 950PR)
  cuda[:N]  fp8              (H20/SM90)

Examples:
  bash runtests_fp8.sh --device npu:0 \
      --output-dir test_results/950pr-fp8-suite

  bash runtests_fp8.sh --device cuda:0 \
      --output-dir test_results/h20-fp8-suite

  bash runtests_fp8.sh --device npu:0 --precisions fp8,mxfp8 \
      --quick --dry-run

When --warmup and --iterations are both omitted, the safe canonical protocols
are used per family: Linear uses memory-bounded adaptive fresh-storage counts,
while GroupGemm and NormQuant use W10/I30.  A shared explicit override is
forwarded to every family and may be unsuitable for the largest square Linear
shapes.  R defaults to 5 measured repeats after 2 excluded stabilization
repeats.  The underlying formal dispatcher emits CSV/checkpoint data and
intentionally skips local per-entry plotting.  After a successful, unsharded
full run, the default Ascend three-precision matrix writes the canonical
five-panel overview to:
  <output-dir>/fp8_suite_overview.png
The overview contains Linear, GroupGemm, and captured FP8/MXFP8/MXFP4
NormQuant token sweeps.  CUDA and explicit precision subsets fall back to:
  <output-dir>/normquant_captured_comparison.png
Quick, dry-run, sharded, and failed suites do not draw a potentially partial
or stale figure.  Use --no-plot to generate CSV/checkpoint artifacts only.
EOF
}

die() {
    echo "错误: $*" >&2
    usage >&2
    exit 2
}

is_nonnegative_integer() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

is_positive_integer() {
    is_nonnegative_integer "$1" && [ "$1" -gt 0 ]
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --device|--output-dir|--precisions|--warmup|--iterations|--repeats|--stabilization-repeats|--task-queue|--shard-index|--num-shards)
            [ "$#" -ge 2 ] || die "$1 缺少值"
            case "$1" in
                --device) device=$2 ;;
                --output-dir) output_dir=$2 ;;
                --precisions) precisions_csv=$2 ;;
                --warmup) warmup=$2 ;;
                --iterations) iterations=$2 ;;
                --repeats) repeats=$2 ;;
                --stabilization-repeats) stabilization_repeats=$2 ;;
                --task-queue) task_queue=$2 ;;
                --shard-index) shard_index=$2 ;;
                --num-shards) num_shards=$2 ;;
            esac
            shift 2
            ;;
        --quick)
            quick=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --no-plot)
            plot_results=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "未知参数: $1"
            ;;
    esac
done

[ -f "$RUN_TESTS" ] || die "未找到 formal dispatcher: $RUN_TESTS"
[ -n "$device" ] || die "必须指定 --device"

case "$device" in
    npu|npu:[0-9]*)
        default_precisions=fp8,mxfp8,mxfp4
        device_family=npu
        ;;
    cuda|cuda:[0-9]*)
        default_precisions=fp8
        device_family=cuda
        ;;
    *)
        die "--device 仅支持 npu[:N] 或 cuda[:N]，收到: $device"
        ;;
esac

if [ -z "$precisions_csv" ]; then
    precisions_csv=$default_precisions
fi

IFS=',' read -r -a precisions <<< "$precisions_csv"
[ "${#precisions[@]}" -gt 0 ] || die "--precisions 不能为空"

normalized_precisions=()
for precision in "${precisions[@]}"; do
    case "$precision" in
        fp8|mxfp8|mxfp4) ;;
        *) die "不支持的 precision: $precision" ;;
    esac
    if [ "$device_family" = cuda ] && [ "$precision" != fp8 ]; then
        die "CUDA/H20 不支持 $precision；仅可运行 fp8"
    fi
    duplicate=0
    for existing in "${normalized_precisions[@]}"; do
        if [ "$existing" = "$precision" ]; then
            duplicate=1
            break
        fi
    done
    if [ "$duplicate" -eq 0 ]; then
        normalized_precisions+=("$precision")
    fi
done

plot_kind=normquant-captured
plot_script=$PLOT_NORMQUANT
plot_filename=normquant_captured_comparison.png
if [ "$device_family" = npu ] && \
        [ "${#normalized_precisions[@]}" -eq 3 ]; then
    has_fp8=0
    has_mxfp8=0
    has_mxfp4=0
    for precision in "${normalized_precisions[@]}"; do
        case "$precision" in
            fp8) has_fp8=1 ;;
            mxfp8) has_mxfp8=1 ;;
            mxfp4) has_mxfp4=1 ;;
        esac
    done
    if [ "$has_fp8" -eq 1 ] && [ "$has_mxfp8" -eq 1 ] && \
            [ "$has_mxfp4" -eq 1 ]; then
        plot_kind=suite-overview
        plot_script=$PLOT_SUITE
        plot_filename=fp8_suite_overview.png
    fi
fi

if [ "$dry_run" -eq 0 ] && [ "$plot_results" -eq 1 ] && \
        [ "$quick" -eq 0 ] && [ "$num_shards" -eq 1 ]; then
    [ -f "$plot_script" ] || die "未找到绘图脚本: $plot_script"
fi

if [ -n "$warmup" ]; then
    is_nonnegative_integer "$warmup" || die "--warmup 必须是非负整数"
    [ "$warmup" -ge 2 ] || die "统一 warmup 必须 >= 2，以满足 NormQuant"
fi
if [ -n "$iterations" ]; then
    is_positive_integer "$iterations" || die "--iterations 必须是正整数"
fi
is_positive_integer "$repeats" || die "--repeats 必须是正整数"
is_nonnegative_integer "$stabilization_repeats" || \
    die "--stabilization-repeats 必须是非负整数"
is_nonnegative_integer "$shard_index" || \
    die "--shard-index 必须是非负整数"
is_positive_integer "$num_shards" || die "--num-shards 必须是正整数"
[ "$shard_index" -lt "$num_shards" ] || \
    die "--shard-index 必须小于 --num-shards"
case "$task_queue" in
    unset|0|1|2) ;;
    *) die "--task-queue 必须是 unset、0、1 或 2" ;;
esac

if [ -z "$output_dir" ]; then
    output_dir="test_results/fp8-curves-$(date +%Y%m%d-%H%M%S)"
fi
case "$output_dir" in
    /*) ;;
    *) output_dir="$CALLER_DIR/$output_dir" ;;
esac

echo "Quantized operator formal curves"
echo "================================"
echo "Device:     $device"
echo "Precisions: ${normalized_precisions[*]}"
echo "Families:   Linear, GroupGemm, NormQuant"
echo "Output:     $output_dir"
echo "Protocol:   R${repeats}/S${stabilization_repeats}, task_queue=$task_queue"
if [ "$dry_run" -eq 1 ]; then
    echo "Plot:       skipped for dry-run"
elif [ "$quick" -eq 1 ]; then
    echo "Plot:       skipped for quick run"
elif [ "$num_shards" -ne 1 ]; then
    echo "Plot:       skipped for sharded run"
elif [ "$plot_results" -eq 1 ]; then
    echo "Plot:       $output_dir/$plot_filename"
else
    echo "Plot:       disabled"
fi
if [ -n "$warmup" ]; then
    echo "Warmup:     $warmup (explicit shared override)"
else
    echo "Warmup:     Linear adaptive; GroupGemm/NormQuant 10"
fi
if [ -n "$iterations" ]; then
    echo "Iterations: $iterations (explicit shared override)"
else
    echo "Iterations: Linear adaptive; GroupGemm/NormQuant 30"
fi
echo ""

suite_status=0
plot_created=0
succeeded=()
failed=()
families=(linear groupgemm norm_quant)

for precision in "${normalized_precisions[@]}"; do
    precision_output="$output_dir/$precision"
    for family in "${families[@]}"; do
        family_warmup=$warmup
        family_iterations=$iterations
        if [ "$family" != linear ]; then
            family_warmup=${family_warmup:-10}
            family_iterations=${family_iterations:-30}
        fi
        command=(
            bash "$RUN_TESTS"
            --formal
            --operator "$family"
            --precision "$precision"
            --device "$device"
            --output-dir "$precision_output"
            --repeats "$repeats"
            --stabilization-repeats "$stabilization_repeats"
            --task-queue "$task_queue"
            --shard-index "$shard_index"
            --num-shards "$num_shards"
        )
        if [ -n "$family_warmup" ]; then
            command+=(--warmup "$family_warmup")
        fi
        if [ -n "$family_iterations" ]; then
            command+=(--iterations "$family_iterations")
        fi
        if [ "$quick" -eq 1 ]; then
            command+=(--quick)
        fi
        if [ "$dry_run" -eq 1 ]; then
            command+=(--dry-run)
        fi

        echo "[$precision/$family] running"
        if "${command[@]}"; then
            succeeded+=("$precision/$family")
        else
            status=$?
            suite_status=1
            failed+=("$precision/$family(exit=$status)")
        fi
        echo ""
    done
done

if [ "$dry_run" -eq 0 ] && [ "$plot_results" -eq 1 ] && \
        [ "$quick" -eq 0 ] && [ "$num_shards" -eq 1 ] && \
        [ "$suite_status" -eq 0 ]; then
    plot_output="$output_dir/$plot_filename"
    plot_command=(
        python3 "$plot_script"
        --input-root "$output_dir"
        --output "$plot_output"
    )
    if [ "$plot_kind" = normquant-captured ]; then
        plot_command+=(--precisions "${normalized_precisions[@]}")
    fi

    echo "[plot/$plot_kind] running"
    if "${plot_command[@]}"; then
        plot_created=1
        succeeded+=("plot/$plot_kind")
    else
        status=$?
        suite_status=1
        failed+=("plot/$plot_kind(exit=$status)")
    fi
    echo ""
elif [ "$dry_run" -eq 0 ] && [ "$plot_results" -eq 1 ] && \
        [ "$quick" -eq 0 ] && [ "$num_shards" -eq 1 ] && \
        [ "$suite_status" -ne 0 ]; then
    echo "[plot/$plot_kind] skipped because benchmark suite failed"
    echo ""
fi

echo "Suite summary"
echo "============="
if [ "${#succeeded[@]}" -gt 0 ]; then
    echo "Succeeded: ${succeeded[*]}"
fi
if [ "${#failed[@]}" -gt 0 ]; then
    echo "Failed:    ${failed[*]}"
fi
echo "Results:   $output_dir/<precision>/"
if [ "$plot_created" -eq 1 ]; then
    echo "Plot:      $output_dir/$plot_filename"
fi

exit "$suite_status"
