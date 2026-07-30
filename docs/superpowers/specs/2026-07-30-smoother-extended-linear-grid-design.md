# Smoother extended quantized Linear grid

## Goal

Replace the visually sparse three-point Linear extension with a modest
eight-point extension, then remeasure every compared provider under the same
OperatorTestFramework V2 protocol.

The exact square `M=N=K` grid is:

- `256..4096` in steps of `128`;
- `5120`;
- `6144`;
- `7168`;
- `8192`;
- `12288`;
- `16384`;
- `24576`;
- `32768`.

This is exactly 39 unique ascending points. No other Linear shapes are part of
the refreshed curves.

## Providers and protocol

The refreshed series are:

- H20 FP8: `cuda_vllm_cutlass_scaled_mm_fp8_bf16`;
- Ascend 950PR FP8:
  `npu_quant_matmul_fp8_e4m3_per_token_per_channel_bf16`;
- Ascend 950PR MXFP8:
  `npu_quant_matmul_mxfp8_e4m3_e8m0_group32_bf16`;
- Ascend 950PR MXFP4:
  `npu_quant_matmul_mxfp4_e2m1_e8m0_group32_bf16`.

Every series uses two stabilization repeats and five measured repeats. Each
invocation uses independent prepared input storage, and every output is
retained until its repeat ends. Quantization and layout preparation remain
outside Event timing.

All providers share the MXFP8 worst-case retained-byte estimate and the
existing 40 GiB hard planner. The required schedules are:

| Shape | Estimated bytes/invocation | Warmup | Iterations |
| ---: | ---: | ---: | ---: |
| 5120 | 106,496,000 | 10 | 50 |
| 6144 | 153,354,240 | 10 | 50 |
| 7168 | 208,732,160 | 10 | 50 |
| 8192 | 272,629,760 | 10 | 50 |
| 12288 | 613,416,960 | 10 | 50 |
| 16384 | 1,090,519,040 | 7 | 32 |
| 24576 | 2,453,667,840 | 3 | 14 |
| 32768 | 4,362,076,160 | 2 | 7 |

## Collection and plotting

The four complete 39-point curves must be rerun; supplementary points are not
spliced into the prior 34-point CSVs. Each source CSV must report complete
formal coverage and the exact provider, protocol, repeat, storage, and
per-shape warmup/iteration metadata.

The refreshed plots are written to a new dated artifact directory and do not
overwrite the prior 34-point artifacts. Linear plots retain a linear,
zero-based throughput y-axis. The left panel shows the original dense 31
points through 4096; the right panel shows all eight extension points. Lines
use the median of all five post-stabilization repeat means, and bands show the
complete repeat min/max range without filtering.

The audit builder must reject an incorrect row count, grid, provider,
precision, Framework V2 protocol, repeat policy, storage policy, or per-shape
warmup/iteration schedule before exporting or plotting.

## Success criteria

- the source implementation and dispatcher select the exact 39-point formal
  grid for FP8, MXFP8, and MXFP4;
- focused tests demonstrate RED before the grid change and GREEN afterward;
- H20 produces 39 successful FP8 rows;
- 950PR produces 39 successful rows for FP8, MXFP8, and MXFP4;
- all four sources use identical per-shape warmup and iteration counts;
- the two refreshed Linear plots visibly connect eight extension points;
- the audit export contains exactly 156 Linear rows and no failed or filtered
  point.
