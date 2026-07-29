# Task 2 — FP8 Linear Provider

## Files changed

- `tests/ops/linear/linear_fp8_operator.py` — H20 CUDA FP8 W8A8 Linear
  provider using caller-owned BF16 output and the low-level vLLM CUTLASS op.
- `tests/ops/tests/test_linear.py` — FP8 suite/CLI routing and FP8 fresh-byte
  accounting (`A + B + BF16 C + FP32 row/channel scales`).
- `tests/ut/ops/test_operator_curve_providers.py` — provider payload layout,
  exact out-first invocation, output contract, and alignment coverage.
- `tests/ut/ops/test_operator_curve_entries.py` — FP8 formal-curve and CLI
  dispatch coverage.

## TDD evidence

### RED

```bash
pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_curve_entries.py -k 'linear and fp8'
```

Initial result: `5 failed, 98 deselected`; failures identified the missing
`LinearFp8OperatorTest` and rejected `--precision fp8` CLI selection.

An additional alignment test was added after the provider existed and produced
the expected RED: `1 failed, 5 passed, 98 deselected` because unaligned K/N
dimensions were accepted.

### GREEN

```bash
pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_curve_entries.py -k 'linear and fp8'
```

Result: `6 passed, 98 deselected in 1.01s`.

Additional local verification:

```bash
pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_curve_entries.py
python3 -m py_compile tests/ops/linear/linear_fp8_operator.py \
  tests/ops/tests/test_linear.py \
  tests/ut/ops/test_operator_curve_providers.py \
  tests/ut/ops/test_operator_curve_entries.py
git diff --check
```

Result: `104 passed in 1.78s`; compilation and diff checks exited 0.

## Commit

`test(ops): add H20 FP8 Linear provider` (this report is included in that
Task 2-only commit).

## Concerns

- The required command without `--confcutdir` remains blocked before test
  collection by the existing local `ModuleNotFoundError: torch_npu` from
  `tests/ut/conftest.py`; the equivalent focused tests above passed with the
  permitted local isolation.
- H20/vLLM CUTLASS runtime validation is intentionally outside this local Task
  2 scope. The provider rejects K/N values not divisible by 16 rather than
  silently padding them.
- Existing unrelated changes remain untouched: `tests/ops/flashattention/__init__.py`
  and the untracked plan/spec documents.
