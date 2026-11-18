# Device field evaluation validation

Measured on 2026-11-18 against main `7a6a619`, using an NVIDIA L40, driver
570.158.01, Warp 1.19.0+cu12, Python 3.10.12, and NumPy 2.2.6. The container's
effective CPU quota was **four cores**. Raw reports include source/input hashes,
solver settings, every timing sample, and cgroup throttling counters.

The original batched reconstruction-field kernel is retained unchanged (its
Python AST matches the baseline). The device function has a separate forward
implementation. All three routes produce bitwise-identical values, gradients,
and validity flags on these inputs. Reconstruction coefficients and isovalues
also match bitwise. There is no material regression in the existing batch route;
the new device function's throughput depends on the dataset and precision.

## Full-cloud queries

Times are warmed GPU kernel milliseconds for one query per input point.
“Fused” calls `evaluate_field` from an application kernel and writes the same
outputs as the batch kernels. Applications can instead consume the result
directly, avoiding intermediate output arrays and a separate launch.

| Dataset | Depth | Precision | Queries | Original batch | Current batch | Fused device function |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Bunny | 8 | float32 | 362,271 | 0.2029 | 0.2029 | 0.2055 |
| Horse | 8 | float32 | 100,000 | 0.1997 | 0.1997 | 0.1912 |
| Eagle | 8 | float32 | 796,825 | 0.2768 | 0.2767 | 0.3394 |
| Eagle | 9 | float32 | 796,825 | 0.5104 | 0.5069 | 0.4709 |
| Bunny | 8 | float64 | 362,271 | 0.2595 | 0.2596 | 0.3060 |

The batch route remains preferable for standalone large queries on eagle depth 8
and double-precision bunny. Raw reports also cover one and 4,096 queries.
The double-precision run uses the same input points converted to float64.

## Reconstruction and batched backward evaluation

Reconstruction includes allocation, host dispatch, solve, and input-point
evaluation, but excludes uploads, compilation, meshing, validation, and garbage
collection. The benchmark exchanges only the original/current reference field
kernel; the remaining forward reconstruction implementation is unchanged.
These small wall-time fluctuations do not establish a reconstruction speedup.

| Dataset / depth / precision | Original reconstruction (ms) | Current reconstruction (ms) | Original backward (ms) | Current backward (ms) |
| --- | ---: | ---: | ---: | ---: |
| Bunny / 8 / float32 | 110.55 | 106.77 | 0.2525 | 0.2426 |
| Horse / 8 / float32 | 70.98 | 70.69 | 0.2297 | 0.2255 |
| Eagle / 8 / float32 | 87.73 | 89.03 | 0.3982 | 0.3822 |
| Eagle / 9 / float32 | 213.37 | 204.69 | 0.6238 | 0.5999 |
| Bunny / 8 / float64 | 145.55 | 144.29 | 0.5550 | 0.5614 |

The backward kernel now shares the device function's analytic per-query VJP.
Large-cloud timing and comparison use value seeds of one and spatial-gradient
seeds of zero. Nonnegative coefficient contributions avoid cancellation-sensitive
atomic summation differences, which also occur between repeated baseline runs.
The correctness suite separately checks mixed value/spatial-gradient losses
against finite differences; comparison tolerances were not relaxed.

## Method and correctness

- Five rounds, ten samples per variant. Query timings use CUDA events around ten
  graph replays of ten launches, divided by 100, in
  original/current/fused/fused/current/original order. Backward and reconstruction
  use original/current/current/original order; reconstruction uses synchronized
  wall time. Warm-up and compilation are outside all measurements.
- The scheduler waits for ten consecutive idle observations and samples CPU/GPU
  processes every second. Every accepted case has no foreign GPU process or
  foreign CPU process above 0.25 cores. The first horse attempt was rejected for
  CPU contention; its report and observations remain in `rejected/`.
- Full suite: **715 passed, 6 skipped, 1 expected failure**. The 48 focused field
  checks cover float32/float64, CPU/CUDA, adaptive cells, knots, transformed and
  invalid queries, empty batches, coefficient/query derivatives, differentiable
  reconstruction, helper composition, and CUDA graph capture.
- Compute Sanitizer: **five supported GPU/autodiff cases passed, zero errors**.
  The direct-field query derivative regression test fails on the original
  baseline and passes with the analytic VJP.
- Pinned Warp requires `@wp.func(inline=True)` on additional device helpers that
  accept the field handle. See the [API contract](../../field-queries.md) and
  [open composition investigation](https://github.com/alecjacobson/warp-psr/issues/101).

Cold compilation time and unrelated solver/meshing performance are outside this
comparison. The benchmark loads baseline evaluators from Git; it does not claim
to benchmark a separate checkout of every unchanged reconstruction component.

## Reproduce

Use the project's existing Warp/test environment and validated bunny, horse, and
eagle fixtures from the [dataset catalog](../../datasets.md). This run reuses
`.build/gpu-baselines/inputs/{dataset}-full-d8-spsr`; hashes and settings are in
each report. No external FFT or NeuralGalerkin dependencies are needed.

```sh
uv run --no-sync python docs/results/device-field/run.py
uv run --no-sync python -m pytest -q --disable-warnings --maxfail=1
uv run --no-sync compute-sanitizer --tool memcheck --error-exitcode 9 \
  .venv/bin/python docs/results/device-field/sanitize.py
```

If Compute Sanitizer is not on `PATH`, use its installed absolute path. To run
one case with a different fixture location, use `python -m
tools.benchmark_device_field --fixture PATH --depth 9 --precision float
--output REPORT.json` in the project environment. Use the isolation scheduler
when accepting measurements on a shared machine. `validation.json` records the
test outcomes and exact production source hashes.
