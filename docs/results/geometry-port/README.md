# Geometry port validation

Written by Codex (AI agent), 2026-11-26.

The draft's mesh API was subsequently simplified; see the [API revision and validation](api-revision.md) for the current tuple return, string methods, removal of the Poisson-specific Lipschitz path, and expanded PR examples. The measurements below retain their original source revisions.

The implementation and public API plan is tracked in [issue 104](https://github.com/alecjacobson/warp-psr/issues/104).
The feature branch is [`alecjacobson/poisson-surface-reconstruction`](https://github.com/alecjacobson/warp/tree/alecjacobson/poisson-surface-reconstruction).
The completed draft is [NVIDIA/warp#2452](https://github.com/NVIDIA/warp/pull/2452).
The upstream feature is tracked in [NVIDIA/warp#2450](https://github.com/NVIDIA/warp/issues/2450).
The branch carries the separate device-function compiler prerequisite from [PR 2363](https://github.com/NVIDIA/warp/pull/2363), which remains open. The geometry PR stays draft for GitLab migration.

## Revisions and scope

- Baseline: this repository at `6c5ebe54ee49743d3cb1836aead9905afc3490fe`.
- Final published feature: `57473e17b32b65d3264b9a593967dd78d403614d` (only a private module docstring cleanup after the measured revision; executable code is identical).
- Measured feature: `e426fed8fe8222c2dec7badf7278adadb0c93b6c` in the Warp fork.
- Both implementations use the same Warp Python/native build, based on NVIDIA main `98d6b5ed56800213e94cc191f0c0588239d6049e`, plus compiler prerequisite `51ac591`.
- Linux x86_64, NVIDIA L40 48 GB, driver 570.158.01, CUDA 12.6.85, Python 3.12.16, Warp 1.21.0.dev0. The container and limiting ancestor each expose a **4-core CPU quota**, despite 128 visible CPUs.

The runtime moved into `warp/_src/geometry/poisson/`. Only `warp.geometry` gains public exports: `poisson_reconstruct`, `PoissonReconstruction`, `poisson_extract_mesh`, `PoissonMesh`, `PoissonMeshMethod`, `PoissonField`, `PoissonField64`, and `poisson_evaluate`. Solver/FEM internals remain private. The public profile is degree-1, Neumann, approximate screening; weight zero selects unscreened reconstruction. Reconstruction supports float32/float64 CUDA arrays, and existing Warp dense/sparse marching cubes provide float32 extraction.

The independent fast batch kernel remains in use. The host wrapper validates arrays and tape ownership, caches invariant precision/device metadata, and returns the existing batch result directly in the default path. Device queries and first-order reconstruction/mesh derivatives are optional. Discrete topology, grouping, support, stopping and connectivity decisions remain fixed during backward. Reconstruction and extraction cannot be captured; warmed device queries can.

Documentation follows Warp's generator: the API reference is generated, while the explanatory guide lives at `docs/user_guide/poisson_reconstruction.rst`. No licensing files or runtime dependencies changed. The upstream branch contains no source-project references.

## Performance

| Case | Points | Baseline (ms) | Geometry API (ms) | Change |
| --- | ---: | ---: | ---: | ---: |
| Bunny d8 float | 362,271 | 100.422 | 100.566 | +0.14% |
| Horse d8 float | 100,000 | 58.905 | 59.767 | +1.46% |
| Eagle d8 float | 796,825 | 78.094 | 78.790 | +0.89% |
| Eagle d9 float | 796,825 | 193.360 | 192.407 | -0.49% |
| Bunny d8 double | 362,271 | 129.757 | 131.693 | +1.49% |


The primary comparison uses the fixed **12/13/14/15** process sequence: baseline, geometry, geometry, baseline. Every process acquired the machine-wide exclusive GPU lock, inspected GPU activity inside that window, and retained the lock through initialization, warmup and cleanup. Each process was also preceded by ten observed idle one-second samples. One-second monitoring recorded no foreign GPU context or unrelated CPU process exceeding 0.25 cores in any of the four runs. Neither limiting cgroup accumulated throttled periods or throttled time during the recorded measurement windows. Cooperative locks and sampled process monitoring do not prove complete physical-machine isolation.

Each configuration has two warmup reconstructions followed by seven measured reconstructions per process, giving **14 synchronized wall-time samples per implementation**. Upload, JIT, output checking and garbage collection are outside the warm timing interval. Mesh extraction is separate and is not timed here. Both variants use the identical oriented input arrays/settings; `raw/bench-*.json` records input hashes, dataset provenance, warm samples, first-use timings and environment counters. The first-use values use an existing kernel cache and are not cold-compilation benchmarks. No new official-CPU timing or CPU/GPU speedup claim is made by this port comparison.

Query measurements cover one point, 4,096 points, and the complete input cloud. GPU-event measurements replay ten captured kernel launches ten times, reporting per-launch times; each process contributes seven samples. A separate public batch-call wall time includes allocations, host validation and synchronization. These are different measurements and should not be mixed.

| Case | Queries | Batch GPU ms, before → after | Composed GPU ms, before → after | Batch host µs, before → after |
| --- | ---: | ---: | ---: | ---: |
| Bunny d8 float | 1 | 0.0311 → 0.0311 | 0.0309 → 0.0309 | 101.5402 → 101.1696 |
| Bunny d8 float | 4,096 | 0.0631 → 0.0630 | 0.0629 → 0.0628 | 133.8925 → 136.3084 |
| Bunny d8 float | 362,271 | 0.2023 → 0.1993 | 0.2017 → 0.2014 | 271.1322 → 271.5513 |
| Horse d8 float | 1 | 0.0271 → 0.0271 | 0.0270 → 0.0270 | 98.7500 → 98.2732 |
| Horse d8 float | 4,096 | 0.0765 → 0.0767 | 0.0763 → 0.0766 | 148.5050 → 151.0568 |
| Horse d8 float | 100,000 | 0.1998 → 0.2021 | 0.1994 → 0.2018 | 271.6873 → 275.4871 |
| Eagle d8 float | 1 | 0.0292 → 0.0291 | 0.0291 → 0.0290 | 98.7966 → 98.7742 |
| Eagle d8 float | 4,096 | 0.0627 → 0.0632 | 0.0625 → 0.0629 | 134.0192 → 137.2062 |
| Eagle d8 float | 796,825 | 0.3178 → 0.3206 | 0.3413 → 0.3431 | 345.7163 → 345.9696 |
| Eagle d9 float | 1 | 0.0320 → 0.0320 | 0.0319 → 0.0318 | 101.4993 → 102.8012 |
| Eagle d9 float | 4,096 | 0.0843 → 0.0842 | 0.0840 → 0.0841 | 157.1439 → 158.0063 |
| Eagle d9 float | 796,825 | 0.5051 → 0.4985 | 0.4733 → 0.4713 | 573.2384 → 556.3125 |
| Bunny d8 double | 1 | 0.0349 → 0.0351 | 0.0345 → 0.0346 | 102.1363 → 103.4085 |
| Bunny d8 double | 4,096 | 0.0768 → 0.0779 | 0.0765 → 0.0779 | 148.8533 → 153.4671 |
| Bunny d8 double | 362,271 | 0.2613 → 0.2609 | 0.3105 → 0.3105 | 333.8736 → 335.4158 |


All reconstruction hashes (tree keys, coefficients and isovalue) and query hashes (composed values, batch values, spatial gradients and validity masks) match bitwise across all 16 process reports, including the primary comparison. `kernel-audit.json` additionally records 186/186 matching ordered kernel/device-function expression trees after normalizing renamed identifiers and omitting decorators, annotations and docstrings. This structural audit complements numerical tests; it does not prove compilation or host-path equivalence.

### Retained earlier measurements

All 16 initial process reports are retained, including contaminated and superseded runs. Runs 0–7 precede the final host-validation optimization. Runs 0 and 6 observed unrelated work and are excluded from performance conclusions; the earliest clean counterbalanced subset was 1/3/4/5. Runs 8–11 measure the final production source, launched before the new GPU-lock rule arrived; their clean ABBA comparison is in `prelock-performance-summary.json`. Runs 12–15 apply the new lock rule and are the primary report. Selection depends on execution order, source revision and observed contention, never on choosing faster timings.

### Interleaved diagnostic

A further exclusive-lock process interleaved both implementations on the same input arrays. Its batch-kernel comparison uses the **same field, query and output buffers**, removing allocation/address differences between the kernel measurements. The host-wrapper comparison calls the raw and public evaluators on the same result. No foreign work was observed. These query checks distinguish GPU kernel cost from the new public validation/tape guards.

| Case | Queries | Batch GPU change | Public wrapper overhead (µs) |
| --- | ---: | ---: | ---: |
| Bunny d8 float | 4,096 | +0.009% | 1.067 |
| Bunny d8 float | 362,271 | +0.073% | 0.425 |
| Horse d8 float | 4,096 | +0.008% | 1.006 |
| Horse d8 float | 100,000 | -0.025% | 1.341 |
| Eagle d8 float | 4,096 | -0.005% | 1.466 |
| Eagle d8 float | 796,825 | +0.030% | 1.404 |
| Eagle d9 float | 4,096 | +0.010% | 0.831 |
| Eagle d9 float | 796,825 | +0.081% | 1.807 |
| Bunny d8 double | 4,096 | +0.019% | 1.345 |
| Bunny d8 double | 362,271 | -0.019% | 0.585 |

The kernel differences are below 0.1%; the host checks add about 0.4–1.8 µs in this diagnostic. This supports preservation of the fast batch GPU route, while explicitly accounting for the public host contract.

The same process also alternated reconstruction, retaining the other variant’s last result. That changes allocation/lifetime conditions relative to the primary separate-process comparison; these wall times are diagnostic, not a replacement for the primary experiment:

| Case | Baseline (ms) | Geometry API (ms) | Change |
| --- | ---: | ---: | ---: |
| Bunny d8 float | 96.734 | 97.296 | +0.58% |
| Horse d8 float | 57.015 | 56.842 | -0.30% |
| Eagle d8 float | 74.175 | 74.715 | +0.73% |
| Eagle d9 float | 195.376 | 200.741 | +2.75% |
| Bunny d8 double | 129.155 | 128.229 | -0.72% |

The horse and double-bunny increases in the primary run reverse here, while eagle d9 rises 2.75% despite being 0.49% faster in the primary run. This indicates sensitivity to process/allocation conditions; it does not establish a zero-cost host API or a stable reconstruction speedup. No arithmetic or solver-iteration change was made in response. The primary and diagnostic data are both retained. An earlier diagnostic driver was superseded during a change to acquire the admission lock before its idle checks; its incomplete monitoring is excluded from conclusions.


`raw/*-isolation.json` retains sampled PIDs, memory use, timing and CPU-equivalent activity. Executable paths are reduced to basenames and cgroup identifiers to ancestry labels, so unrelated project paths are not published. Numerical values and throttling counters are unchanged.

## Validation

- **715 external numerical tests passed**, six inapplicable CPU variants skipped, one existing expected failure. The skipped variants need CUDA Nanogrid/classification or CUDA streams. The expected failure is the existing mixed-precision FEM gradient reproducer (`uniform_space` float64 on the default float32 grid); it is not counted as a compatibility pass. The adapter only remaps package/kernel names to the relocated runtime.
- **125 distinct native tests passed**: 22 feature/API tests, plus 103 related compiler, marching-cubes and FEM checks. The combined suite initially ran 124 tests; the additional fixed-reference test passed separately, then all 22 feature tests passed again after the final host-wrapper optimization. The external numerical suite and the 103 related tests preceded that host-only optimization; their numerical code did not change.
- The fixed-reference test checks double-precision field values, spatial gradients and isovalues against existing official C++ fixtures for screened and unscreened sphere reconstruction. `reference-provenance.json` records the C++ commit, input/oracle hashes and selected query indices; the Warp test embeds small literal values and needs no external binary or dataset.
- **Three Compute Sanitizer cases passed with zero errors**: ordinary composed device queries, cross-stream reconstruction derivatives and sparse mesh derivatives. This run preceded the final upstream-main rebase; the subsequent native rebuild and related tests passed, and the relevant port kernel expressions were unchanged.
- The docs build passed with warnings treated as errors; pre-commit passed for the complete feature and final edits. The headless example produced 9,922 vertices and 19,840 triangles. The ASV setup, reconstruction and evaluation smoke passed.

`validation.json` preserves source/build identifiers, observed result excerpts and hashes of the original logs. Large logs and disposable generated documentation were removed after publishing this record.

## Reproduction

The saved scripts retain the paths used for this experiment. On this machine, run them from `/home/horde/projects/warp-psr`, with the existing Warp source environment at `/home/horde/projects/warp-field-composition/.venv`. On another machine, adjust these two roots in the scripts; do not copy native binaries between Warp revisions. Inputs are the existing `.build/gpu-baselines/inputs/{bunny,horse,eagle}-full-d8-spsr` fixtures, whose sources and hashes are in the raw reports. Reuse local datasets; restore missing inputs with the documented dataset tools rather than substituting a different point cloud.

Copy the scripts to a fresh `.build/geometry-port` scratch directory if needed, then run:

```bash
uv run --no-project --python ../warp-field-composition/.venv/bin/python \
  python .build/geometry-port/run_benchmarks.py --offset 12 --gpu-lock
uv run --no-project --python ../warp-field-composition/.venv/bin/python \
  python .build/geometry-port/summarize.py --runs 12 13 14 15
uv run --no-project --python ../warp-field-composition/.venv/bin/python \
  python .build/geometry-port/run_paired.py
```

The benchmark runner takes a separate bounded exclusive lock for each child and checks activity inside the window. A lock timeout or observed activity defers the process; never bypass the lock. Existing machine rules require a shared `gpu_run` lock for correctness tests and GPU compilation. With that function defined as specified in machine `AGENTS.md`, the native suite is:

```bash
# Run from the Warp worktree.
gpu_run shared uv run --no-sync python -m unittest \
  warp.tests.geometry.test_poisson_reconstruction \
  warp.tests.test_func_inline \
  warp.tests.geometry.test_marching_cubes \
  warp.tests.geometry.test_sparse_marching_cubes \
  warp.tests.fem.test_fem_fp64
```

For the broader development suite, run `scripts/validate_port.py` from this repository with `PYTHONPATH` containing the Warp worktree, this repository's `src`, and this repository's root, using the Warp environment and a shared GPU lock. `scripts/audit_kernels.py` is CPU-only and requires no GPU lock. The focused upstream tests are native `unittest`; the external adapter uses this repository's existing pytest suite.
