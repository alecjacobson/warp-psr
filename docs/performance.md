# Public cloud performance

All 15 cases below passed the fixed serial/GPU field, centered-field, gradient,
isovalue, and exact active/support-key gates. The [complete report](results/public-scaling.json)
contains original/subset input hashes, effective parameters, per-level node counts,
integration cells, per-stage GPU timings, individual runs, and environment provenance.
The [four-thread reference checks](results/openmp-scaling.json) also pass on every
case; [separate repeated runs](results/openmp-repeatability.json) quantify its variability.

Measurements use an NVIDIA L40 and Xeon Platinum 8362 host. Although 128 logical
CPUs are visible, this VM has a **four-core cgroup quota**. OpenMP is explicitly
limited to four threads; the pinned default-async backend would configure 128
workers. We do not use that oversubscribed backend to claim GPU speedups.
Both C++ builds use GCC 11.4, `-O2`, and no fast-math; the parallel build adds
`-fopenmp`. These are the recorded compatibility builds, not the upstream
Makefile's `-O3 -ffast-math` release configuration.

GPU medians use three warm reconstructions after the first invocation in a fresh
process. Timings include Python orchestration and exclude input upload and queries.
The first invocation may reuse compiled kernels from disk. C++ medians use three
subsequent fresh-process solves after an initial run; the timer brackets the
unchanged `Solver::Solve`, excluding parsing, queries, exports, and process startup.
Memory is the GPU process CUDA-pool high-water mark, including temporaries, not a
measurement of every CUDA allocation. Timing runs were isolated from rendering,
accuracy generation, and other benchmark jobs.

## Full original inputs at depth 8

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Horse PSR | 100,000 | 8 | 1.5 | 362,534 | 369,408 | 0.414 | 0.869 | 0.544 | 2.90 |
| Horse SPSR | 100,000 | 8 | 1.5 | 362,534 | 369,408 | 0.454 | 1.008 | 0.608 | 2.96 |
| Bunny PSR | 362,271 | 8 | 1.5 | 1,175,586 | 1,185,936 | 0.784 | 2.751 | 1.718 | 8.65 |
| Bunny SPSR | 362,271 | 8 | 1.5 | 1,175,586 | 1,185,936 | 0.913 | 3.312 | 1.873 | 9.10 |
| Eagle PSR | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.588 | 2.017 | 1.220 | 5.94 |
| Eagle SPSR | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.722 | 2.459 | 1.349 | 6.29 |
| Torso PSR | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.425 | 1.602 | 1.121 | 3.51 |
| Torso SPSR | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.516 | 1.833 | 1.211 | 3.88 |

## Point count sweep for torso SPSR at depth 8

The original input order is retained within each reproducible subset. Only the
sample count changes; the depth and density settings remain fixed.

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k subset | 10,000 | 8 | 1.5 | 139,850 | 145,696 | 0.342 | 0.453 | 0.350 | 1.30 |
| 100k subset | 100,000 | 8 | 1.5 | 446,482 | 453,504 | 0.484 | 1.249 | 0.700 | 3.30 |
| 1m subset | 1,000,000 | 8 | 1.5 | 479,386 | 486,472 | 0.498 | 1.539 | 0.931 | 3.65 |
| Full original | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.516 | 1.833 | 1.211 | 3.88 |

## Density sweep for full eagle SPSR at depth 8

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Target 1.5 | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.722 | 2.459 | 1.349 | 6.29 |
| Target 3 | 796,825 | 8 | 3 | 832,378 | 839,776 | 0.713 | 2.354 | 1.471 | 6.29 |
| Target 10 | 796,825 | 8 | 10 | 822,450 | 829,824 | 0.701 | 2.394 | 1.709 | 6.27 |

## Depth sweep with identical points and density settings

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eagle depth 8 | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.722 | 2.459 | 1.349 | 6.29 |
| Eagle depth 9 | 796,825 | 9 | 1.5 | 3,323,866 | 3,331,592 | 2.424 | 9.358 | 4.607 | 24.89 |
| Torso 1m depth 8 | 1,000,000 | 8 | 1.5 | 479,386 | 486,472 | 0.498 | 1.539 | 0.931 | 3.65 |
| Torso 1m depth 9 | 1,000,000 | 9 | 1.5 | 2,051,674 | 2,059,232 | 1.652 | 6.463 | 3.368 | 15.37 |

## Earlier torus experiments

The torus remains a regression and historical performance fixture. Earlier
optimizations reduced a 100,000-point depth-8 warm solve from 2.68 s to 0.69 s
and its peak pool from 19.46 GiB to 6.81 GiB. Those measurements belong to their
recorded revisions and are not substituted for the public-cloud measurements above.

See [per-level normal integration](results/normal-level-performance.json),
[reusable launches](results/launch-performance.json), [per-level geometry](results/level-performance.json),
[FEM assembly](results/fem-performance.json), and [tree construction](results/tree-performance.json).
The [historical serial CPU measurements](results/reference-performance.json)
use the explicitly recorded backend and compiler configuration.
