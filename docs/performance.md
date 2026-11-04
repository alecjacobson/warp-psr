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

The depth-9 eagle now takes **0.491 s**, versus **4.884 s with four CPU threads**
(9.96×), with a **3.96 GiB** peak pool. The
[direct screening comparison](results/screening-performance.json) measured
0.854 s before and 0.490 s after eliminating fine-level FEM point assembly.
The earlier [GPU pipeline comparison](results/gpu-pipeline-performance.json)
measured 2.451 s for the original implementation and 0.860 s after its first
optimization pass. All use the original 796,825 samples, depth 9, full depth 4
and eight iterations. All 31 public accuracy cases pass on the final screening
source without changing their tolerances.

Exact degree-one stiffness stencils and direct point screening avoid global FEM
triplet sorting. Constraint positions are binned on the GPU, allowing multiple
aggregates per cell and centroids that round across a cell boundary. Fine levels
using stencils allocate no integration grid; small reference-sensitive levels
retain FEM assembly. One-dimensional stiffness integrals are computed once per
coordinate rather than once per matrix entry. Tensor-product normal integrals
also avoid materializing quadrature geometry. Cached
CUDA graphs replay the original sliced relaxation order, and cooperative loads
retain the reference reduction arithmetic. Sparse per-level cells and reduced tree
seed expansion cut temporary storage. CUDA events measure stages without forcing
the host to synchronize at every stage boundary; these stage durations include
submission gaps and are not pure kernel timings.

All GPU timings are fresh. CPU timings preserve their individual provenance with
identical inputs, parameters, CPU quota and reference builds. Both depth-9 OpenMP
cases were remeasured in isolation; the torso result replaces a previously
excluded repeat that overlapped an unrelated compilation.

A clean warmed Nsight trace of direct screening attributes 130.7 ms to 31,904
ordered relaxation graph kernels, 41.9 ms to unsigned-key radix sorting and
33.0 ms to signed-key radix sorting. Direct point screening costs 14.0 ms.
These are summed kernel durations from a profiled run, not wall-clock benchmarks.
A separate device-only row-sizing experiment measured 0.499 s versus 0.490 s
and was not adopted. Remaining relaxation work is tracked in
[#64](https://github.com/alecjacobson/warp-psr/issues/64).

To profile a warmed reconstruction with Nsight Systems:

```sh
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
  --cuda-event-trace=false --output=.build/eagle-profile \
  python -m tools.profile_reconstruction --fixture .build/fixtures/eagle-full-d9-spsr \
  --mode nsight --output .build/eagle-profile
```

Use an otherwise idle machine. `--mode cpu` records Python submission costs instead.

## Full original inputs at depth 8

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Horse PSR | 100,000 | 8 | 1.5 | 362,534 | 369,408 | 0.125 | 0.883 | 0.544 | 0.38 |
| Horse SPSR | 100,000 | 8 | 1.5 | 362,534 | 369,408 | 0.130 | 1.014 | 0.606 | 0.38 |
| Bunny PSR | 362,271 | 8 | 1.5 | 1,175,586 | 1,185,936 | 0.243 | 2.759 | 1.610 | 1.42 |
| Bunny SPSR | 362,271 | 8 | 1.5 | 1,175,586 | 1,185,936 | 0.264 | 3.269 | 1.986 | 1.43 |
| Eagle PSR | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.178 | 2.018 | 1.207 | 1.05 |
| Eagle SPSR | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.179 | 2.361 | 1.327 | 1.05 |
| Torso PSR | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.137 | 1.580 | 1.116 | 0.82 |
| Torso SPSR | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.144 | 1.820 | 1.190 | 0.82 |

## Point count sweep for torso SPSR at depth 8

The original input order is retained within each reproducible subset. Only the
sample count changes; the depth and density settings remain fixed.

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k subset | 10,000 | 8 | 1.5 | 139,850 | 145,696 | 0.101 | 0.452 | 0.349 | 0.14 |
| 100k subset | 100,000 | 8 | 1.5 | 446,482 | 453,504 | 0.136 | 1.254 | 0.703 | 0.54 |
| 1m subset | 1,000,000 | 8 | 1.5 | 479,386 | 486,472 | 0.151 | 1.543 | 0.945 | 0.65 |
| Full original | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.144 | 1.820 | 1.190 | 0.82 |

## Density sweep for full eagle SPSR at depth 8

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Target 1.5 | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.179 | 2.361 | 1.327 | 1.05 |
| Target 3 | 796,825 | 8 | 3 | 832,378 | 839,776 | 0.185 | 2.377 | 1.319 | 1.05 |
| Target 10 | 796,825 | 8 | 10 | 822,450 | 829,824 | 0.184 | 2.386 | 1.427 | 1.04 |

## Depth sweep with identical points and density settings

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eagle depth 8 | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.179 | 2.361 | 1.327 | 1.05 |
| Eagle depth 9 | 796,825 | 9 | 1.5 | 3,323,866 | 3,331,592 | 0.491 | 9.367 | 4.884 | 3.97 |
| Torso 1m depth 8 | 1,000,000 | 8 | 1.5 | 479,386 | 486,472 | 0.151 | 1.543 | 0.945 | 0.65 |
| Torso 1m depth 9 | 1,000,000 | 9 | 1.5 | 2,051,674 | 2,059,232 | 0.343 | 6.414 | 3.200 | 2.60 |

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
