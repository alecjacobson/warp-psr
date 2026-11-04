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
accuracy generation, and other benchmark jobs. The current timing sweep also
monitors foreign CPU/GPU activity and retains interrupted attempts in the local
audit.

The depth-9 eagle now takes **0.215 s**, versus **4.884 s with four CPU threads**
(22.68×), with a **2.34 GiB** peak pool. The
[direct screening comparison](results/screening-performance.json) measured
0.854 s before and 0.490 s after eliminating fine-level FEM point assembly.
The earlier [GPU pipeline comparison](results/gpu-pipeline-performance.json)
measured 2.451 s for the original implementation and 0.860 s after its first
optimization pass. All use the original 796,825 samples, depth 9, full depth 4
and eight iterations. All 31 public accuracy cases pass on the final coalesced-stiffness
source without changing their tolerances.

Direct prolongation CSR avoids transfer triplets and sorting; topology-preserving
precision casts avoid rebuilding existing sparse rows. The
[CSR comparison](results/csr-performance.json) records the fresh timings and profile.

Exact degree-one stiffness stencils and direct point screening avoid global FEM
triplet sorting. Constraint positions are binned on the GPU, allowing multiple
aggregates per cell and centroids that round across a cell boundary. Fine levels
using stencils allocate no integration grid; small reference-sensitive levels
retain FEM assembly. One-dimensional stiffness integrals are computed once per
coordinate rather than once per matrix entry. Tensor-product normal integrals
also avoid materializing quadrature geometry. Cached
CUDA graphs batch independent slices while preserving the update order for every
coupled pair of slices. Within each slice, the four colors remain ordered.
Cooperative loads retain the reference reduction arithmetic. Sparse per-level cells and reduced tree
seed expansion cut temporary storage. CUDA events measure stages without forcing
the host to synchronize at every stage boundary; these stage durations include
submission gaps and are not pure kernel timings.

All GPU timings were refreshed after the stiffness change. CPU timings preserve
their individual provenance with identical inputs, parameters, CPU quota and reference builds. Both depth-9 OpenMP
cases were remeasured in isolation; the torso result replaces a previously
excluded repeat that overlapped an unrelated compilation.

The preceding wavefront Nsight trace measured **4,268 relaxation launches and 24.7 ms**,
down from 31,904 launches and 130.5 ms before batching. Degree-one operators
couple only adjacent slices: visit `t` of slice `z` depends on visit `t` of `z+1`
and visit `t-1` of `z-1`. Grouping by `q = 2*t + 2**depth - z` preserves these
dependencies while executing independent slices two cells apart together.
Tests require bitwise-identical results for full and sparse local operators,
both precisions, multiple iteration counts, fresh buffers and independent streams.

The final weighted mean now reduces blocks of 1,024 samples in parallel, then
reduces their double-precision partial sums on the GPU. Products retain the input
precision. This changes the summation order; the fixed isovalue and centered-field
budgets remain unchanged. Tests cover both precisions, block boundaries, widely
varying weights and repeatability against an independent compensated sum.
The new two-kernel reduction costs **0.009 ms**, compared with 14.6 ms for the
previous serial reduction. These are profiled kernel durations, not wall times.
The [isovalue comparison](results/isovalue-performance.json) includes a fresh
five-run eagle comparison (303 ms before, 276 ms after), all public timings,
validation and the isolation audit. The [relaxation comparison](results/wavefront-performance.json)
retains the preceding launch-count measurements.

Tree neighborhoods now deduplicate complete broods before expanding their eight
children. Sorted parent prefix ranges place children directly in lexicographic
order; active field keys are compacted in that same order. The next coarse level
reuses the already unique parents. This eliminates duplicate-child sort traffic,
the parent sort and the active-key sort without changing any support or active key.
Signed-key radix-sort onesweep kernels now total **8.51 ms**, versus 32.7 ms
in the preceding wavefront profile. The [tree comparison](results/broods-performance.json)
records the isolated five-run eagle comparison (270 ms before, 226 ms after), the
profile and all 31 unchanged public acceptance gates. Sixteen structural cases
cover sparse/signed layouts, zero normal seeds and depths through 15 in both precisions.

Stiffness filling now assigns neighboring matrix entries to neighboring GPU
threads. The neighbor search records each entry's row-local rank in one byte,
so the fill writes compact CSR rows without scanning or sorting them again.
The arithmetic for each entry is unchanged. Fill kernels take **3.59 ms**,
down from 18.5 ms; the extra rank writes increase neighbor-search time from about
4.0 to 6.1 ms. The [stiffness comparison](results/stiffness-performance.json)
records the five-run whole-reconstruction comparison (220 ms before, 210 ms after),
all 15 refreshed public GPU timings and all 31 fixed acceptance gates.

Applying the same layout to screening was slower and was not adopted. The
[screening layout experiment](results/screening-layout-experiment.json) measured
227 ms versus 209 ms for its paired baseline; screening kernels increased from
13.8 to 35.2 ms. That layout repeats the point-bin search for each matrix entry,
consistent with the extra work outweighing more contiguous writes. Its source
and measurements are retained on the experimental branch.

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
| Horse PSR | 100,000 | 8 | 1.5 | 362,534 | 369,408 | 0.067 | 0.883 | 0.544 | 0.24 |
| Horse SPSR | 100,000 | 8 | 1.5 | 362,534 | 369,408 | 0.072 | 1.014 | 0.606 | 0.24 |
| Bunny PSR | 362,271 | 8 | 1.5 | 1,175,586 | 1,185,936 | 0.098 | 2.759 | 1.610 | 0.84 |
| Bunny SPSR | 362,271 | 8 | 1.5 | 1,175,586 | 1,185,936 | 0.107 | 3.269 | 1.986 | 0.84 |
| Eagle PSR | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.087 | 2.018 | 1.207 | 0.63 |
| Eagle SPSR | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.095 | 2.361 | 1.327 | 0.63 |
| Torso PSR | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.076 | 1.580 | 1.116 | 0.56 |
| Torso SPSR | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.078 | 1.820 | 1.190 | 0.56 |

## Point count sweep for torso SPSR at depth 8

The original input order is retained within each reproducible subset. Only the
sample count changes; the depth and density settings remain fixed.

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k subset | 10,000 | 8 | 1.5 | 139,850 | 145,696 | 0.068 | 0.452 | 0.349 | 0.09 |
| 100k subset | 100,000 | 8 | 1.5 | 446,482 | 453,504 | 0.076 | 1.254 | 0.703 | 0.32 |
| 1m subset | 1,000,000 | 8 | 1.5 | 479,386 | 486,472 | 0.077 | 1.543 | 0.945 | 0.40 |
| Full original | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.078 | 1.820 | 1.190 | 0.56 |

## Density sweep for full eagle SPSR at depth 8

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Target 1.5 | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.095 | 2.361 | 1.327 | 0.63 |
| Target 3 | 796,825 | 8 | 3 | 832,378 | 839,776 | 0.097 | 2.377 | 1.319 | 0.63 |
| Target 10 | 796,825 | 8 | 10 | 822,450 | 829,824 | 0.092 | 2.386 | 1.427 | 0.63 |

## Depth sweep with identical points and density settings

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eagle depth 8 | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.095 | 2.361 | 1.327 | 0.63 |
| Eagle depth 9 | 796,825 | 9 | 1.5 | 3,323,866 | 3,331,592 | 0.215 | 9.367 | 4.884 | 2.34 |
| Torso 1m depth 8 | 1,000,000 | 8 | 1.5 | 479,386 | 486,472 | 0.077 | 1.543 | 0.945 | 0.40 |
| Torso 1m depth 9 | 1,000,000 | 9 | 1.5 | 2,051,674 | 2,059,232 | 0.149 | 6.414 | 3.200 | 1.52 |

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

## Root solvers and preconditioners

[Solver experiments](solver-experiments.md) compare Jacobi, block-Jacobi, FSAI
and optional cuDSS with all setup costs included. The default eagle and horse
cases skip root CG at the existing initial-residual threshold. A stricter fixture
exercises all seven methods; FSAI halves root iterations but changes its total
reconstruction time by only about 1.6 ms. Production defaults remain unchanged.
