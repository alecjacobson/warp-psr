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

The depth-9 eagle now takes **0.856 s**, versus **4.717 s with four CPU threads**
(5.51×), with a **7.14 GiB** peak pool. An independent
[before/after run](results/gpu-pipeline-performance.json) measured 2.451 s for the
previous GPU implementation and 0.860 s for this implementation, a 2.85× improvement.
Both use the original 796,825 samples, depth 9, full depth 4 and eight iterations.
All 31 public accuracy cases pass without changing their tolerances.

Exact degree-one stiffness stencils avoid global FEM triplet sorting. Tensor-product
normal integrals avoid materializing quadrature geometry for those loads. Cached
CUDA graphs replay the original sliced relaxation order, and cooperative loads
retain the reference reduction arithmetic. Sparse per-level cells and reduced tree
seed expansion cut temporary storage. CUDA events measure stages without forcing
the host to synchronize at every stage boundary; these stage durations include
submission gaps and are not pure kernel timings.

The final torso depth-9 OpenMP repeat overlapped an unrelated compilation. Its
table entry retains the previously published isolated 3.368 s measurement with
identical input, parameters, CPU quota and reference build. The JSON records both
the retained provenance and the excluded repeat; the other 44 timings are fresh.

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
| Horse PSR | 100,000 | 8 | 1.5 | 362,534 | 369,408 | 0.140 | 0.883 | 0.544 | 0.48 |
| Horse SPSR | 100,000 | 8 | 1.5 | 362,534 | 369,408 | 0.173 | 1.014 | 0.606 | 0.65 |
| Bunny PSR | 362,271 | 8 | 1.5 | 1,175,586 | 1,185,936 | 0.292 | 2.759 | 1.610 | 1.76 |
| Bunny SPSR | 362,271 | 8 | 1.5 | 1,175,586 | 1,185,936 | 0.379 | 3.269 | 1.986 | 2.68 |
| Eagle PSR | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.217 | 2.018 | 1.207 | 1.28 |
| Eagle SPSR | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.278 | 2.361 | 1.327 | 1.96 |
| Torso PSR | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.163 | 1.580 | 1.116 | 0.91 |
| Torso SPSR | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.208 | 1.820 | 1.190 | 1.47 |

## Point count sweep for torso SPSR at depth 8

The original input order is retained within each reproducible subset. Only the
sample count changes; the depth and density settings remain fixed.

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k subset | 10,000 | 8 | 1.5 | 139,850 | 145,696 | 0.150 | 0.452 | 0.349 | 0.24 |
| 100k subset | 100,000 | 8 | 1.5 | 446,482 | 453,504 | 0.188 | 1.254 | 0.703 | 0.97 |
| 1m subset | 1,000,000 | 8 | 1.5 | 479,386 | 486,472 | 0.206 | 1.543 | 0.945 | 1.27 |
| Full original | 3,488,432 | 8 | 1.5 | 492,578 | 499,680 | 0.208 | 1.820 | 1.190 | 1.47 |

## Density sweep for full eagle SPSR at depth 8

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Target 1.5 | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.278 | 2.361 | 1.327 | 1.96 |
| Target 3 | 796,825 | 8 | 3 | 832,378 | 839,776 | 0.283 | 2.377 | 1.319 | 1.96 |
| Target 10 | 796,825 | 8 | 10 | 822,450 | 829,824 | 0.285 | 2.386 | 1.427 | 1.94 |

## Depth sweep with identical points and density settings

| Case | Points | Depth | Density target | Active coefficients | Support nodes | GPU (s) | C++ serial (s) | C++ OpenMP 4 (s) | Peak pool (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eagle depth 8 | 796,825 | 8 | 1.5 | 832,634 | 840,032 | 0.278 | 2.361 | 1.327 | 1.96 |
| Eagle depth 9 | 796,825 | 9 | 1.5 | 3,323,866 | 3,331,592 | 0.856 | 9.367 | 4.717 | 7.14 |
| Torso 1m depth 8 | 1,000,000 | 8 | 1.5 | 479,386 | 486,472 | 0.206 | 1.543 | 0.945 | 1.27 |
| Torso 1m depth 9 | 1,000,000 | 9 | 1.5 | 2,051,674 | 2,059,232 | 0.588 | 6.414 | 3.368 | 5.07 |

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
