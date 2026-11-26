# Native aligned sparse mesh extraction

Written by Codex (AI agent). Measured 2026-11-26 for [issue #105](https://github.com/alecjacobson/warp-psr/issues/105).

Implemented in [NVIDIA/warp draft #2452](https://github.com/NVIDIA/warp/pull/2452), commit `5db07ad6d3c374b48f533ee1b5e46ee640865bf8`, on NVIDIA main `98d6b5ed56800213e94cc191f0c0588239d6049e` plus the existing compiler prerequisite. The standalone production implementation in this repository is unchanged; this work targets the geometry port.

## Alignment and conservative certificate

`poisson_extract_mesh(result)` samples exact dyadic coordinates in the reconstruction cube and applies its retained cube-to-model transform. Depth 8 means 257 nodes per axis; depth 9 means 513. Dense and sparse native extraction use that same lattice. An optional native resolution is `2**depth + 1` (maximum depth 15). Explicit world bounds plus resolution preserve independent uniform resampling.

The reference evaluator selects left-owned cells `(lower, upper]`. Within a leaf at level L, its field combines the level-L correction, the level-(L-1) cumulative polynomial, and the next-level correction. Missing coefficients contribute zero. Nonnegative trilinear weights give convex-hull intervals from eight coarse control values and up to 27 fine control values. Adding intervals remains conservative when cancellation makes a certificate inconclusive. Finer query cells restrict the combined polynomial to their eight corners.

Bounds reduce through the actual selector hierarchy; parent closure is validated on device. Fully refined cells use their children; unrefined portions also include the local polynomial. A temporary GPU hash table accelerates lookup without adding reconstruction work. Closed MC cells include their lower-face neighbors' bounds, covering exact-knot ownership and domain-boundary nudges.

Float64 bounds are widened by `128 * float32_epsilon * component_magnitude + 1e-35`, covering float32 evaluation rounding and subnormal flushing. Exact-zero coefficients have exact-zero bounds. Unsafe intermediate ranges and nonfinite data fail explicitly. A cell is discarded only if its lower bound is at least the isovalue, or its upper bound is below it, matching MC's nonnegative/negative convention.

Only uncertain cells subdivide. Target-depth candidates use actual corner signs; crossing cells all have one common width and go to `sparse_marching_cubes_from_cells`. Eight child cells share 27 samples instead of 64. Untracked sampling uses the existing device evaluator without allocating unused gradient arrays. The fast batched evaluator is unchanged. Tracked extraction resamples retained corners through the existing VJP and differentiates the isovalue, interpolation, and native normalization frame; selection/connectivity stay fixed.

Coverage is relative to matching dense MC, not recovery of sub-grid features. No signed-distance or Lipschitz assumption is used. Extraction retains an adaptive count read per refinement level and the count reads inside Warp MC.

## Performance

| Case | Points | Scan mesh (ms) | Certified mesh (ms) | Speedup | Scan points-to-mesh (ms) | Certified points-to-mesh (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bunny d8 | 362,271 | 10.004 | 5.634 | 1.78× | 109.463 | 103.668 |
| Bunny d9 | 362,271 | 82.632 | 21.791 | 3.79× | 323.657 | 262.169 |
| Horse d8 | 100,000 | 9.698 | 5.063 | 1.92× | 69.895 | 65.598 |
| Horse d9 | 100,000 | 72.881 | 11.825 | 6.16× | 157.389 | 95.687 |
| Eagle d8 | 796,825 | 10.099 | 6.198 | 1.63× | 93.816 | 83.876 |
| Eagle d9 | 796,825 | 83.985 | 20.669 | 4.06× | 271.391 | 212.857 |

Five warm synchronized wall-time samples per variant/case in separate processes. Both variants use the same float32 field, isovalue, cube lattice, sparse MC, and model transform. The baseline exhaustively classifies that lattice in 16-cell slabs using the retained pre-change resampling implementation. Upload, parsing, JIT, GC, and validation are outside timing. Points-to-mesh starts with resident inputs and includes reconstruction and extraction.

Each process held the machine-wide exclusive GPU lock and checked for unrelated GPU processes inside it before initialization. Hardware: L40 48 GB, driver 570.158.01, toolkit 12.6.85, Warp 1.21.0.dev0. Both limiting cgroups expose four CPU cores; all twelve processes recorded zero additional throttling. GPU exclusivity does not reserve CPU capacity. These are warm measurements, not CPU/GPU reconstruction speedups.

| Case | Candidate cells | Full uniform cells | Crossing cells | Pool high-water (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Bunny d8 | 767,472 | 16,777,216 | 187,922 | 863.7 |
| Bunny d9 | 3,129,400 | 134,217,728 | 758,377 | 3032.6 |
| Horse d8 | 343,696 | 16,777,216 | 84,097 | 248.0 |
| Horse d9 | 1,359,808 | 134,217,728 | 335,788 | 561.1 |
| Eagle d8 | 641,192 | 16,777,216 | 160,391 | 647.6 |
| Eagle d9 | 2,689,928 | 134,217,728 | 666,593 | 2401.8 |

Candidates occupy 1.0–4.6% of the finest uniform grid. Pool high-water marks include reconstruction and are identical between methods in each case: reconstruction dominates this metric. These do not isolate incremental mesh workspace; no measured whole-process peak-memory reduction is claimed.

The separate eagle d9 event diagnostic measures candidate construction at 8.679 ms, corner-value sampling at 2.418 ms, and complete sparse MC at 3.399 ms. Candidate/MC spans include host submission and count-read gaps. Instrumentation changes host overhead and is separate from primary timings; individual-kernel totals exclude native sort/scan calls. See [raw profile](raw/profile-eagle-d9.json).

## Validation

- All six real cases have bitwise-identical sorted crossing cells and corner values against the exhaustive scan. Supplementary canonical mesh digests rounded to 1e-6 also match.
- All 26 native API tests pass: dense/sparse geometry, a fine feature hidden behind same-sign coarse corners, adaptive interfaces, boundary knots, coarse/finer grids, constant-zero depth-15 early exit, invalid inputs, mesh derivatives, and normalization-frame translation derivatives.
- The adversarial regression fails with coarse-corner-only pruning; [mutation_check.py](mutation_check.py) reproduces the expected failure.
- Compute Sanitizer: certificate and sparse-backward tests, two cases, zero errors.
- Docs pass with warnings treated as errors; pre-commit, the headless example, and extraction/points-to-mesh benchmark smoke pass. Reconstruction and batched evaluator files are unchanged.

## Reproduce

Use the existing geometry worktree environment and offline fixtures in `.build/gpu-baselines/inputs`. From this repository, define `gpu_run` from `/home/horde/AGENTS.md`, then run:

```bash
gpu_run exclusive env \
  PYTHONPATH=/home/horde/projects/warp-field-composition:/home/horde/projects/warp-psr \
  uv run --no-project --python /home/horde/projects/warp-field-composition/.venv/bin/python \
  python docs/results/native-meshing/benchmark.py \
  --dataset eagle --depth 9 --variant native --repeats 5 --output eagle-native.json
```

Use `--variant slabs` for the exhaustive aligned baseline; repeat for bunny/horse/eagle and depths 8/9. `--profile` enables the separate instrumented diagnostic. The script records input provenance, source hashes, exact crossing hashes, samples, cgroup limits/counters, and pool high-water marks. See [summary.json](summary.json) and the thirteen reports in `raw/`.
