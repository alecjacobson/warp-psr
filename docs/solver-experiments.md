# Root solver experiments

The default eagle and horse reconstructions **do not invoke CG**: after coarse
relaxation, the eight-row root residual is already below the reference's initial
threshold. Preconditioning that skipped solve cannot accelerate these cases.
Fine levels use dependency-preserving Gauss–Seidel relaxation.

To exercise the root, this experiment changes `cg_accuracy` to `1e-8` in both the
pinned oracle and GPU configuration for a 10,000-point horse subset at depth 6,
full/base depth 3, and eight relaxation iterations. All seven variants pass the
unchanged field, centered-field, gradient, isovalue and exact-tree gates.

| Root method | CG iterations | Warm reconstruction median (ms) |
| --- | ---: | ---: |
| Reference CG | 8 | 65.20 |
| Jacobi | 8 | 66.43 |
| Block-Jacobi, direct | 8 | 66.68 |
| Block-Jacobi, sequential | 8 | 65.64 |
| Block-Jacobi, tiled | 8 | 65.92 |
| FSAI | 4 | 63.55 |
| cuDSS | direct | 68.15 |

These are five-run medians after the first reconstruction, including Python,
preconditioner construction and factorization setup. Each method receives fresh
matrix data; no factorization is reused between reconstructions. Inputs and query
evaluation are excluded from timing. The monitored runs used an L40 and the same
four-core CPU quota as the public benchmarks. The recorded source predates the
wavefront optimization; these measurements compare root methods with each other,
not with later production timings.

FSAI halves root iterations and saves about 1.6 ms (2.5%) on this strict fixture.
Jacobi and block-Jacobi offer no measured improvement; cuDSS costs more with setup
included. The root is only eight unknowns, so this result does not predict behavior
for a large sparse direct solve. Block-Jacobi uses two explicit 4×4 blocks because
applying it directly to scalar CSR would reduce to ordinary Jacobi. The reference
and Jacobi hit the eight-iteration cap before their very small residual tolerance;
the report records the residual and stopping tolerance without claiming convergence.

Production defaults remain unchanged and require only Warp. The optional cuDSS
experiment uses `nvidia-cudss-cu12==0.8.0.10` and
[warp-cuDSS e73e3bc](https://github.com/alecjacobson/warp-cuDSS/tree/e73e3bc8db7943c7dfccea2624f32d5dc044f599).
The [complete report](results/root-solvers.json) records fixture/build provenance,
all warm samples, root iteration/residual counts, accuracy and isolation checks,
and the optional library's hash. This addresses [#67](https://github.com/alecjacobson/warp-psr/issues/67).

Reproduce in an idle environment:

```sh
python -m tools.make_fixture --name solver-horse-tight --dataset horse \
  --max-points 10000 --depth 6 --full-depth 3 --cg-accuracy 1e-8 --grid 16
python -m tools.compare_root_solvers --fixture .build/fixtures/solver-horse-tight \
  --method fsai --output .build/root-solvers/fsai
```

Other method names are `reference`, `diag`, `block_jacobi_direct`,
`block_jacobi_sequential`, `block_jacobi_tile`, and `cudss`. Install the optional
cuDSS package and adapter separately to run the final method. `--validate-only`
checks accuracy and whether the method was actually invoked, without timing.
