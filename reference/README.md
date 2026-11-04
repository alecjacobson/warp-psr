# Offline reference oracle

This directory is development tooling, never a production dependency. Sources are
fetched at the exact revision in `pins.json`. The build refuses a modified checkout.
`oracle` calls the unchanged official library; `oracle_instrumented` adds read-only
exports before temporary data is released. Both use C++17, `-O2`, no fast-math,
serial `ThreadPool::NONE` by default, and explicit solver settings. `PoissonReconCLI` is the
unmodified official command-line source compiled with its degree-1/Neumann profile.

```sh
sudo apt-get install g++ libjpeg-dev libturbojpeg0-dev libpng-dev zlib1g-dev
python tools/build_reference.py
python tools/make_fixture.py --name sphere
python -m tools.check_field .build/fixtures/sphere
```

`make_fixture.py` asserts bitwise equality of the two drivers' values, gradients,
isovalue and transform. This tests instrumentation neutrality; the separate `python -m tools.check_cli .build/fixtures/sphere` check also compares
the actual CLI grid with the unchanged library driver. Library `alignDir=0` and
CLI default `alignDir=2` differ, so the CLI comparison explicitly passes `0`.

Files in each fixture directory:

- `input.pts`: quantized positions and normals in preserved order.
- `inputs.txt`: one `input` row per original input: zero-based ID, status,
  accepted leaf depth and x/y/z. Status 0 is accepted, 1 invalid data, 2 outside
  the reconstruction cube, and 3 rejected by the sample processor. Rejected
  inputs without a leaf have depth -1. These read-only rows are checksum-bound
  to the fixture and checked against GPU classification and aggregate membership.
- `queries.txt`: visualization grid, random volume points, input samples, and
  named cell-interior, refinement-interface, boundary, and far-from-sample probes.
- `oracle.txt`: `iso`, `transform`, `q` (value + model-space gradient), and
  `parallel_async`, configured `threads`, `schedule`, and `chunk_size`.
- `oracle_instrumented.txt`: the same plus `node` rows (depth, x, y, z,
  flags, inherited ghost status, incidental reference index, coefficient).
- `topology.txt` (opt-in `make_fixture.py --topology`): `topology` rows containing
  global depth and x/y/z, local depth and x/y/z, flags, inherited ghost status,
  parent global depth and x/y/z, and child count (0 or 8). The root has parent
  coordinates and depth -1. Global coordinates uniquely identify the padding
  nodes at local depths -2 and -1; the ordinary local accessor returns -1 for
  all coordinates at those depths, so those local tuples cannot identify nodes.
- `preprocess.txt`: ordered aggregate position/normal sums and density coefficients
  before tree padding (which relocates the reference root coefficient).
- `splats.txt`: normalized sample positions, raw depth, density weight, top level,
  blend fraction, and weighted normals.
- `operators.txt` (opt-in `WARP_PSR_OPERATORS=/absolute/fixture/path`): per-level
  matrices, complete right-hand sides, and restriction matrices, keyed by exported
  reference indices that map back to canonical tree coordinates.
- `stages.txt`: resolved settings, density scale, aggregated sample position sums
  and weights, canonical density and normal coefficients, and unscreened FEM load.
- `manifest.json`: revision, source/build settings, checksums, seeds, hardware,
  query-family ranges, and measured depth/interface coverage (format version 2).

Mathematical keys are `(depth,x,y,z)`. Negative/support coordinates are preserved
in exports. Array order and the incidental node index are not compatibility keys.
The `transform` is cube-to-model in the reference's column-first indexing.

`result.diagnostics['samples']` retains GPU input statuses, input-to-group IDs
(-1 for rejected inputs), stable sorted input IDs, and aggregate offsets. These
are populated during the existing classification and ordered group reductions,
without additional host status reads. Groups use canonical leaf-key order and
retain original input order within each group. The reconstruction checker checks
these arrays when the fixture contains `inputs.txt`; older fixtures remain usable
for their existing field and tree gates.

`warp_psr.tree_metadata.build_tree_metadata(tree, samples, density, normals)`
materializes the complete retained hierarchy on the GPU when requested. Pass the
four arrays/structures from `result.diagnostics`. Each level has global keys,
flags, inherited ghost status, parent indices, and eight child indices in
x-bit-first order. Missing links are -1. Global depth is local depth plus two.
Retained input, density and normal broods include nodes pruned from the solve;
ordinary reconstruction only constructs its active support sets. This diagnostic
adds no work to ordinary reconstruction. `check_reconstruction` verifies every
global identity, flag and link when a fixture includes `topology.txt`.

Detailed solver iteration traces remain tracked in issue #1.
Float32 independent reconstruction passes the small and large
field gates; see the [main README](../README.md) and committed reports. The
float64 reference driver and Warp reconstruction both process float64 input
throughout. Five end-to-end double fixtures meet the design's diagnostic budgets,
including strict preprocessing and operator comparisons. The built-in B-spline
issue has [a tested upstream fix](https://github.com/NVIDIA/warp/pull/2219), as does
[float64 Nanogrid lookup](https://github.com/NVIDIA/warp/pull/2221). This repository
uses its public basis extension and explicit containing-cell coordinates on the
pinned, unmodified released wheel. A mixed-depth lookup test probes both sides
of an interface at separations smaller than float32 coordinate resolution.

`make_fixture.py --operators` additionally exports interpolation records, every
small per-level screened matrix and initial right-hand side, and restriction
matrices. Use it for small fixtures; large explicit dumps are expensive.
`reference/cells.py` constructs the canonical integration partition offline from
reference basis and normal supports, solely to assign the design's gradient
scaling widths. No reference tree enters production reconstruction.

The independent reconstruction report preserves the design's original
cell-width-scaled gradient budget. Imported-coefficient evaluator tests retain
their additional, stronger unscaled check. The torus reconstruction has a few
near-zero gradient components outside that stronger optional check, while every
component passes the design criterion; both raw errors and budget ratios are
reported. Float32 cofactor inversion is applied twice for world evaluation, as
in the pinned `unitCubeToModel.inverse()` API.

The pinned exact path's empty interpolation spans are verified against explicit
operator exports and the actual CLI. See issue #16 for the SPACE_FLAG ordering
bug. This repository does not silently substitute corrected exact screening.

At exact knots, strict tree ownership alone does not determine the pinned
reference gradient. Its leaf and neighbor queries choose the left cell while its
polynomial evaluator uses floor (right pieces). Reconstruction fields now use the
same three contributions: cumulative coefficients at leafDepth-1, corrections at
the leaf, and overlapping fine corrections at leafDepth+1. This reproduces the
neighbor/polynomial intersection, and the 1e-6 inward nudge at the cube boundary.
Direct standalone LevelField summation remains available as a mathematical field
utility; the Reconstruction API supplies the reference cumulative representation.
The offline coefficient adapter now uses that same convention and the reference
cofactor inverse; plane inputs provide exact-knot regression coverage (#34).

The pinned normal-load assembly also uses degree-1 test-space prolongation for
degree-2 normal coefficients. This matters when normals occupy several levels,
as in the clustered fixture (#35). Production reproduces its same-level,
fine-restriction, and cumulative-coarse contributions with Warp FEM. Per-level
quadrature is compared against the full integration partition in both precisions;
the clustered operator exports separately check every normal load against C++.

`make_fixture.py` first runs a read-only pilot to select queries from the canonical
reference support partition. It then runs both independent reference drivers on
the complete query set and requires bitwise agreement. The initial visualization
grid is preserved. `python -m tools.check_geometry_profiles` covers planes,
nearby sheets, nonuniform clusters, disconnected components, and translated/scaled
inputs in float32 and float64. Reconstruction reports check raw and centered
values and report maximum/RMS errors by query family and integration depth.

`python -m tools.check_profiles` varies each exposed parameter independently and
checks a stronger iteration schedule. `python -m tools.validate` includes these
profiles. The stronger schedule is run compatibility, not a proof of convergence.
All depth, density, normalization, screening, and solver arguments are passed
explicitly to both the oracle and CLI checks. Legacy oracle defaults are tested
for bitwise equality with their explicit counterparts.

Density-driven splat depths preserve the pinned `log(Real)` overload's result
precision before later divisions and additions. The float32 path evaluates the
logarithm internally in double and rounds its result to float32 at that point.
This avoids CUDA `logf` approximation differences on the diagnostic sphere,
whose raw depths and weights now match the reference bitwise (#44).

`python -m tools.check_thresholds` constructs quantized triples immediately below,
on, and above dyadic cell boundaries, alongside coincident, opposing, and zero
normals. It also bisects the serial oracle's density decision to adjacent
representable samples-per-node settings where a selected splat changes from
depth 4 to depth 3. The final fixtures verify that transition independently.
Both precisions compare aggregate keys and multiplicity, density, raw and integer
splat depths, normal coefficients, exact support/active trees, and three fresh
reconstructions per case. The reference's finalization can remove stored zero
normal coefficients; differing records must be exactly zero before comparing
the shared coefficients and final tree. The [report](../docs/results/thresholds.json)
retains the oracle search, exact input/query/export hashes, and every acceptance run.

Base-depth-zero profiles cover screening, PSR, the pinned exact mode, zero full
depth, a solve stopping at the root, repeated root cycles, and reconstruction
depth one in both precisions. Their preprocessing, canonical active/support
nodes, normal loads, operator actions, and transfers have separate checks.
For unscreened base depth zero, the pinned root CG operator is `K x + mean(x)`
on all 64 padded rows, including 56 inactive rows. Warp retains those rows for
this operation and extracts the eight active coefficients after the solve.
Its enabling condition compares the root row count against dimensions at the
requested base depth, so it is absent at higher base depths. Exact interpolation
still reports that it constrains the constant mode despite its empty spans, and
also omits this term. These reference decisions are preserved explicitly.
Repeated root cycles retain the previous solution and use its current residual
for the initial and relative stopping tests. No fitted gauge offset is applied
to the field comparisons.

Float32 base-depth-zero loads use ordered polynomial accumulation because the
root's cancellation exposes the reference's rounding order (#43). Same-level
gathers and restrictions follow lexicographic neighbor order; fine-to-parent
contributions follow z slices and Morton order within each slice. Public Warp
segmented sorting constructs that order on the GPU. Polynomial integrals are
checked against `basis --normal` and randomized multilevel loads against the
assembled FEM path in double precision. The latter remains the default for
other profiles and is available as `ordered_root=False` for diagnostics.

`python -m tools.benchmark_reference --count 100000 --depth 8 --full-depth 4`
measures the unchanged serial library solve using a timer around `Solver::Solve`.
Input parsing, query evaluation, export, and process startup are outside the
timer. The benchmark records the CPU, compiler, reference pin, parameters, input
checksum, and every run; it does not measure the parallel CLI configuration.

Add `--parallelism async` to measure the pinned build's default CLI backend
without OpenMP: `ThreadPool::ASYNC`, dynamic scheduling, 128-element chunks,
and the hardware thread count. The library may use fewer threads for a small
loop. The wrapper selects this existing public backend through
`WARP_PSR_REFERENCE_PARALLELISM=async`; no reference algorithms are changed.
Fixture generation explicitly selects serial execution regardless of that
environment variable. Invalid backend names fail.

`python -m tools.check_repeatability` separately compares three serial reference,
three default-parallel reference, and three GPU reconstructions on the same
queries. The default fixtures are `sphere` and `geometry-cluster-double`,
generated by `tools.validate`. Serial instrumentation must preserve values,
gradients, isovalue, and transform exactly. Every GPU run must match the serial
field budget and canonical active-basis keys; its own repeatability is checked
separately. Parallel variation is reported against the first parallel run and
against serial, including topology hashes. Parallel measurements never relax
the serial acceptance budget. This is a bounded fixture measurement, not a
claim of universal bitwise determinism.
