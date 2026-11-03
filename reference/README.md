# Offline reference oracle

This directory is development tooling, never a production dependency. Sources are
fetched at the exact revision in `pins.json`. The build refuses a modified checkout.
`oracle` calls the unchanged official library; `oracle_instrumented` adds read-only
exports before temporary data is released. Both use C++17, `-O2`, no fast-math,
serial `ThreadPool::NONE`, and explicit solver settings. `PoissonReconCLI` is the
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
- `queries.txt`: grid interiors, random volume points and input samples.
- `oracle.txt`: `iso`, `transform` and `q` (value + model-space gradient).
- `oracle_instrumented.txt`: the same plus `node` rows (depth, x, y, z,
  flags, inherited ghost status, incidental reference index, coefficient).
- `preprocess.txt`: ordered aggregate position/normal sums and density coefficients
  before tree padding (which relocates the reference root coefficient).
- `splats.txt`: normalized sample positions, raw depth, density weight, top level,
  blend fraction, and weighted normals.
- `operators.txt` (opt-in `WARP_PSR_OPERATORS=/absolute/fixture/path`): per-level
  matrices, complete right-hand sides, and restriction matrices, keyed by exported
  reference indices that map back to canonical tree coordinates.
- `stages.txt`: resolved settings, density scale, aggregated sample position sums
  and weights, canonical density and normal coefficients, and unscreened FEM load.
- `manifest.json`: revision, source/build settings, checksums, seeds and hardware.

Mathematical keys are `(depth,x,y,z)`. Negative/support coordinates are preserved
in exports. Array order and the incidental node index are not compatibility keys.
The `transform` is cube-to-model in the reference's column-first indexing.

Current omissions are tracked in issue #1: complete acceptance/aggregation IDs,
ghost/parent/child metadata, solver traces, parallel repeatability, and broader
query families. Float32 independent reconstruction passes the small and large
field gates; see the [main README](../README.md) and committed reports. The
float64 reference driver processes float64 input throughout, while end-to-end
float64 Warp reconstruction remains unfinished. The built-in B-spline issue has
[a tested upstream fix](https://github.com/NVIDIA/warp/pull/2219); this repository
still uses the pinned, unmodified released wheel.

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
