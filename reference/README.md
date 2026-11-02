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
isovalue and transform. This tests instrumentation neutrality; equality with a
separate CLI invocation is an additional check, not implied by that assertion.

Files in each fixture directory:

- `input.pts`: quantized positions and normals in preserved order.
- `queries.txt`: grid interiors, random volume points and input samples.
- `oracle.txt`: `iso`, `transform` and `q` (value + model-space gradient).
- `oracle_instrumented.txt`: the same plus `node` rows (depth, x, y, z,
  flags, inherited ghost status, incidental reference index, coefficient).
- `stages.txt`: resolved settings, density scale, aggregated sample position sums
  and weights, canonical density and normal coefficients, and unscreened FEM load.
- `manifest.json`: revision, source/build settings, checksums, seeds and hardware.

Mathematical keys are `(depth,x,y,z)`. Negative/support coordinates are preserved
in exports. Array order and the incidental node index are not compatibility keys.
The `transform` is cube-to-model in the reference's column-first indexing.

Current omissions are tracked in issue #1: accepted/rejected input IDs, complete
aggregation membership, normals before splatting, effective interpolation records,
level matrices/transfers/solver traces, parallel repeatability, and stronger profile
provenance. The float64 driver processes float64 input throughout, but float64 Warp
FEM assembly is blocked by issue #8. No end-to-end compatibility gate has passed.
