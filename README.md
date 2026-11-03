# warp-psr

Poisson Surface Reconstruction (PSR) and Screened Poisson Surface Reconstruction
(SPSR) in NVIDIA Warp. The GPU builds its own adaptive tree, density and normal
fields, assembles through `warp.fem`, solves the reference's multilevel system,
and evaluates implicit values and gradients.

The float32 and float64 degree-1 implementations match the pinned official output on
sphere, sparse, and torus fixtures, including PSR and both interpolation modes.
Public-cloud checks also cover horse, Stanford bunny, eagle, and torso scans in
both precisions; see the [dataset experiments](docs/datasets.md).
[Explicit parameter checks](docs/results/profiles.json) also vary base/solve/kernel
depths, density, scale, screening, and solver controls.
Broader profiles and performance work are tracked in
[issues](https://github.com/alecjacobson/warp-psr/issues), following the
[design](Warp%20Poisson%20Reconstruction%20Design%20for%20Matching%20Implicit%20Fields.md).

## Getting started

Requires Python 3.10+ and an NVIDIA GPU supported by Warp 1.19.

```sh
git clone https://github.com/alecjacobson/warp-psr.git
cd warp-psr
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Linux x86-64 with a CUDA 12 driver, install NVIDIA's CUDA 12 wheel:

```sh
pip install 'https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
```

```python
import warp as wp
from warp_psr import reconstruct

wp.init()
# points and normals: sequences of 3D positions and oriented normal vectors.
p = wp.array(points, dtype=wp.vec3, device="cuda")
n = wp.array(normals, dtype=wp.vec3, device="cuda")
result = reconstruct(p, n, depth=6, full_depth=3, point_weight=2.0)
values, gradients, valid = result.evaluate(p)
print(result.iso_value)  # The reconstructed surface is value == iso_value.
```

Use `point_weight=0.0` for PSR. `gradients` are in input coordinates; out-of-domain
queries produce NaN and a false validity mask. Mesh extraction is external.
Use `dtype=wp.vec3d` for both input arrays to run the entire pipeline in float64,
including normalization, density, assembly, solving, and evaluation.
Production reconstruction uses Warp arrays, kernels, FEM, and sparse algebra;
NumPy, SciPy, and the C++ reference are used only by tests and visualization.

## Flagship: the eagle

![Official PoissonRecon and independent pure-Warp eagle reconstruction](docs/results/eagle-reconstruction.png)

The **796,825-point eagle scan** reconstructs at depth 9 with **3,323,866 basis
coefficients**. Both independent solves have exactly the same active and support
keys. Across **812,815 diagnostic queries**, including every input point and
refinement interfaces, maximum field error is **3.12e-6**; scaled gradient error
uses **29.4% of the fixed design tolerance**. The isovalues differ by **5.97e-8**.

The figure compares another **134,217,728 shared volume-grid samples**, with
maximum field error **2.95e-6**. Both panels use the same marching-cubes extraction,
material, camera, and lighting, with no mesh smoothing or trimming. See the
[field report](docs/results/eagle.json) and [render provenance](docs/results/eagle-reconstruction.json).
The scan is from the **EPFL Geometric Computing Laboratory**,
[Scanning 3D Statues from Photos](https://lgg.epfl.ch/statues.php), distributed
through PoissonRecon's example datasets.

## Reproduce the comparison

```sh
pip install -e '.[test,rendering]'
sudo apt-get install g++ libjpeg-dev libturbojpeg0-dev libpng-dev zlib1g-dev
python tools/build_reference.py
python tools/make_fixture.py --name eagle-full-d9-spsr --dataset eagle --depth 9 --full-depth 4 --grid 24
python -m tools.check_reconstruction .build/fixtures/eagle-full-d9-spsr
python -m tools.export_grid .build/fixtures/eagle-full-d9-spsr --resolution 512
python -m tools.render_flagship .build/fixtures/eagle-full-d9-spsr --resolution 512
```

The pinned reference is
[PoissonRecon 262b0f5](https://github.com/mkazhdan/PoissonRecon/tree/262b0f539d404057d1f36e1adc07fc9388678899),
and Warp is [1.19.0](https://github.com/NVIDIA/warp/tree/06ce5ea6d5e29590706c7d3d48135ffdcc730912).
Downloads and build products stay in ignored `.build/`. The large dense grid is
used only for visualization; the reconstruction itself is adaptive.

`python -m tools.validate` runs the synthetic reference, operator, preprocessing,
and reconstruction gates. `python -m tools.check_datasets --suite smoke` runs
[16 public-cloud PSR/SPSR checks](docs/results/public-smoke.json) in float32 and
float64. The [dataset guide](docs/datasets.md) describes full-cloud scaling runs,
matched benchmarks, source checksums, and reproducible subsets.

## Supported profile and known limits

The current API accepts matching float32 or float64 CUDA inputs, degree 1, Neumann boundaries,
confidence disabled, finite positions, and scale > 1. Reconstruction depth is
1 through 15; base and full depth may be zero. The constraint
`depth - full_depth <= 7` reflects Warp’s eight-level geometry limit.
Depth, full/base/solve/kernel depths, density target, screening weight,
interpolation mode, and iteration settings are explicit arguments. Unsupported
profiles fail before reconstruction. The alignment matches the reference library's
`alignDir=0`; CLI comparisons explicitly request that setting.

The pinned reference has three consequential quirks preserved for compatibility:
its normal splat's parent contribution goes to the same fine level
([#12](https://github.com/alecjacobson/warp-psr/issues/12)), and `--exact` discards
all point constraints because space flags are initialized too late
([#16](https://github.com/alecjacobson/warp-psr/issues/16)). Thus
`exact_interpolation=True` reproduces that pinned behavior. It does not provide
corrected exact screening. It also prolongs quadratic normal coefficients with
degree-1 test-space weights ([#35](https://github.com/alecjacobson/warp-psr/issues/35));
matching that transfer is essential for heavily clustered samples.

The end-to-end float64 diagnostic fixtures cover sphere PSR/SPSR/exact, a sparse
tree, and the torus. Maximum field error is **5.78e-15** across those fixtures.
Preprocessing and operator checks use `atol=1e-10, rtol=1e-9`; field, isovalue,
and cell-width-scaled gradient checks use `atol=1e-8, rtol=1e-6`.
The [validation report](docs/results/validation.json) records both precisions.

The [geometry checks](docs/results/geometry-profiles.json) also cover plane patches,
nearby sheets, dense clusters, disconnected components, and translated/scaled
inputs in both precisions. Named queries cover every integration depth, both
sides of refinement interfaces, cube boundaries, and regions far from samples.
Reports include raw and isovalue-centered fields and per-family gradient errors.

The [repeatability report](docs/results/repeatability.json) compares three fresh
GPU solves with three serial and three default-parallel official solves on a
float32 sphere and a float64 cluster. It records canonical active-basis hashes,
raw/centered fields, gradients, isovalues, and the actual thread settings.
Parallel variation is measured separately from the fixed serial compatibility
budgets. Run `python -m tools.check_repeatability` after generating the fixtures;
it is also included in `tools.validate`.

The [threshold checks](docs/results/thresholds.json) cover quantized positions on
both sides of dyadic cell boundaries, coincident and opposing normals, zero normals,
and adjacent density settings that change the pinned oracle's integer splat depth.
Both precisions pass exact aggregate and tree checks and three independent field
acceptance runs per case. Run `python -m tools.check_thresholds` to reproduce the
oracle search and all eight cases without changing the fixed numerical budgets.

Remaining work includes complete ghost/flag metadata and broader
performance experiments. Two Warp float64
issues have tested upstream proposals: [built-in B-spline gradients](https://github.com/NVIDIA/warp/pull/2219)
and [sparse-grid particle lookup](https://github.com/NVIDIA/warp/pull/2221).
The application uses public basis and explicit-coordinate quadrature interfaces
and needs no Warp fork.

## Performance

On an NVIDIA L40, quadrature reuse, occupied-cell point assembly, indexed
relaxation with reusable launch commands, and per-level integration reduce the 100,000-point warm median
from **2.68 s to 0.69 s** and peak device-pool usage from **19.46 GiB to
6.81 GiB**. These measurements include the corrected hierarchical normal loads. This follows the earlier tree optimization from 496 ms to 34 ms.

| Torus samples | Depth / full depth | Basis coefficients | Integration cells | Warp warm median | Official serial median | Peak device pool |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 8 / 4 | 911,002 | 1,223,993 | 0.69 s | 2.52 s | 6.81 GiB |
| 1,000,000 | 8 / 4 | 963,898 | 1,265,874 | 0.80 s | 3.09 s | 7.27 GiB |
| 1,000,000 | 9 / 4 | 3,770,690 | 5,035,948 | 2.79 s | 12.30 s | 28.32 GiB |

The [current measurements](docs/results/normal-level-performance.json)
and [earlier launch measurements](docs/results/launch-performance.json) include Python
orchestration and exclude input upload and queries. Warm medians use three runs
after the first invocation; the disk kernel cache may already be populated.
Memory is the CUDA pool's process high-water mark. The
[official timings](docs/results/reference-performance.json) measure only the
unchanged library solve on one Xeon Platinum 8362 thread, matching the accuracy
fixtures; input parsing, query evaluation, and export are excluded. Parallel
reference throughput remains unmeasured.

Grouping normal coefficients by depth removes repeated scans and empty
quadrature work. The [preceding normal-load measurements](docs/results/normal-performance.json)
record the earlier memory/time tradeoff and both source revisions.

```sh
python -m tools.benchmark --count 100000 --depth 8 --full-depth 4
python -m tools.benchmark_reference --count 100000 --depth 8 --full-depth 4
```

All larger reconstructions pass the same accuracy budgets. Maximum field error
is [1.02e-6 for 100,000 points](docs/results/torus-100k.json) and
[9.54e-7 for one million](docs/results/torus-1m.json), including every sample
position in the query set. At depth 9, the [million-point report](docs/results/torus-1m-d9.json)
records a maximum field error of 1.32e-6 and scaled gradient errors within 7.81%
of the budget. Gradient comparisons include exact knot planes.
The pinned reference intersects left-cell neighbors with right-cell polynomial
pieces at knots, which can differ from either ordinary one-sided derivative.
Reconstruction evaluation preserves this behavior and the reference's inward
boundary nudges; see [#20](https://github.com/alecjacobson/warp-psr/issues/20).
