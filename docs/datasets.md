# Public point-cloud experiments

The dataset catalog comes from the [pinned PoissonRecon examples](https://github.com/mkazhdan/PoissonRecon/blob/262b0f539d404057d1f36e1adc07fc9388678899/README.md#usage-examples-with-sample-data).
Raw files are downloaded into ignored `.build/datasets/`, verified against
SHA-256 checksums in `tools/datasets.py`, and never bundled into this repository.

All [16 smoke cases](results/public-smoke.json) and
[15 scaling cases](results/public-scaling-accuracy.json) pass the fixed field,
gradient, isovalue, and exact active/support-key gates. The
[performance guide](performance.md) reports matched GPU, serial C++, and
four-thread OpenMP measurements for all 15 scaling cases. The
[gallery](gallery.md) compares independently reconstructed surfaces for each input.

| Dataset | Original points | Distribution and input |
| --- | ---: | --- |
| Horse | 100,000 | Curvature-weighted samples of a virtual horse; ASCII positions and normals |
| Stanford bunny | 362,271 | Merged range scans; normals estimated from scan connectivity |
| Eagle | 796,825 | Photogrammetry of a detailed statue; supplied normals and RGBA |
| Torso | 3,488,432 | Large scan; supplied normals; the separate envelope is excluded from this profile |

The eagle data is credited to the **EPFL Geometric Computing Laboratory**,
[Scanning 3D Statues from Photos](https://lgg.epfl.ch/statues.php).
Its source permits research use and credited scholarly renderings; commercial use
requires the source's permission. Dataset terms remain separate from this code.
The bunny originates from the Stanford 3D Scanning Repository; the oriented input
used here is the version distributed by PoissonRecon. Horse and torso are likewise
downloaded from PoissonRecon's example-data server.

## Reproduce the experiments

After installing the test and visualization extras and building the reference:

```sh
python -m tools.datasets horse bunny eagle torso
python -m tools.check_datasets --suite smoke --output .build/public-smoke.json
python -m tools.check_datasets --suite scaling --benchmark --output .build/public-scaling.json
```

The smoke suite contains 16 cases: each of four datasets, a deterministic
10,000-point subset, PSR and SPSR, and both precisions. The 15 scaling cases use
every original point at depth 8 for PSR and SPSR; independently vary torso sample
count (10,000, 100,000, 1,000,000, full), eagle density target (1.5, 3, 10), and
depth (8, 9); and include a million-point torso at depth 9. Full depth is 3 for
the smoke suite and 4 for scaling. `--case NAME` selects individual cases.

Subsets use NumPy's seeded uniform sampling without replacement, retain source
file order, and record a checksum of the selected indices. Supplied normals
are passed unchanged to reconstruction. No denoising, outlier removal, normal
reorientation, envelope constraints, color reconstruction, or surface trimming
is applied. The reconstruction itself applies the reference's normal processing.
The fixture manifest records ignored PLY properties, original and selected counts,
source and quantized-input checksums, complete parameters, and software revisions.

Every case compares independent official and GPU solves at every input sample,
random volume points, grid points, each integration depth, refinement interfaces,
cube boundaries, and empty regions. Acceptance includes raw and centered field
values, cell-width-scaled gradients, the isovalue, and **exact canonical active
and non-ghost support keys** at each level. The fixed design tolerances are shared
with synthetic regression tests. Raw gradient errors are reported separately.

`--benchmark` runs each passing case in fresh processes, with three timed warm
runs after the first invocation. GPU timing includes Python orchestration and
excludes upload and queries; the official timer brackets the unchanged library
solve and excludes parsing, evaluation, export, and startup. Both serial and
OpenMP official backends are measured and their actual thread settings recorded.
The OpenMP build uses the unchanged pinned sources with `-fopenmp`; its runtime
thread limit defaults to the visible CPU quota and affinity. `tools.check_openmp`
measures field and topology variability separately from serial acceptance.
Default-async timing remains available explicitly. These are solve comparisons, not end-to-end command-line mesh timings.
The GPU memory figure is the process CUDA-pool high-water mark. Cached compilation
may be reused on the first invocation; it is not a clean-cache compilation test.

To benchmark an existing fixture with exactly the validated data and settings:

```sh
python -m tools.benchmark --fixture .build/fixtures/eagle-full-d9-spsr
python -m tools.benchmark_reference --fixture .build/fixtures/eagle-full-d9-spsr --parallelism none
python -m tools.benchmark_reference --fixture .build/fixtures/eagle-full-d9-spsr --parallelism openmp --threads 4
```

The measurement VM exposes 128 logical CPUs but has a **four-core cgroup quota**.
The no-OpenMP reference defaults to 128 async workers here; those oversubscribed
results are not representative of unrestricted multicore throughput. New reports
record affinity, visible ancestor quotas, and throttling counters. The OpenMP
baseline explicitly limits the runtime to four threads. See [#50](https://github.com/alecjacobson/warp-psr/issues/50).

To separate accuracy runs from uncontended timing, run the scaling suite first
without `--benchmark`, then repeat with `--benchmark-only`. This checks the saved
production source hashes, exact-key acceptance, input checksum, and case settings
before timing. The acceptance record also binds the query checksum and every
reference export checksum to the complete fixture manifest, including precision.
Stale or modified inputs, settings, exports, queries, and production code require
a fresh compatibility check. Do not run other GPU jobs or CPU-heavy work
concurrently with benchmarks.

The [OpenMP report](results/openmp-scaling.json) checks every scaling case against
the serial field budgets and canonical topology. The
[repeatability report](results/openmp-repeatability.json) repeats the float32
eagle and float64 torso smoke cases three times each. To reproduce the latter:

```sh
python -m tools.check_openmp --threads 4 --repeats 3 --output .build/openmp-repeatability.json
```

## Visual comparison

`tools.export_grid` evaluates the unchanged official and independent Warp fields
on identical model-space coordinates. The binary official export is tested
against ordinary ASCII query evaluation in both precisions and leaves the solve,
transform, isovalue, and existing query output unchanged. Dense grids are strictly
offline visualization assets; production reconstruction remains adaptive.

`tools.render_flagship` extracts both fields with the same marching-cubes settings,
uses a common camera and lighting, and applies no mesh smoothing, decimation, or
trimming. Its adjacent volume slice shows absolute field error; its reported
maximum is over the entire grid, including empty space. The render's JSON sidecar
records field hashes, grid bounds, errors, extraction sizes, camera, and versions.

## Additional geometry and controlled input defects

The expanded catalog adds the Stanford drill, armadillo, dragon and Happy Buddha
reconstructions from the [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
These are **mesh-derived inputs**, distinct from the four oriented clouds above.
Stanford cautions that reconstructed meshes suppress acquisition defects; these
cases therefore expand geometry coverage, rather than representing untouched
scanner measurements. Source attribution and dataset terms remain with Stanford;
the code license does not replace the data's terms. Raw files are not bundled.

| Pinned mesh | Vertices | Triangles | Input feature |
| --- | ---: | ---: | --- |
| Drill | 1,961 | 3,855 | Thin, elongated geometry |
| Armadillo | 172,974 | 345,944 | Detailed articulated geometry |
| Dragon | 437,645 | 871,414 | Detailed geometry and holes |
| Happy Buddha | 543,652 | 1,087,716 | Detailed geometry and small connections |

Counts describe the actual checksum-pinned archive members, not every historical
version of those models. Archive and extracted-file SHA-256 checksums are pinned
in `tools/datasets.py`. Only the named member is extracted. Vertex normals are
derived from source triangle winding: float64 area-weighted sums traverse corner
0/1/2 and then stored face order, followed by unit normalization. Zero sums remain
zero. No welding, reorientation, resampling, or vertex removal is performed.
Normals are computed on the complete source mesh before selecting a subset.
The manifest records degenerate triangles, zero normals, ignored attributes,
original counts, selected-index hashes, and the derivation algorithm.

The `extended` suite contains 24 cases. Each model runs PSR/SPSR in float32 and
float64 at depth 7 with full depth 3, using 10,000 selected vertices (all 1,961 for
the drill). Each also runs both methods at depth 8/full depth 4 in float32 using
100,000 vertices (again the complete drill). Subsets preserve source order.

The `stress` suite adds 16 cases: four variants of the same 10,000-vertex armadillo
subset, each tested with PSR/SPSR and float32/float64 at depth 7/full depth 3:

| Variant | Precisely defined change |
| --- | --- |
| `position-noise` | Independent Gaussian position noise, standard deviation 0.002 of the selected input's bounding-box diagonal; normals unchanged |
| `normal-noise` | Add independent Gaussian components with standard deviation 0.15 to unit normals, then renormalize; zero normals stay zero |
| `missing-cap` | Keep points at or below the 85th percentile of source z; retained rows preserve source order |
| `outliers` | Append 1% additional points uniformly in the bounding box padded by 10% per axis, with independent random unit normals |

These are controlled defects, not simulated scanner ground truth. They occur
after source subsetting and before optional input scale/translation. The variant
uses a separate RNG seeded with `--seed + 1009`; its algorithm version, seed,
parameters, retained-index hash where applicable, and before/after counts are
recorded in `input_variant`. The `10k` stress-case names denote the base subset;
actual final counts are recorded explicitly. Source files remain unchanged.

```sh
python -m tools.datasets drill armadillo dragon buddha
python -m tools.check_datasets --suite extended --list
python -m tools.check_datasets --suite extended --output .build/extended.json
python -m tools.check_datasets --suite stress --output .build/stress.json
```

The existing 16 smoke and 15 scaling configurations are unchanged. `--suite all`
selects all 71 named configurations. New cases use fresh independent C++ solves
and the existing fixed field, gradient, isovalue and exact-key budgets, including
all diagnostic query families. Passing these checks establishes compatibility
with the pinned PSR/SPSR implementation; it does not assert that defective input
produces a high-quality or correctly oriented surface. Large generated fixtures
and raw sources stay in ignored persistent storage.
