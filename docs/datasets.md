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
