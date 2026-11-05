# GPU mesh extraction

Mesh extraction is opt-in and lives in `warp_psr.meshing`. Reconstructing a field
does not import the meshing module, allocate mesh buffers, or record a tape.
The API uses geometry extractors already present in the pinned Warp 1.19.0.

```python
from warp_psr.meshing import extract_mesh

# Explicit model-space bounds inside the field's normalization cube.
# Resolution counts grid nodes, not cells.
mesh = extract_mesh(result, resolution=512, bounds=(model_min, model_max),
                    backend="sparse")
vertices = mesh.vertices   # wp.array(dtype=wp.vec3), on the field's device
indices = mesh.indices     # flat wp.int32 array, three entries per triangle
```

The domain clips the surface: a surface leaving it has an open boundary.
Warp's fixed MC table determines ambiguous cases and winding; faces point toward
increasing centered field values. Both endpoints of an edge exactly on the
threshold are classified nonnegative. A grid entirely equal to the threshold
produces an empty mesh. Features smaller than grid cells may remain unresolved.
An exact-threshold endpoint can produce coincident vertices belonging to distinct
grid edges and zero-area triangles. The API preserves Warp's output; it does not
weld vertices or remove triangles. The gradient contract excludes these case
transitions.

## Backends and coverage

| Backend | Selection and storage | Required field assumption |
| --- | --- | --- |
| `dense` | Full scalar grid; `IsoSurfaceMarchingCubes.extract` | Finite values at grid nodes |
| `sparse` (default) | Slabs, crossing mask, `sparse_marching_cubes_from_cells` | Finite, deterministic grid values |
| `lipschitz` | Octree and batched `sparse_marching_cubes` callback | A caller-proven global Lipschitz bound |

The default sparse path examines **every** extraction cell. It retains exactly
those whose eight corner signs differ, with the same classification as Warp MC.
It cannot omit a crossing cell from the matched dense grid, including cells in
disconnected components or across reconstruction refinement interfaces. It does
not treat reconstruction support nodes as extraction cells or assume an SDF.

Field values are streamed through `slab_size` cell layers (default 16); the shared
node layer is resampled at slab boundaries. A bitmask and two scan arrays use
approximately **3/8 byte per extraction cell**. After compaction, retained cell
corners are sampled for Warp. Warp shares corner and edge identities across
cells and slab boundaries. Indices name uniform cells by their minimum corner,
not NanoVDB voxel regions. Extraction in integer grid coordinates followed by a
model-space mapping supports anisotropic bounds.

This trades additional corner evaluations for a smaller working set than dense
MC. Selection storage still scales with volume; this is not a surface-only
memory algorithm. It avoids a dense scalar field and host field/mesh transfers.
The wrapper reads one combined count/validity result after selection. Warp also
reads allocation counts and cell bounds. Extraction is not synchronization-free
or CUDA-graph-capturable.

The `lipschitz` backend requires an explicit `lipschitz_bound`. SPSR fields are
not generally signed distances, and a sampled maximum gradient does not prove a
bound. The bound must hold in **model coordinates** throughout the box. Warp pads
its octree to a power-of-two cell count; the wrapper extends the callback by
clamping outside queries to the box, which does not increase a valid bound.
Ordinary grid queries are also clamped at endpoints to avoid roundoff overshoot.
No automatic Lipschitz bound is claimed for reconstructed SPSR fields.

## Differentiation

`extract_isosurface(callback, resolution=..., bounds=..., device=...)` takes a
batched callback from `wp.array[wp.vec3]` to contiguous float32 values on the same
device, and extracts its **zero set**. Tracked values returned inside `wp.Tape`
allow vertex gradients through edge interpolation and the callback's parameters.
Cell selection, cases, connectivity, resolution, and bounds are fixed during
backward. Crossing a case boundary is nondifferentiable; indices have no gradient.

Each callback invocation must return its own unchanged output buffer: a tape can
need earlier invocations during backward. Streaming selection can retain extra
callback intermediates when a tape is active. Forward-only callbacks should
allocate untracked arrays.

For `backend="sparse"`, `classification_field` can supply the same scalar values
without recording gradients; only retained corners then call the tracked `field`
callback. Both callbacks must agree at every sampled grid node. Classification
uses the existing finite-value/coverage checks and remains a fixed decision.
`extract_mesh` supplies this untracked classifier automatically for differentiable
reconstruction results, using the same device transforms and isovalue. No tape
state is suspended or modified. `classification_field=None` explicitly retains
the generic single-callback behavior.

`extract_mesh(..., differentiable=True)` also supports a directly specified
`LevelField` with tracked coefficients and no cumulative coefficients; its
transform stays fixed. A one-element tracked float32 `iso_value` device array is
subtracted before extraction at zero, retaining threshold gradients.
For these non-cumulative fields, `LevelField.evaluate(..., requires_grad=True)` tracks output values and
spatial gradients with respect to tracked array inputs.

Results from `reconstruct_differentiable` retain the finite reconstruction
backward pass. Use `extract_mesh(..., differentiable=True)` on the same tape to
propagate dense/sparse MC vertex losses to input positions and normals; see the
[reconstruction example and contract](differentiability.md). Ordinary
`Reconstruction` objects still reject differentiable meshing because their
forward did not retain that connection. No converged-system adjoint is
substituted for the finite-iteration solve.

Warp MC consumes float32 corner values and returns float32 vertices. Float64
reconstruction fields are rejected instead of silently rounding or detaching
them. Empty meshes have zero vertices and indices. Nonfinite sampled values,
inconsistent callback arrays, invalid bounds, and int32 index overflows fail
explicitly.

## Validation and measurement

`tests/test_meshing.py` checks CPU/CUDA dense/sparse vertex and oriented triangle
parity, closed-surface edge incidence, all 256 MC cases, empty/threshold fields,
disconnected non-SDF fields, anisotropic grids, slab interfaces, and actual
adaptive reconstruction. Mesh losses are checked against central differences
away from case changes, including shared corners, coefficients/isovalue, and
independent tape reuse.

```sh
python -m tools.benchmark_meshing --fixture .build/fixtures/eagle-full-d9-spsr \
  --resolution 512 --backend sparse --output .build/eagle-mesh-sparse.json
```

This records reconstruction, field-to-mesh, and combined wall times separately,
with individual first/warm runs and peak device-pool usage. It refuses foreign
GPU compute workloads and profiles field evaluation and MC separately after
timing. Input upload, bounds preparation, and validation/downloads are excluded.
The published reconstruction-only 0.215-second result is not a mesh timing.

### Full eagle measurements

The 796,825-point eagle uses the published depth-9, full-depth-4, eight-iteration
reconstruction on L40 with Warp 1.19.0+cu12. Each row uses a fresh process and five
warm runs after the first invocation. The complete per-run evidence is in the
[mesh report](results/gpu-meshing-performance.json).

| Extraction grid | Backend | Field-to-mesh | Reconstruction + mesh | Peak device pool |
| --- | --- | ---: | ---: | ---: |
| 512³ nodes | Dense Warp MC | 277.7 ms | 488.1 ms | 7.568 GiB |
| 512³ nodes | Sparse Warp MC | 95.6 ms | 309.6 ms | 2.344 GiB |
| 1024³ nodes | Sparse Warp MC | 785.2 ms | 1,009.8 ms | 2.392 GiB |

Sparse field-to-mesh is 2.91× faster than dense at 512³. These wall times include
field evaluation, selection, allocation, and extraction. Separately instrumented
CUDA elapsed times are 91.0 ms for field evaluation and 2.75 ms for sparse MC;
they include submission gaps and exclude query generation/centering and final
coordinate mapping. The sparse 512³ run evaluates 147,593,240 points, retaining
656,131 cells and producing 656,754 vertices and 1,312,404 triangles. At 1024³ it
retains 2,648,227 cells and produces 2,649,395 vertices and 5,296,568 triangles.

The matched-grid check verifies every crossing cell, every retained corner value,
unique sorted cell indices, raw vertex/triangle counts, and oriented geometric
triangles. Dense/sparse maximum vertex displacement is 1.284e-6 model units.
For the triangle comparison only, near-coincident positions are grouped within
2e-6; raw mesh indices and counts remain unchanged. The CPU Lewiner extractor
has a different ambiguity policy: it produces 656,752 vertices and 1,312,392
triangles, with maximum bidirectional nearest-vertex displacement 1.493e-6.
Its recorded 1.568-second warm extraction excludes field evaluation and transfer.
All three meshes have 1,110 boundary edges, all on the extraction box, and no
edges shared by more than two faces. The raw float32 meshes contain 16 zero-area
triangles for dense Warp, 30 for sparse Warp, and 22 for CPU Lewiner; endpoint
rounding can collapse these tiny triangles differently. Extraction preserves
them rather than performing an implicit cleanup.

All three 512³ meshes have centered-field residual RMS about 0.01174 and maximum
about 0.14594 when evaluated in the reconstructed field. These are grid/linear
interpolation errors, not the reconstruction's reference-field error. The
1024³ sparse mesh reduces the RMS to about 0.002883 and maximum to about 0.06091.
The [coverage report](results/gpu-meshing-validation.json) records the measurements
and raw mesh edge incidence, including threshold degeneracies.

All [31 public reconstruction gates](results/gpu-meshing-public.json) pass with
unchanged tolerances and the original input/query hashes. The local suite and
restored-fixture rerun total 387 passing tests and one expected upstream failure.
Fresh-process A–B–B–A reconstruction-only checks use five warm runs per process:

| Case | Baseline A | Candidate B | Candidate B | Baseline A |
| --- | ---: | ---: | ---: | ---: |
| Sphere | 45.46 ms | 45.88 ms | 43.18 ms | 44.39 ms |
| Torus | 48.94 ms | 50.30 ms | 48.90 ms | 48.88 ms |
| Eagle | 208.21 ms | 218.17 ms | 203.93 ms | 223.39 ms |

The timings show no reproducible slowdown; device-pool peaks are byte-identical
for each case. The baseline's production source is identical to pre-mesh main.
Individual runs and environment data are included in the mesh report.

Reproduce the geometric and CPU comparison after installing the visualization
dependencies:

```sh
python -m tools.check_meshing --fixture .build/fixtures/eagle-full-d9-spsr \
  --resolution 512 --output .build/eagle-mesh-validation.json
```

## Complete mesh-loss backward measurements

The full eagle is measured with a loss equal to the mean vertex dot product with
`(1, 0.3, -0.2)`. Each backend/resolution uses a separate isolated process with
five warm repetitions after the first invocation. Input upload, bounds and loss
seed preparation, and validation are excluded. Full backward includes MC, field
and reconstruction gradients to input positions/normals. Peak pool includes the
retained tape and excludes native CUDA graph storage.

The [performance report](results/mesh-backward-performance.json) contains every
run and a separate profile of sampling, MC, mesh/field backward and reconstruction
backward. The initial costs in [PR #83's report](results/reconstruction-mesh-adjoints.json)
are retained for comparison:

| 512³ backend | Initial backward | Current backward | Speedup | Initial peak | Current peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 5.649 s | 0.561 s | 10.08× | 23.041 GiB | 23.041 GiB |
| Sparse | 6.231 s | 0.518 s | 12.02× | 19.117 GiB | 5.860 GiB |

Sparse classification now uses untracked slabs and retains only the corner
samples needed for interpolation. The reverse passes skip exactly zero incoming
gradients before basis traversal and shared transform/isovalue atomic additions.
Nonzero spatial-gradient seeds retain their Hessian contributions. Callbacks own
array primals without retaining the field wrapper in a tape reference cycle.
This addresses [#84](https://github.com/alecjacobson/warp-psr/issues/84).

| Grid / backend | Reconstruction | Field-to-mesh | Full backward | Forward + backward | Peak pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512³ dense | 202.6 ms | 126.3 ms | 0.561 s | 0.902 s | 23.041 GiB |
| 512³ sparse | 208.9 ms | 99.7 ms | 0.518 s | 0.839 s | 5.860 GiB |
| 1024³ sparse | 208.3 ms | 809.9 ms | 0.569 s | 1.587 s | 8.443 GiB |

The 512³ outputs have 656,754 vertices and 1,312,404 triangles; 1024³ sparse has
2,649,395 vertices and 5,296,568 triangles. These are complete tracked execution
costs, separate from the forward-only measurements above and from reconstruction-
only backward timings. The 1024³ tracked sparse case is measured directly; no
performance claim is made for an unmeasured dense 1024³ backward run.

The [validation report](results/mesh-backward-validation.json) records 660 passing
tests, six CPU parameterization skips and one expected upstream failure. All 31
public comparisons retain exact default, tracked and untracked-classification
field results against archived reference-validated outputs; accuracy budgets are
unchanged. Full-eagle 512³ tracked and forward-only vertices and indices are
bitwise equal for each backend. Derivative tests cover value and spatial-gradient
losses, explicit tracked isovalues, empty meshes, repeated tapes and streams.
Every enabled benchmark checks finite gradients and the normal-scale invariant.

Forward-only sparse 512³ uses A–B–B–A ordering (A: PR #83, B: this implementation):

| Phase | A₀ | B₀ | B₁ | A₁ |
| --- | ---: | ---: | ---: | ---: |
| Reconstruction | 208.7 ms | 226.3 ms | 206.7 ms | 230.4 ms |
| Field-to-mesh | 94.2 ms | 95.7 ms | 95.9 ms | 94.4 ms |
| Combined | 305.6 ms | 314.7 ms | 304.0 ms | 324.8 ms |

Default peak pool is byte-identical across all four processes (2.344 GiB).
The individual timings and first invocations are retained; these are measured
results on L40, not portable CI performance thresholds.

```sh
python -m tools.benchmark_mesh_adjoint --fixture .build/fixtures/eagle-full-d9-spsr \
  --resolution 512 --backend sparse --repeats 5 --output .build/eagle-mesh-backward.json
```
