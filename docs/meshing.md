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

`extract_mesh(..., differentiable=True)` currently supports a directly specified
`LevelField` with tracked coefficients and no cumulative coefficients; its
transform stays fixed. A one-element tracked float32 `iso_value` device array is
subtracted before extraction at zero, retaining threshold gradients.
`LevelField.evaluate(..., requires_grad=True)` also tracks output values and
spatial gradients with respect to tracked array inputs.

**This does not yet differentiate reconstruction from input points.** Current
`Reconstruction` objects explicitly reject differentiable meshing. The assembly,
solver, and transform backward path remains
[issue #76](https://github.com/alecjacobson/warp-psr/issues/76); chaining it to mesh
vertices remains part of [issue #77](https://github.com/alecjacobson/warp-psr/issues/77).
No converged-system adjoint is silently substituted for the finite-iteration solve.

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
