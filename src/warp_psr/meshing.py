"""Optional GPU marching cubes. Importing reconstruction does not load this module.

The default sparse path visits every grid cell in bounded slabs, stores a one-bit
crossing mask, and samples only retained cell corners for Warp's sparse extractor.
It needs no signed-distance or Lipschitz assumption. All backends use Warp's
fixed marching-cubes case table; selection/connectivity carry no derivative.
"""
from dataclasses import dataclass
import math
import operator

import warp as wp
from warp.geometry import (IsoSurfaceMarchingCubes, sparse_marching_cubes,
                           sparse_marching_cubes_from_cells)
from warp.utils import array_scan

# Match Warp geometry's grid-position arithmetic, including fused operations.
wp.set_module_options({"fuse_fp": True})


@dataclass
class Mesh:
    vertices: wp.array
    indices: wp.array
    metadata: dict


@wp.func
def _corner(c: int):
    # Warp MC ordering: (000,100,110,010,001,101,111,011).
    x = int(c == 1 or c == 2 or c == 5 or c == 6)
    y = int(c == 2 or c == 3 or c == 6 or c == 7)
    return wp.vec3i(x, y, c // 4)


@wp.kernel
def _grid_points(ny: int, nz: int, start: int, lower: wp.vec3, upper: wp.vec3, step: wp.vec3,
                 points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    cell = wp.vec3(float(start + i // (ny * nz)), float((i // nz) % ny), float(i % nz))
    points[i] = wp.min(wp.max(lower + wp.cw_mul(step, cell), lower), upper)


@wp.kernel
def _mark_cells(values: wp.array(dtype=float), ny: int, nz: int, start: int,
                mask: wp.array(dtype=wp.uint32), status: wp.array(dtype=int)):
    i = wp.tid()
    x, y, z = i // ((ny - 1) * (nz - 1)), (i // (nz - 1)) % (ny - 1), i % (nz - 1)
    code = int(0)
    for c in range(8):
        q = wp.vec3i(x, y, z) + _corner(c)
        v = values[(q[0] * ny + q[1]) * nz + q[2]]
        if not wp.isfinite(v):
            wp.atomic_or(status, 1, 1)
        if v >= 0.0:
            code = code | (1 << c)
    if code != 0 and code != 255:
        index = ((start + x) * (ny - 1) + y) * (nz - 1) + z
        wp.atomic_or(mask, index // 32, wp.uint32(1) << wp.uint32(index % 32))


@wp.kernel
def _word_counts(mask: wp.array(dtype=wp.uint32), counts: wp.array(dtype=int)):
    i = wp.tid()
    counts[i] = wp.bit_count(mask[i])


@wp.kernel
def _total(offsets: wp.array(dtype=int), status: wp.array(dtype=int)):
    status[0] = offsets[offsets.shape[0] - 1]


@wp.kernel
def _emit_cells(mask: wp.array(dtype=wp.uint32), offsets: wp.array(dtype=int),
                ny: int, nz: int, cells: wp.array(dtype=wp.vec3i)):
    word = wp.tid()
    bits = mask[word]
    out = offsets[word] - wp.bit_count(bits)
    for b in range(32):
        if bits & (wp.uint32(1) << wp.uint32(b)):
            i = word * 32 + b
            cells[out] = wp.vec3i(i // ((ny - 1) * (nz - 1)),
                                  (i // (nz - 1)) % (ny - 1), i % (nz - 1))
            out += 1


@wp.kernel
def _cell_points(cells: wp.array(dtype=wp.vec3i), lower: wp.vec3, upper: wp.vec3, step: wp.vec3,
                 points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    c = cells[i // 8] + _corner(i % 8)
    p = lower + wp.cw_mul(step, wp.vec3(float(c[0]), float(c[1]), float(c[2])))
    points[i] = wp.min(wp.max(p, lower), upper)


@wp.kernel
def _clamp_points(points: wp.array(dtype=wp.vec3), lower: wp.vec3, upper: wp.vec3,
                   out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = wp.min(wp.max(points[i], lower), upper)


@wp.kernel
def _model_vertices(vertices: wp.array(dtype=wp.vec3), lower: wp.vec3, step: wp.vec3,
                    out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = lower + wp.cw_mul(step, vertices[i])


@wp.kernel
def _check_values(values: wp.array(dtype=float), status: wp.array(dtype=int)):
    i = wp.tid()
    if not wp.isfinite(values[i]):
        wp.atomic_or(status, 1, 1)


@wp.kernel
def _center(values: wp.array(dtype=float), iso: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = values[i] - iso[0]


def _grid(resolution, bounds):
    if isinstance(resolution, int) and not isinstance(resolution, bool):
        resolution = (resolution,) * 3
    try:
        shape = tuple(operator.index(v) for v in resolution)
    except (TypeError, ValueError):
        raise ValueError('resolution must be an integer or three integers >= 2') from None
    if len(shape) != 3 or any(isinstance(v, bool) for v in resolution) or min(shape) < 2:
        raise ValueError('resolution must be an integer or three integers >= 2')
    try:
        lo, hi = (tuple(float(v) for v in row) for row in bounds)
    except (TypeError, ValueError):
        raise ValueError('bounds must contain two finite 3D corners') from None
    if (len(lo) != 3 or len(hi) != 3 or not all(math.isfinite(v) for v in (*lo, *hi))
            or any(a >= b for a, b in zip(lo, hi))):
        raise ValueError('bounds must be finite with lower < upper on every axis')
    if math.prod(n - 1 for n in shape) > 2**31 - 1:
        raise ValueError('extraction cell count exceeds int32 indexing')
    if math.prod(shape) > 2**31 - 1:
        raise ValueError('grid node count exceeds int32 sampling indices')
    lower, upper = wp.vec3(*lo), wp.vec3(*hi)
    # Match Warp MC's float32 bounds/spacing arithmetic.
    step = wp.cw_div(upper - lower, wp.vec3(*(n - 1 for n in shape)))
    if not all(math.isfinite(float(step[a])) and step[a] > 0 for a in range(3)):
        raise ValueError('bounds/spacing must be representable in float32')
    return shape, lower, upper, step


def extract_isosurface(field, *, resolution, bounds, backend='sparse', device=None,
                       slab_size=16, lipschitz_bound=None):
    """Extract the zero set of a batched GPU callback ``field(points) -> values``.

    ``points`` are float32 ``wp.vec3`` model positions; returned values must be
    finite 1D float32 arrays on the same device. Subtract any (possibly tracked)
    isovalue in the callback. Vertices/flat triangle indices remain on device.

    ``sparse`` exhaustively classifies cells using streamed slabs and a bitmask,
    then calls Warp's explicit-cell sparse MC. It is conservative relative to
    the matched dense grid, including disconnected surfaces and non-SDF fields.
    ``dense`` materializes the full scalar grid for Warp's dense MC.
    ``lipschitz`` uses Warp's octree callback API and REQUIRES a caller-proven
    global bound in model coordinates; a sampled gradient maximum is not proof.

    Warp tracks vertex interpolation when the callback returns tracked values
    inside a Tape. Grid/cell selection, bounds and connectivity are fixed.
    Callbacks must be deterministic and must not mutate shared output buffers:
    backward needs each invocation's values. GPU count reads inside Warp's
    extractors remain; this wrapper never downloads field values or meshes.
    """
    if backend not in ('dense', 'sparse', 'lipschitz'):
        raise ValueError('backend must be dense, sparse, or lipschitz')
    shape, lower, upper, step = _grid(resolution, bounds)
    if not isinstance(slab_size, int) or isinstance(slab_size, bool) or slab_size < 1:
        raise ValueError('slab_size must be a positive integer')
    if backend == 'lipschitz':
        if lipschitz_bound is None or not math.isfinite(lipschitz_bound) or lipschitz_bound <= 0:
            raise ValueError('lipschitz backend requires a finite positive proven lipschitz_bound')
    elif lipschitz_bound is not None:
        raise ValueError('lipschitz_bound is only used by the lipschitz backend')
    device = wp.get_device(device)
    samples = 0

    def sample(points):
        nonlocal samples
        values = field(points)
        if (not isinstance(values, wp.array) or values.dtype != wp.float32
                or values.shape != (len(points),) or values.device != device or not values.is_contiguous):
            raise ValueError('field must return contiguous float32 values, one per point, on the query device')
        samples += len(points)
        return values

    metadata = {'backend': backend, 'resolution': shape, 'bounds': (tuple(lower), tuple(upper)),
                'precision': 'float32', 'cell_selection': 'fixed for backward'}
    with wp.ScopedDevice(device):
        if backend == 'lipschitz':
            status = wp.zeros(2, dtype=int, device=device)

            def bounded_sample(points):
                # Warp's octree pads the box to a power-of-two cell count.
                # f(clamp(p, box)) extends f without increasing its bound and
                # avoids calling a domain-limited field outside the requested box.
                clamped = wp.empty_like(points)
                wp.launch(_clamp_points, len(points), inputs=[points, lower, upper, clamped],
                          device=device, record_tape=False)
                values = sample(clamped)
                wp.launch(_check_values, len(values), inputs=[values, status], device=device, record_tape=False)
                return values

            vertices, indices, stats = sparse_marching_cubes(
                bounded_sample, *shape, lower=lower, upper=upper, threshold=0.0,
                lipschitz_bound=lipschitz_bound, device=device, return_stats=True)
            if status.numpy()[1]:
                raise ValueError('field returned nonfinite values inside extraction bounds')
            metadata.update(warp_stats=stats, lipschitz_bound=lipschitz_bound)
        elif backend == 'dense':
            if 3 * math.prod(shape) > 2**31 - 1:
                raise ValueError('dense MC edge indexing exceeds int32; use the sparse backend')
            points = wp.empty(math.prod(shape), dtype=wp.vec3, device=device)
            wp.launch(_grid_points, len(points), inputs=[shape[1], shape[2], 0, lower, upper, step, points], device=device)
            values = sample(points)
            status = wp.zeros(2, dtype=int, device=device)
            wp.launch(_check_values, len(values), inputs=[values, status], device=device, record_tape=False)
            if status.numpy()[1]:
                raise ValueError('field returned nonfinite values inside extraction bounds')
            vertices, indices = IsoSurfaceMarchingCubes.extract(values.reshape(shape), threshold=0.0,
                                                                lower=lower, upper=upper)
        else:
            nx, ny, nz = shape
            words = (math.prod(n - 1 for n in shape) + 31) // 32
            mask = wp.zeros(words, dtype=wp.uint32, device=device)
            status = wp.zeros(2, dtype=int, device=device)
            tracked = False
            for start in range(0, nx - 1, slab_size):
                width = min(slab_size, nx - 1 - start)
                points = wp.empty((width + 1) * ny * nz, dtype=wp.vec3, device=device)
                wp.launch(_grid_points, len(points), inputs=[ny, nz, start, lower, upper, step, points], device=device,
                          record_tape=False)
                values = sample(points)
                tracked = tracked or values.requires_grad
                wp.launch(_mark_cells, width * (ny - 1) * (nz - 1),
                          inputs=[values, ny, nz, start, mask, status], device=device, record_tape=False)
            counts = wp.empty(words, dtype=int, device=device)
            offsets = wp.empty_like(counts)
            wp.launch(_word_counts, words, inputs=[mask, counts], device=device, record_tape=False)
            array_scan(counts, offsets, inclusive=True)
            wp.launch(_total, 1, inputs=[offsets, status], device=device, record_tape=False)
            count, invalid = map(int, status.numpy())
            if invalid:
                raise ValueError('field returned nonfinite values inside extraction bounds')
            if count > (2**31 - 1) // 8:
                raise ValueError('retained corner count exceeds int32 indexing')
            metadata.update(retained_cells=count, slab_size=slab_size, selection_bytes=words * 12,
                            peak_slab_nodes=(min(slab_size, nx - 1) + 1) * ny * nz)
            if count:
                cells = wp.empty(count, dtype=wp.vec3i, device=device)
                wp.launch(_emit_cells, words, inputs=[mask, offsets, ny, nz, cells], device=device, record_tape=False)
                del mask, counts, offsets, points, values
                points = wp.empty(count * 8, dtype=wp.vec3, device=device)
                wp.launch(_cell_points, len(points), inputs=[cells, lower, upper, step, points], device=device,
                          record_tape=False)
                values = sample(points)
                grid_vertices, indices = sparse_marching_cubes_from_cells(
                    cells, values.reshape((count, 8)), cell_width=1.0, threshold=0.0, device=device)
                vertices = wp.empty_like(grid_vertices, requires_grad=grid_vertices.requires_grad)
                wp.launch(_model_vertices, len(vertices), inputs=[grid_vertices, lower, step, vertices], device=device)
            else:
                vertices = wp.empty(0, dtype=wp.vec3, device=device, requires_grad=tracked)
                indices = wp.empty(0, dtype=wp.int32, device=device)
    metadata.update(field_evaluations=samples, vertices=len(vertices), triangles=len(indices) // 3)
    return Mesh(vertices, indices, metadata)


def extract_mesh(reconstruction, *, resolution, bounds, iso_value=None, differentiable=False, **options):
    """Mesh a Reconstruction or LevelField without changing reconstruction cost.

    Warp MC currently takes float32 fields. Float64 inputs fail explicitly.
    An optional one-element device ``iso_value`` array retains threshold gradients.
    Differentiation is currently supported for a directly specified LevelField
    without cumulative coefficients. It differentiates its tracked coefficients
    and optional isovalue, with its transform fixed. The reconstruction solver's
    input-point backward pass is tracked separately in issue #76.
    """
    field = getattr(reconstruction, 'field', reconstruction)
    if field.coefficients.dtype != wp.float32:
        raise ValueError('Warp mesh extraction currently requires float32 fields')
    if differentiable and (field.cumulative is not None or not field.coefficients.requires_grad):
        raise ValueError('differentiable meshing requires a direct LevelField with tracked coefficients; '
                         'the reconstruction backward pass is not yet supported (#76)')
    device = field.coefficients.device
    if 'device' in options and wp.get_device(options['device']) != device:
        raise ValueError('mesh device must match the field device')
    options['device'] = device
    if iso_value is None:
        iso_value = field.iso_value
    if isinstance(iso_value, wp.array):
        if iso_value.dtype != wp.float32 or iso_value.shape != (1,) or iso_value.device != device:
            raise ValueError('iso_value must be one float32 value on the field device')
        if iso_value.requires_grad and not differentiable:
            raise ValueError('tracked iso_value requires differentiable=True')
        iso = iso_value
    else:
        if not math.isfinite(iso_value):
            raise ValueError('iso_value must be finite')
        iso = wp.full(1, float(iso_value), dtype=wp.float32, device=device)

    def sample(points):
        values, _, _ = field.evaluate(points, requires_grad=differentiable)
        out = wp.empty_like(values, requires_grad=differentiable)
        wp.launch(_center, len(values), inputs=[values, iso, out], device=device)
        return out

    return extract_isosurface(sample, resolution=resolution, bounds=bounds, **options)
