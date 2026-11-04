"""Pinned-reference normal-field splatting on GPU (including documented issue #12)."""

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
import time
import warp as wp
import warp.utils
from .samples import point_div
from .field import basis_key
from .keyed import reduce_records, Records
from .density import quadratic

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _depth_offsets(keys: wp.array(dtype=wp.int64), offsets: wp.array(dtype=int)):
    d = wp.tid()
    target = wp.int64(d) << wp.int64(48)
    lo = int(0)
    hi = keys.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    offsets[d] = lo


@lru_cache(None)
def precision_kernels(dtype):
    vec2 = wp.types.vector(length=2, dtype=dtype)
    vec3 = wp.types.vector(length=3, dtype=dtype)
    vec4 = wp.types.vector(length=4, dtype=dtype)

    @wp.kernel
    def _input_order(
        first: wp.array(dtype=int), keys: wp.array(dtype=int), ids: wp.array(dtype=int)
    ):
        i = wp.tid()
        keys[i] = first[i]
        ids[i] = i

    @wp.kernel
    def _splats(
        sums: wp.array(dtype=vec3),
        normals: wp.array(dtype=vec3),
        weights: wp.array(dtype=dtype),
        dw: wp.array(dtype=vec2),
        order: wp.array(dtype=int),
        minimum: int,
        maximum: int,
        keys: wp.array(dtype=wp.int64),
        values: wp.array(dtype=vec4),
    ):
        i, k = wp.tid()
        src = order[i]
        dst = i * 54 + k
        keys[dst] = wp.int64(9223372036854775807)
        values[dst] = vec4(dtype(0.0))
        w = weights[src]
        normal = point_div(normals[src], w)
        length = wp.length(normal)
        if length > dtype(0.0):
            normal = point_div(normal, length) * w
            p = point_div(sums[src], w)
            depth = wp.clamp(dw[src][0], dtype(minimum), dtype(maximum))
            top = int(wp.ceil(depth))
            dx = dtype(1.0) - (dtype(top) - depth)
            if top <= minimum:
                top = minimum
                dx = dtype(1.0)
            if top > maximum:
                top = maximum
                dx = dtype(1.0)
            fraction = dx
            if k >= 27:
                fraction = dtype(1.0) - dx
            if k < 27 or wp.abs(wp.float64(1.0) - wp.float64(dx)) > wp.float64(1e-06):
                # The pinned reference's Splat lambda ignores its node argument
                # and uses captured temp for both calls (issue #12).
                n = 1 << top
                h = dtype(1.0) / dtype(n)
                c = wp.vec3i(
                    wp.max(0, int(wp.ceil(p[0] * dtype(n))) - 1),
                    wp.max(0, int(wp.ceil(p[1] * dtype(n))) - 1),
                    wp.max(0, int(wp.ceil(p[2] * dtype(n))) - 1),
                )
                t = (p - vec3(dtype(c[0]) * h, dtype(c[1]) * h, dtype(c[2]) * h)) / h
                bx, by, bz = (
                    quadratic(wp.float64(t[0])),
                    quadratic(wp.float64(t[1])),
                    quadratic(wp.float64(t[2])),
                )
                q = k % 27
                x, y, z = (q // 9, q // 3 % 3, q % 3)
                target = c + wp.vec3i(x - 1, y - 1, z - 1)
                if wp.min(target) >= 0 and wp.max(target) < n:
                    v = normal * dw[src][1] / (h * h * h) * fraction
                    v *= dtype(bx[x] * by[y] * bz[z])
                    keys[dst] = basis_key(top, target[0], target[1], target[2])
                    values[dst] = vec4(v[0], v[1], v[2], dtype(0.0))

    @wp.kernel
    def _negative(values: wp.array(dtype=vec4)):
        i = wp.tid()
        values[i] = -values[i]

    @wp.kernel
    def _average(
        normals: wp.array(dtype=vec3),
        weights: wp.array(dtype=dtype),
        dw: wp.array(dtype=vec2),
        order: wp.array(dtype=int),
        out: wp.array(dtype=vec3),
    ):
        total = vec2(dtype(0.0))
        weight = dtype(0.0)
        for i in range(weights.shape[0]):
            s = order[i]
            if wp.length(point_div(normals[s], weights[s])) > dtype(0.0):
                total += dw[s] * weights[s]
                weight += weights[s]
        # Match ProjectiveData::value(): an empty weighted sum has value zero.
        # Opposing normals can cancel in every accepted sample aggregate.
        out[0] = vec3(dtype(0.0))
        if weight > dtype(0.0):
            out[0] = vec3(total[0] / weight, total[1] / weight, weight)

    @wp.kernel
    def _average_contributions(
        normals: wp.array(dtype=vec3),
        weights: wp.array(dtype=dtype),
        dw: wp.array(dtype=vec2),
        order: wp.array(dtype=int),
        contributions: wp.array(dtype=vec3),
    ):
        i = wp.tid()
        s = order[i]
        value = vec3(dtype(0.0))
        if wp.length(point_div(normals[s], weights[s])) > dtype(0.0):
            product = dw[s] * weights[s]
            value = vec3(product[0], product[1], weights[s])
        contributions[i] = value

    @wp.kernel(module="unique")
    def _average_tiled(contributions: wp.array(dtype=vec3), out: wp.array(dtype=vec3)):
        block, lane = wp.tid()
        total = vec3(dtype(0.0))
        # Cooperative loads hide memory latency; additions retain the exact
        # serial input order and precision of ProjectiveData accumulation.
        for begin in range(0, contributions.shape[0], 128):
            tile = wp.tile_load(contributions, shape=128, offset=begin)
            for j in range(wp.min(128, contributions.shape[0] - begin)):
                value = wp.tile_extract(tile, j)
                if value[2] > dtype(0.0):
                    total += value
        if lane == 0:
            out[0] = vec3(dtype(0.0))
            if total[2] > dtype(0.0):
                out[0] = vec3(total[0] / total[2], total[1] / total[2], total[2])

    return SimpleNamespace(
        _input_order=_input_order,
        _splats=_splats,
        _negative=_negative,
        _average=_average,
        _average_contributions=_average_contributions,
        _average_tiled=_average_tiled,
    )


@dataclass
class NormalField:
    records: Records
    average: wp.array
    levels: list = None


def build_normal_field(samples, density, minimum, maximum):
    if not 0 <= minimum <= maximum <= 15:
        raise ValueError("Require 0 <= minimum <= maximum <= 15")
    device = samples.keys.device
    dtype = samples.weights.dtype
    vec4 = wp.types.vector(length=4, dtype=dtype)
    kernels = precision_kernels(dtype)
    n = len(samples.keys)
    sortkeys = wp.empty(2 * n, dtype=int, device=device)
    order = wp.empty(2 * n, dtype=int, device=device)
    wp.launch(
        kernels._input_order,
        n,
        inputs=[samples.first_indices, sortkeys, order],
        device=device,
    )
    wp.utils.radix_sort_pairs(sortkeys, order, n)
    keys = wp.empty(2 * 54 * n, dtype=wp.int64, device=device)
    vals = wp.empty(54 * n, dtype=vec4, device=device)
    wp.launch(
        kernels._splats,
        (n, 54),
        inputs=[
            samples.position_sums,
            samples.normal_sums,
            samples.weights,
            density.depths_and_weights,
            order,
            minimum,
            maximum,
            keys,
            vals,
        ],
        device=device,
    )
    records = reduce_records(keys, vals)
    wp.launch(
        kernels._negative, len(records.keys), inputs=[records.values], device=device
    )
    average = wp.empty(1, dtype=samples.positions.dtype, device=device)
    contributions = wp.empty(n, dtype=samples.positions.dtype, device=device)
    wp.launch(kernels._average_contributions, n,
              inputs=[samples.normal_sums, samples.weights,
                      density.depths_and_weights, order, contributions], device=device)
    wp.launch_tiled(kernels._average_tiled, dim=1,
                    inputs=[contributions, average], block_dim=32, device=device)
    offsets = wp.empty(maximum + 2, dtype=int, device=device)
    wp.launch(
        _depth_offsets, maximum + 2, inputs=[records.keys, offsets], device=device
    )
    start = time.perf_counter()
    boundaries = offsets.numpy().tolist()
    records.sync_seconds += time.perf_counter() - start
    empty = Records(
        wp.empty(0, dtype=wp.int64, device=device),
        wp.empty(0, dtype=vec4, device=device),
    )
    # Warp 1.19 rejects a slice whose start equals the array length, even when
    # that slice is empty. Empty high levels need explicit zero-length arrays.
    levels = [
        Records(records.keys[a:b], records.values[a:b]) if a < b else empty
        for a, b in zip(boundaries, boundaries[1:])
    ]
    return NormalField(records, average, levels)
