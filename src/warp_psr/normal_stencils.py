"""Tensor-product normal loads, checked against the public FEM quadrature path."""
from functools import lru_cache
import warp as wp
from .field import basis_key, find_key
from .keyed import coordinates
from .ordered_normal_loads import normal_integral

wp.set_module_options({"fuse_fp": False})


@wp.func
def normal_center(index: int, test_depth: int, normal_depth: int):
    center = index
    if test_depth > normal_depth:
        center = index >> 1
    elif test_depth < normal_depth:
        center = index << 1
    return center


@wp.kernel
def _integrals(test_depth: int, normal_depth: int, first: int,
               table: wp.array2d(dtype=wp.vec2d)):
    i, k = wp.tid()
    j = normal_center(i, test_depth, normal_depth) + first + k
    table[i, k] = normal_integral(test_depth, i, normal_depth, j)


@lru_cache(None)
def load_kernel(vector):
    @wp.kernel
    def load(keys: wp.array(dtype=wp.int64), normal_keys: wp.array(dtype=wp.int64),
             normals: wp.array(dtype=vector), test_depth: int, normal_depth: int,
             first: int, width: int, table: wp.array2d(dtype=wp.vec2d),
             out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        c = coordinates(keys[i])
        center = wp.vec3i(normal_center(c[0], test_depth, normal_depth),
                          normal_center(c[1], test_depth, normal_depth),
                          normal_center(c[2], test_depth, normal_depth))
        value = wp.float64(0.0)
        for k in range(width * width * width):
            x, y, z = k // (width * width), (k // width) % width, k % width
            n = center + wp.vec3i(x + first, y + first, z + first)
            if wp.min(n) >= 0 and wp.max(n) < 1 << normal_depth:
                j = find_key(normal_keys, basis_key(normal_depth, n[0], n[1], n[2]))
                if j >= 0:
                    ix, iy, iz = table[c[0], x], table[c[1], y], table[c[2], z]
                    normal = normals[j]
                    value += wp.float64(normal[0]) * ix[1] * iy[0] * iz[0]
                    value += wp.float64(normal[1]) * ix[0] * iy[1] * iz[0]
                    value += wp.float64(normal[2]) * ix[0] * iy[0] * iz[1]
        out[i] = value
    return load


@wp.kernel
def _row_bounds(keys: wp.array(dtype=wp.int64), depth: int, n: int,
                offsets: wp.array(dtype=int)):
    row = wp.tid()
    key = basis_key(depth, row // n, row % n, 0)
    lo, hi = int(0), keys.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    offsets[row] = lo


def normal_row_bounds(keys, depth):
    n = 1 << depth
    offsets = wp.empty(n * n + 1, dtype=int, device=keys.device)
    wp.launch(_row_bounds, len(offsets), inputs=[keys, depth, n, offsets], device=keys.device)
    return offsets


@lru_cache(None)
def range_load_kernel(vector):
    @wp.kernel
    def load_ranges(keys: wp.array(dtype=wp.int64), normal_keys: wp.array(dtype=wp.int64),
                    normals: wp.array(dtype=vector), row_bounds: wp.array(dtype=int),
                    test_depth: int, normal_depth: int, first: int, width: int,
                    table: wp.array2d(dtype=wp.vec2d),
                    out: wp.array(dtype=wp.float64)):
        i = wp.tid()
        c = coordinates(keys[i])
        center = wp.vec3i(normal_center(c[0], test_depth, normal_depth),
                          normal_center(c[1], test_depth, normal_depth),
                          normal_center(c[2], test_depth, normal_depth))
        value = wp.float64(0.0)
        n = 1 << normal_depth
        z0 = wp.max(0, center[2] + first)
        z1 = wp.min(n - 1, center[2] + first + width - 1)
        for x in range(width):
            nx = center[0] + first + x
            if nx >= 0 and nx < n:
                for y in range(width):
                    ny = center[1] + first + y
                    if ny >= 0 and ny < n and z0 <= z1:
                        begin = basis_key(normal_depth, nx, ny, z0)
                        end = basis_key(normal_depth, nx, ny, z1)
                        row = nx * n + ny
                        lo = int(row_bounds[row])
                        hi = int(row_bounds[row + 1])
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if normal_keys[mid] < begin:
                                lo = mid + 1
                            else:
                                hi = mid
                        # Sorted z keys are contiguous for each (x, y). One
                        # search replaces width searches while retaining the
                        # original x/y/z and component accumulation order.
                        j = lo
                        while j < normal_keys.shape[0]:
                            key = normal_keys[j]
                            if key > end:
                                break
                            z = int(key & wp.int64(65535)) - center[2] - first
                            ix, iy, iz = table[c[0], x], table[c[1], y], table[c[2], z]
                            normal = normals[j]
                            value += wp.float64(normal[0]) * ix[1] * iy[0] * iz[0]
                            value += wp.float64(normal[1]) * ix[0] * iy[1] * iz[0]
                            value += wp.float64(normal[2]) * ix[0] * iy[0] * iz[1]
                            j += 1
        out[i] = value
    return load_ranges


def assemble_normal_stencil(keys, records, test_depth, normal_depth):
    if abs(test_depth - normal_depth) > 1:
        raise ValueError("Normal stencils require adjacent levels")
    device = keys.device
    out = wp.zeros(len(keys), dtype=wp.float64, device=device)
    if not len(records.keys):
        return out
    width, first = (6, -3) if test_depth < normal_depth else (4, -2)
    table = wp.empty(((1 << test_depth) + 1, width), dtype=wp.vec2d, device=device)
    wp.launch(_integrals, table.shape, inputs=[test_depth, normal_depth, first, table], device=device)
    # Bound auxiliary storage to about 4 MiB and avoid building a dense x/y index
    # for tiny or very sparse inputs. The original lookup remains the fallback.
    if normal_depth <= 10 and len(keys) >= 4096 and (1 << (2 * normal_depth)) <= 2 * len(records.keys):
        offsets = normal_row_bounds(records.keys, normal_depth)
        wp.launch(range_load_kernel(records.values.dtype), len(keys),
                  inputs=[keys, records.keys, records.values, offsets, test_depth,
                          normal_depth, first, width, table, out], device=device)
    else:
        wp.launch(load_kernel(records.values.dtype), len(keys),
                  inputs=[keys, records.keys, records.values, test_depth, normal_depth,
                          first, width, table, out], device=device)
    return out
