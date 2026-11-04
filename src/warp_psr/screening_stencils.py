"""Add clustered point forms directly to the verified 27-neighbor CSR stencil."""
from functools import lru_cache
from dataclasses import dataclass
import warp as wp
from .field import basis_key, find_key
from .keyed import coordinates

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def screening_kernel(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)

    @wp.kernel
    def bin_points(positions: wp.array(dtype=vec3), depth: int,
                   keys: wp.array(dtype=wp.int64), ids: wp.array(dtype=int)):
        i = wp.tid()
        n = 1 << depth
        p = positions[i] * dtype(n)
        c = wp.vec3i(wp.clamp(int(wp.floor(p[0])), 0, n - 1),
                     wp.clamp(int(wp.floor(p[1])), 0, n - 1),
                     wp.clamp(int(wp.floor(p[2])), 0, n - 1))
        keys[i] = basis_key(depth, c[0], c[1], c[2])
        ids[i] = i

    @wp.func
    def hat(p: vec3, c: wp.vec3i, depth: int):
        t = p * dtype(1 << depth) - vec3(c)
        h = wp.max(vec3(dtype(1.0)) - wp.abs(t), vec3(dtype(0.0)))
        return h[0] * h[1] * h[2]

    @wp.kernel
    def add(row_keys: wp.array(dtype=wp.int64), col_keys: wp.array(dtype=wp.int64),
            point_keys: wp.array(dtype=wp.int64), point_ids: wp.array(dtype=int), positions: wp.array(dtype=vec3),
            weights: wp.array(dtype=dtype), depth: int, trial_depth: int,
            offsets: wp.array(dtype=int), columns: wp.array(dtype=int),
            values: wp.array(dtype=wp.float64), load: wp.array(dtype=wp.float64)):
        i, local = wp.tid()
        c = coordinates(row_keys[i])
        entry = offsets[i] + local
        active = entry < offsets[i + 1]
        q = wp.vec3i(0)
        if active:
            q = coordinates(col_keys[columns[entry]])
        accumulated = wp.float64(0.0)
        rhs = wp.float64(0.0)
        # Bin actual aggregate positions: rounding can move a centroid across
        # its original tree-cell boundary, so a bin may contain several points.
        # A hat sees only the eight cells touching its vertex.
        # Neighboring lanes accumulate neighboring entries, retaining the same
        # sample order for every entry without a 27-double thread-local vector.
        # Lane zero also produces the RHS when a trial row is empty.
        if active or local == 0:
            for k in range(8):
                cell = c - wp.vec3i(k // 4, (k // 2) % 2, k % 2)
                if wp.min(cell) >= 0 and wp.max(cell) < 1 << depth:
                    key = basis_key(depth, cell[0], cell[1], cell[2])
                    first = find_key(point_keys, key)
                    if first >= 0:
                        j = first
                        while j < point_keys.shape[0]:
                            if point_keys[j] != key:
                                break
                            index = point_ids[j]
                            p = vec3(positions[index])
                            weight = dtype(weights[index])
                            if weight != dtype(0.0):
                                phi = hat(p, c, depth)
                                if local == 0:
                                    rhs += wp.float64(phi * weight)
                                if active:
                                    trial = hat(p, q, trial_depth)
                                    accumulated += wp.float64(phi * trial * weight)
                            j += 1
        if local == 0:
            load[i] = rhs
        if active:
            values[entry] += accumulated
    return bin_points, add


@dataclass
class PointBins:
    keys: wp.array
    ids: wp.array


def bin_constraints(points, depth):
    n = len(points.positions)
    device = points.positions.device
    keys = wp.empty(2 * n, dtype=wp.int64, device=device)
    ids = wp.empty(2 * n, dtype=int, device=device)
    wp.launch(screening_kernel(points.weights.dtype)[0], n,
              inputs=[points.positions, depth, keys, ids], device=device)
    wp.utils.radix_sort_pairs(keys, ids, n)
    return PointBins(keys[:n], ids[:n])


def add_screening_stencil(matrix, row_keys, col_keys, points, depth, trial_depth=None, bins=None):
    if trial_depth is None:
        trial_depth = depth
    device = row_keys.device
    if bins is None:
        bins = bin_constraints(points, depth)
    load = wp.empty(len(row_keys), dtype=wp.float64, device=device)
    wp.launch(screening_kernel(points.weights.dtype)[1], (len(row_keys), 27),
              inputs=[row_keys, col_keys, bins.keys, bins.ids, points.positions, points.weights,
                      depth, trial_depth, matrix.offsets, matrix.columns, matrix.values, load], device=device)
    return load
