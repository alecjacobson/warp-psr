"""Sparse per-level degree-1 Neumann field evaluation entirely in Warp.

Coefficients are level corrections, not an independent global basis. The current
oracle adapter is test-only; a GPU reconstruction builder must supply these arrays.
"""
from dataclasses import dataclass
from functools import lru_cache
import warp as wp


@wp.func
def basis_key(d: int, x: int, y: int, z: int):
    return ((wp.int64(d) << wp.int64(48)) | (wp.int64(x) << wp.int64(32))
            | (wp.int64(y) << wp.int64(16)) | wp.int64(z))


@wp.func
def find_key(keys: wp.array(dtype=wp.int64), key: wp.int64):
    lo = int(0)
    hi = keys.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < keys.shape[0]:
        if keys[lo] == key:
            return lo
    return -1


@lru_cache(None)
def evaluation_kernel(dtype):
    vec = wp.types.vector(length=3, dtype=dtype)

    mat = wp.types.matrix(shape=(3, 3), dtype=dtype)

    @wp.kernel
    def evaluate(keys: wp.array(dtype=wp.int64), coefficients: wp.array(dtype=dtype),
                 queries: wp.array(dtype=vec), linear: mat, translation: vec, depth: int,
                 values: wp.array(dtype=dtype), gradients: wp.array(dtype=vec),
                 valid: wp.array(dtype=wp.int32)):
        i = wp.tid()
        p = linear * queries[i] + translation
        value = dtype(0)
        grad = vec(dtype(0))
        ok = wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])
        ok = ok and wp.min(p) >= dtype(0) and wp.max(p) <= dtype(1)
        valid[i] = wp.int32(ok)
        if ok:
            for d in range(depth + 1):
                n = 1 << d
                t = p * dtype(n)
                x = wp.min(int(wp.floor(t[0])), n-1)
                y = wp.min(int(wp.floor(t[1])), n-1)
                z = wp.min(int(wp.floor(t[2])), n-1)
                f = t - vec(dtype(x), dtype(y), dtype(z))
                for corner in range(8):
                    cx = corner // 4
                    cy = (corner // 2) % 2
                    cz = corner % 2
                    idx = find_key(keys, basis_key(d, x+cx, y+cy, z+cz))
                    if idx >= 0:
                        h = vec(dtype(1)-f[0], dtype(1)-f[1], dtype(1)-f[2])
                        sign = vec(dtype(-1))
                        if cx != 0:
                            h[0] = f[0]
                            sign[0] = dtype(1)
                        if cy != 0:
                            h[1] = f[1]
                            sign[1] = dtype(1)
                        if cz != 0:
                            h[2] = f[2]
                            sign[2] = dtype(1)
                        c = coefficients[idx]
                        value += c*h[0]*h[1]*h[2]
                        grad += c*dtype(n)*vec(sign[0]*h[1]*h[2], h[0]*sign[1]*h[2], h[0]*h[1]*sign[2])
            values[i] = value
            gradients[i] = wp.transpose(linear) * grad
        else:
            values[i] = dtype(wp.nan)
            gradients[i] = vec(dtype(wp.nan))
    return evaluate


@dataclass
class LevelField:
    keys: wp.array
    coefficients: wp.array
    linear: tuple
    translation: tuple
    depth: int
    iso_value: float

    def __post_init__(self):
        if self.coefficients.dtype not in (wp.float32, wp.float64) or self.keys.dtype != wp.int64:
            raise ValueError("Field requires float32/float64 coefficients and int64 keys")
        if not 0 <= self.depth <= 15:
            raise ValueError("Keys support depths 0..15, including upper boundary nodes")
        if self.keys.shape != self.coefficients.shape:
            raise ValueError("One key per coefficient is required")

    def evaluate(self, queries):
        """Return values, model-space gradients and validity mask; outside = NaN.

        Inputs must share the field device and precision. At knots this evaluator
        chooses the cell on the right (left at the upper boundary).
        """
        dtype = self.coefficients.dtype
        vec = wp.types.vector(length=3, dtype=dtype)
        if queries.dtype != vec or queries.device != self.coefficients.device:
            raise ValueError("Query precision/device must match coefficients")
        if self.keys.device != queries.device:
            raise ValueError("Keys and coefficients must share the query device")
        mat = wp.types.matrix(shape=(3, 3), dtype=dtype)
        values = wp.empty(queries.shape[0], dtype=dtype, device=queries.device)
        gradients = wp.empty(queries.shape[0], dtype=vec, device=queries.device)
        valid = wp.empty(queries.shape[0], dtype=wp.int32, device=queries.device)
        wp.launch(evaluation_kernel(dtype), dim=queries.shape[0],
                  inputs=[self.keys, self.coefficients, queries, mat(*self.linear), vec(*self.translation), self.depth],
                  outputs=[values, gradients, valid], device=queries.device)
        return values, gradients, valid
