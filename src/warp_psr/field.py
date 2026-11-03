"""Sparse per-level degree-1 Neumann field evaluation entirely in Warp.

Coefficients are level corrections, not an independent global basis. The current
oracle adapter is test-only; a GPU reconstruction builder must supply these arrays.
"""
from dataclasses import dataclass
from functools import lru_cache
import warp as wp

wp.set_module_options({"fuse_fp":False})


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
                x = wp.clamp(int(wp.ceil(t[0]))-1, 0, n-1)
                y = wp.clamp(int(wp.ceil(t[1]))-1, 0, n-1)
                z = wp.clamp(int(wp.ceil(t[2]))-1, 0, n-1)
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
    cumulative: wp.array = None

    def __post_init__(self):
        if self.coefficients.dtype not in (wp.float32, wp.float64) or self.keys.dtype != wp.int64:
            raise ValueError("Field requires float32/float64 coefficients and int64 keys")
        if not 0 <= self.depth <= 15:
            raise ValueError("Keys support depths 0..15, including upper boundary nodes")
        if self.keys.shape != self.coefficients.shape:
            raise ValueError("One key per coefficient is required")

    def evaluate(self, queries):
        """Return values, model-space gradients and validity mask; outside = NaN.

        Inputs must share the field device and precision. Direct level summation
        uses left derivatives at knots. Reconstruction fields carry cumulative
        coefficients and reproduce the pinned reference's neighbor/polynomial
        convention, including its boundary nudges (issue #20).
        """
        dtype = self.coefficients.dtype
        vec = wp.types.vector(length=3, dtype=dtype)
        if not wp.types.types_equal(queries.dtype, vec) or queries.device != self.coefficients.device:
            raise ValueError("Query precision/device must match coefficients")
        if self.keys.device != queries.device:
            raise ValueError("Keys and coefficients must share the query device")
        mat = wp.types.matrix(shape=(3, 3), dtype=dtype)
        values = wp.empty(queries.shape[0], dtype=dtype, device=queries.device)
        gradients = wp.empty(queries.shape[0], dtype=vec, device=queries.device)
        valid = wp.empty(queries.shape[0], dtype=wp.int32, device=queries.device)
        kernel=evaluation_kernel(dtype)
        inputs=[self.keys,self.coefficients]
        if self.cumulative is not None:
            kernel=reference_evaluation_kernel(dtype)
            inputs.append(self.cumulative)
        inputs.extend([queries, mat(*self.linear), vec(*self.translation), self.depth])
        wp.launch(kernel, dim=queries.shape[0],
                  inputs=inputs,
                  outputs=[values, gradients, valid], device=queries.device)
        return values, gradients, valid


@lru_cache(None)
def cumulative_kernel(dtype):
    @wp.kernel
    def prolong(keys: wp.array(dtype=wp.int64), coefficients: wp.array(dtype=dtype), cumulative: wp.array(dtype=dtype), depth: int):
        i=wp.tid()
        key=keys[i]
        if int(key>>wp.int64(48))==depth:
            x,y,z=int((key>>wp.int64(32))&wp.int64(65535)),int((key>>wp.int64(16))&wp.int64(65535)),int(key&wp.int64(65535))
            value=dtype(0)
            for corner in range(8):
                a,b,c=corner//4,(corner//2)%2,corner%2
                h=(dtype(1)-wp.abs(dtype(x)*dtype(.5)-dtype(x//2+a)))*(dtype(1)-wp.abs(dtype(y)*dtype(.5)-dtype(y//2+b)))*(dtype(1)-wp.abs(dtype(z)*dtype(.5)-dtype(z//2+c)))
                if h>dtype(0):
                    idx=find_key(keys,basis_key(depth-1,x//2+a,y//2+b,z//2+c))
                    if idx>=0:value+=cumulative[idx]*h
            cumulative[i]=value+coefficients[i]
    return prolong


def cumulative_coefficients(keys,coefficients,depth):
    result=wp.clone(coefficients)
    for d in range(1,depth+1):
        wp.launch(cumulative_kernel(coefficients.dtype),len(keys),inputs=[keys,coefficients,result,d],device=keys.device)
    return result


@lru_cache(None)
def reference_evaluation_kernel(dtype):
    vec=wp.types.vector(length=3,dtype=dtype)
    vec4=wp.types.vector(length=4,dtype=dtype)
    mat=wp.types.matrix(shape=(3,3),dtype=dtype)

    @wp.func
    def add_level(keys: wp.array(dtype=wp.int64), coefficients: wp.array(dtype=dtype), p: vec, d: int):
        n=1<<d
        t=p*dtype(n)
        # Reference neighbors come from left-owned tree cells, while its
        # PointEvaluatorState uses floor, i.e. right-owned polynomial pieces.
        cell=wp.vec3i(wp.clamp(int(wp.ceil(t[0]))-1,0,n-1),wp.clamp(int(wp.ceil(t[1]))-1,0,n-1),wp.clamp(int(wp.ceil(t[2]))-1,0,n-1))
        base=wp.vec3i(int(wp.floor(t[0])),int(wp.floor(t[1])),int(wp.floor(t[2])))
        f=t-vec(dtype(base[0]),dtype(base[1]),dtype(base[2]))
        result=vec4(dtype(0))
        for corner in range(8):
            node=cell+wp.vec3i(corner//4,(corner//2)%2,corner%2)
            if wp.min(node-base)>=0:
                idx=find_key(keys,basis_key(d,node[0],node[1],node[2]))
                if idx>=0:
                    h=vec(dtype(1))-f
                    sign=vec(dtype(-1))
                    for a in range(3):
                        if node[a]>base[a]:
                            h[a]=f[a]
                            sign[a]=dtype(1)
                    c=coefficients[idx]
                    result+=vec4(c*h[0]*h[1]*h[2],c*dtype(n)*sign[0]*h[1]*h[2],c*dtype(n)*h[0]*sign[1]*h[2],c*dtype(n)*h[0]*h[1]*sign[2])
        return result

    @wp.kernel
    def evaluate(keys: wp.array(dtype=wp.int64), coefficients: wp.array(dtype=dtype), cumulative: wp.array(dtype=dtype),
                 queries: wp.array(dtype=vec), linear: mat, translation: vec, depth: int,
                 values: wp.array(dtype=dtype), gradients: wp.array(dtype=vec),valid: wp.array(dtype=int)):
        i=wp.tid()
        p=linear*queries[i]+translation
        ok=wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]) and wp.min(p)>=dtype(0) and wp.max(p)<=dtype(1)
        valid[i]=int(ok)
        if not ok:
            values[i]=dtype(wp.nan);gradients[i]=vec(dtype(wp.nan))
        else:
            leaf=int(0)
            for d in range(depth,-1,-1):
                n=1<<d
                cell=wp.vec3i(wp.clamp(int(wp.ceil(p[0]*dtype(n)))-1,0,n-1),wp.clamp(int(wp.ceil(p[1]*dtype(n)))-1,0,n-1),wp.clamp(int(wp.ceil(p[2]*dtype(n)))-1,0,n-1))
                if find_key(keys,basis_key(d,cell[0],cell[1],cell[2]))>=0:
                    leaf=d
                    break
            for a in range(3):
                if p[a]==dtype(0):p[a]=dtype(1e-6)
                elif p[a]==dtype(1):p[a]=dtype(1.0-1e-6)
            result=add_level(keys,coefficients,p,leaf)
            if leaf>0:result+=add_level(keys,cumulative,p,leaf-1)
            if leaf<depth:result+=add_level(keys,coefficients,p,leaf+1)
            values[i]=result[0]
            gradients[i]=wp.transpose(linear)*vec(result[1],result[2],result[3])
    return evaluate
