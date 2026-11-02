"""GPU input normalization and deterministic leaf aggregation.

This is the first hierarchy stage, not a finished adaptive reconstruction tree.
All bulk computation uses Warp; two tiny host status reads are measured explicitly.
"""
from dataclasses import dataclass
import math
import time
import warp as wp
import warp.utils
from .field import basis_key

wp.set_module_options({"enable_fp_fuse": False})


@wp.kernel
def _init_bounds(bounds: wp.array(dtype=float)):
    i = wp.tid()
    bounds[i] = wp.inf
    bounds[i+3] = -wp.inf


@wp.kernel
def _bounds(points: wp.array(dtype=wp.vec3), bounds: wp.array(dtype=float), status: wp.array(dtype=int)):
    p = points[wp.tid()]
    if wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]):
        for a in range(3):
            wp.atomic_min(bounds,a,p[a])
            wp.atomic_max(bounds,a+3,p[a])
    else:
        wp.atomic_add(status,0,1)


@wp.kernel
def _transform(bounds: wp.array(dtype=float), scale: float, transform: wp.array(dtype=float), status: wp.array(dtype=int)):
    width = wp.max(bounds[3]-bounds[0], wp.max(bounds[4]-bounds[1], bounds[5]-bounds[2]))*scale
    if not wp.isfinite(width) or width <= 0.0:
        status[0] += 1
    else:
        inv = float(wp.float64(1.0)/wp.float64(width))
        transform[0] = inv
        for a in range(3):
            axis = (a+1)%3  # PointExtent::Frame(0): y, z, x
            origin = (bounds[axis+3]+bounds[axis])/2.0 - width/2.0
            transform[a+1] = -origin*inv


@wp.kernel
def _classify(points: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
              transform: wp.array(dtype=float), depth: int,
              q: wp.array(dtype=wp.vec3), unit_normals: wp.array(dtype=wp.vec3),
              keys: wp.array(dtype=wp.int64), order: wp.array(dtype=int)):
    i = wp.tid()
    p = points[i]
    n = normals[i]
    inv = transform[0]
    p = wp.vec3(p[1]*inv+transform[1], p[2]*inv+transform[2], p[0]*inv+transform[3])
    n = wp.vec3(n[1],n[2],n[0])
    # Reference normal stream transform for a similarity is rotation only.
    length2 = wp.dot(n,n)
    q[i] = p
    order[i] = i
    keys[i] = wp.int64(0x7fffffffffffffff)
    unit_normals[i] = wp.vec3(0.0)
    if length2 > 0.0 and wp.isfinite(length2) and wp.min(p)>=0.0 and wp.max(p)<=1.0:
        unit_normals[i] = n/wp.sqrt(length2)
        res = 1 << depth
        # Reference ChildIndex uses > center: exact split planes belong below.
        x = wp.max(0, int(wp.ceil(p[0]*float(res)))-1)
        y = wp.max(0, int(wp.ceil(p[1]*float(res)))-1)
        z = wp.max(0, int(wp.ceil(p[2]*float(res)))-1)
        keys[i] = basis_key(depth,x,y,z)


@wp.kernel
def _mark(keys: wp.array(dtype=wp.int64), n: int, marks: wp.array(dtype=int)):
    i=wp.tid()
    if keys[i] != wp.int64(0x7fffffffffffffff):
        if i == 0:
            marks[i] = 1
        elif keys[i] != keys[i-1]:
            marks[i] = 1


@wp.kernel
def _count(prefix: wp.array(dtype=int), count: wp.array(dtype=int)):
    count[0] = prefix[prefix.shape[0]-1]


@wp.kernel
def _groups(keys: wp.array(dtype=wp.int64), order: wp.array(dtype=int), marks: wp.array(dtype=int),
            prefix: wp.array(dtype=int), positions: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
            unique: wp.array(dtype=wp.int64), sums: wp.array(dtype=wp.vec3), normal_sums: wp.array(dtype=wp.vec3),
            weights: wp.array(dtype=float)):
    i=wp.tid()
    if marks[i] != 0:
        g=prefix[i]-1
        p=wp.vec3(0.0)
        n=wp.vec3(0.0)
        weight=float(0.0)
        j=i
        # Stable radix sorting retains input order within each group.
        while j<marks.shape[0]:
            if keys[j] != keys[i]:
                break
            p += positions[order[j]]
            n += normals[order[j]]
            weight += 1.0
            j += 1
        unique[g]=keys[i]
        sums[g]=p
        normal_sums[g]=n
        weights[g]=weight


@dataclass
class SampleGroups:
    keys: wp.array
    position_sums: wp.array
    normal_sums: wp.array
    weights: wp.array
    transform: wp.array
    positions: wp.array
    normals: wp.array
    host_sync_seconds: float
    host_status_reads: int = 2


def prepare_samples(points, normals, depth=8, scale=1.1):
    """Float32 finite positions, confidence off, default reference alignment.

    Reject nonfinite positions and zero-extent input explicitly. Zero/nonfinite
    normals are omitted from grouping but their positions affect the bounding box,
    as in the reference. Coincident samples retain their multiplicity and order.
    """
    if isinstance(depth,bool) or not isinstance(depth,int) or not 0<=depth<=15:
        raise ValueError("depth must be an integer in 0..15")
    if not math.isfinite(scale) or scale<=1.0:
        raise ValueError("initial sample profile requires finite scale > 1")
    if points.dtype != wp.vec3 or normals.dtype != wp.vec3:
        raise ValueError("initial sample profile requires float32 vec3 arrays")
    if points.ndim!=1 or points.shape != normals.shape or not points.shape[0]:
        raise ValueError("points and normals must have equal nonempty 1D shapes")
    if points.device != normals.device or not points.device.is_cuda:
        raise ValueError("sample processing requires both arrays on the same CUDA device")
    device, count = points.device, points.shape[0]
    bounds=wp.empty(6,dtype=float,device=device)
    transform=wp.zeros(4,dtype=float,device=device)
    status=wp.zeros(1,dtype=int,device=device)
    wp.launch(_init_bounds,3,inputs=[bounds],device=device)
    wp.launch(_bounds,count,inputs=[points,bounds,status],device=device)
    wp.launch(_transform,1,inputs=[bounds,scale,transform,status],device=device)
    t=time.perf_counter()
    invalid=int(status.numpy()[0])
    sync=time.perf_counter()-t
    if invalid:
        raise ValueError("Nonfinite positions or degenerate/overflowing bounding cube")
    q=wp.empty_like(points)
    ns=wp.empty_like(normals)
    keys=wp.empty(2*count,dtype=wp.int64,device=device)
    order=wp.empty(2*count,dtype=int,device=device)
    wp.launch(_classify,count,inputs=[points,normals,transform,depth,q,ns,keys,order],device=device)
    wp.utils.radix_sort_pairs(keys,order,count)
    marks=wp.zeros(count,dtype=int,device=device)
    prefix=wp.empty_like(marks)
    wp.launch(_mark,count,inputs=[keys,count,marks],device=device)
    wp.utils.array_scan(marks,prefix,inclusive=True)
    wp.launch(_count,1,inputs=[prefix,status],device=device)
    t=time.perf_counter()
    groups=int(status.numpy()[0])
    sync+=time.perf_counter()-t
    if not groups:
        raise ValueError("No accepted samples with finite nonzero normals")
    unique=wp.empty(groups,dtype=wp.int64,device=device)
    sums=wp.empty(groups,dtype=wp.vec3,device=device)
    normal_sums=wp.empty_like(sums)
    weights=wp.empty(groups,dtype=float,device=device)
    wp.launch(_groups,count,inputs=[keys,order,marks,prefix,q,ns,unique,sums,normal_sums,weights],device=device)
    return SampleGroups(unique,sums,normal_sums,weights,transform,q,ns,sync)
