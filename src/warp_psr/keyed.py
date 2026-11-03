"""Deterministic sparse records and Morton traversal, using Warp sort/scan."""
from dataclasses import dataclass
import time
import warp as wp
import warp.utils
from .field import basis_key, find_key

wp.set_module_options({"fuse_fp": False})

@wp.func
def coordinates(key: wp.int64):
    return wp.vec3i(int((key >> wp.int64(32)) & wp.int64(65535)),
                    int((key >> wp.int64(16)) & wp.int64(65535)), int(key & wp.int64(65535)))

@wp.func
def morton(c: wp.vec3i, depth: int):
    m=wp.int64(0)
    for bit in range(depth):
        for a in range(3):
            m |= wp.int64((c[a] >> bit) & 1) << wp.int64(3*bit+a)
    return m

@wp.kernel
def _order(keys: wp.array(dtype=wp.int64), depth: int, codes: wp.array(dtype=wp.int64), ids: wp.array(dtype=int)):
    i=wp.tid()
    codes[i]=morton(coordinates(keys[i]),depth)
    ids[i]=i

@wp.kernel
def _ids(ids: wp.array(dtype=int)):
    i=wp.tid()
    ids[i]=i

@wp.kernel
def _mark(keys: wp.array(dtype=wp.int64), marks: wp.array(dtype=int)):
    i=wp.tid()
    if keys[i] != wp.int64(0x7fffffffffffffff):
        if i==0:
            marks[i]=1
        elif keys[i]!=keys[i-1]:
            marks[i]=1

@wp.kernel
def _count(prefix: wp.array(dtype=int), out: wp.array(dtype=int)):
    out[0]=prefix[prefix.shape[0]-1]

@wp.kernel
def _reduce(keys: wp.array(dtype=wp.int64), ids: wp.array(dtype=int), values: wp.array(dtype=wp.vec4),
            marks: wp.array(dtype=int), prefix: wp.array(dtype=int), out_keys: wp.array(dtype=wp.int64), out_values: wp.array(dtype=wp.vec4)):
    i=wp.tid()
    if marks[i]!=0:
        total=wp.vec4(0.0)
        j=i
        while j<marks.shape[0]:
            if keys[j]!=keys[i]:
                break
            total+=values[ids[j]]
            j+=1
        g=prefix[i]-1
        out_keys[g]=keys[i]
        out_values[g]=total

@dataclass
class Records:
    keys: wp.array
    values: wp.array
    sync_seconds: float=0.0


def morton_order(keys,depth):
    n=len(keys)
    codes=wp.empty(2*n,dtype=wp.int64,device=keys.device)
    ids=wp.empty(2*n,dtype=int,device=keys.device)
    wp.launch(_order,n,inputs=[keys,depth,codes,ids],device=keys.device)
    wp.utils.radix_sort_pairs(codes,ids,n)
    return ids


def reduce_records(keys,values):
    """Keys have 2*N capacity; input order defines floating-point sum order."""
    n=len(values)
    device=keys.device
    if not n:
        return Records(wp.empty(0,dtype=wp.int64,device=device),wp.empty(0,dtype=wp.vec4,device=device))
    ids=wp.empty(2*n,dtype=int,device=device)
    wp.launch(_ids,n,inputs=[ids],device=device)
    wp.utils.radix_sort_pairs(keys,ids,n)
    marks=wp.zeros(n,dtype=int,device=device)
    prefix=wp.empty_like(marks)
    count=wp.empty(1,dtype=int,device=device)
    wp.launch(_mark,n,inputs=[keys,marks],device=device)
    wp.utils.array_scan(marks,prefix,inclusive=True)
    wp.launch(_count,1,inputs=[prefix,count],device=device)
    start=time.perf_counter()
    size=int(count.numpy()[0])
    sync=time.perf_counter()-start
    out_keys=wp.empty(size,dtype=wp.int64,device=device)
    out_values=wp.empty(size,dtype=wp.vec4,device=device)
    wp.launch(_reduce,n,inputs=[keys,ids,values,marks,prefix,out_keys,out_values],device=device)
    return Records(out_keys,out_values,sync)


@wp.kernel
def _compact_unique(keys: wp.array(dtype=wp.int64), marks: wp.array(dtype=int),
                    prefix: wp.array(dtype=int), out: wp.array(dtype=wp.int64)):
    i=wp.tid()
    if marks[i]!=0:
        out[prefix[i]-1]=keys[i]


def unique_keys(keys,count):
    """Set union without allocating/reducing a dummy vec4 for every candidate."""
    device=keys.device
    if not count:
        return Records(wp.empty(0,dtype=wp.int64,device=device),wp.empty(0,dtype=wp.vec4,device=device))
    scratch=wp.zeros(2*count,dtype=int,device=device)
    wp.utils.radix_sort_pairs(keys,scratch,count)
    marks=wp.zeros(count,dtype=int,device=device)
    prefix=wp.empty_like(marks)
    scalar=wp.empty(1,dtype=int,device=device)
    wp.launch(_mark,count,inputs=[keys,marks],device=device)
    wp.utils.array_scan(marks,prefix,inclusive=True)
    wp.launch(_count,1,inputs=[prefix,scalar],device=device)
    start=time.perf_counter()
    size=int(scalar.numpy()[0])
    sync=time.perf_counter()-start
    out=wp.empty(size,dtype=wp.int64,device=device)
    wp.launch(_compact_unique,count,inputs=[keys,marks,prefix,out],device=device)
    return Records(out,wp.zeros(size,dtype=wp.vec4,device=device),sync)
