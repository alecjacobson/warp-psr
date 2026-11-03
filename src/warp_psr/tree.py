"""Sparse GPU tree completion for the pinned degree-1/degree-2-normal profile.

Signed coordinates belong to per-level arrays. Padding is never packed into an
unsigned Morton code. The level keys below are for logical nodes, not field keys.
"""
from dataclasses import dataclass
from typing import Any
import warp as wp
from .field import basis_key
from .keyed import coordinates, unique_keys, Records

wp.set_module_options({"fuse_fp": False})

@wp.func
def tree_key(c: wp.vec3i):
    return (wp.int64(c[0]+524288)<<wp.int64(40)) | (wp.int64(c[1]+524288)<<wp.int64(20)) | wp.int64(c[2]+524288)

@wp.func
def tree_coordinates(key: wp.int64):
    return wp.vec3i(int((key>>wp.int64(40))&wp.int64(1048575))-524288,
                    int((key>>wp.int64(20))&wp.int64(1048575))-524288,
                    int(key&wp.int64(1048575))-524288)

@wp.func
def parent(c: wp.vec3i):
    return wp.vec3i(int(wp.floor(float(c[0])/2.0)),int(wp.floor(float(c[1])/2.0)),int(wp.floor(float(c[2])/2.0)))

@wp.kernel
def _seeds(normal_keys: wp.array(dtype=wp.int64), normals: wp.array(dtype=Any), d: int,
           keys: wp.array(dtype=wp.int64)):
    i,k=wp.tid()
    idx=i*8+k
    keys[idx]=wp.int64(0x7fffffffffffffff)
    nd=int(normal_keys[i]>>wp.int64(48))
    if nd>=d and wp.float64(wp.length_sq(normals[i]))>wp.float64(0.0):
        c=coordinates(normal_keys[i])
        c=wp.vec3i(c[0]>>(nd-d),c[1]>>(nd-d),c[2]>>(nd-d))
        c=parent(c)*2+wp.vec3i(k&1,(k>>1)&1,(k>>2)&1)
        keys[idx]=tree_key(c)

@wp.kernel
def _coarse(d: int, count: int, lo: int, offset: int, keys: wp.array(dtype=wp.int64)):
    i=wp.tid()
    c=wp.vec3i(i//(count*count)+lo,(i//count)%count+lo,i%count+lo)
    keys[offset+i]=tree_key(c)

@wp.kernel
def _parent_keys(fine: wp.array(dtype=wp.int64), out: wp.array(dtype=wp.int64)):
    i=wp.tid()
    out[i]=tree_key(parent(tree_coordinates(fine[i])))

@wp.kernel
def _support(fine: wp.array(dtype=wp.int64), radius: int, offset: int, depth: int, keys: wp.array(dtype=wp.int64)):
    i,j,k=wp.tid()
    # Completing broods over a radius-two node window produces exactly the
    # 3^3 adjacent broods around parent(center), regardless of center parity.
    # Emit each of their eight children once (216 candidates instead of 1000).
    brood_radius=radius//2
    width=brood_radius*2+1
    c=parent(tree_coordinates(fine[i]))
    c+=wp.vec3i(j//(width*width)-brood_radius,(j//width)%width-brood_radius,j%width-brood_radius)
    c=c*2+wp.vec3i(k&1,(k>>1)&1,(k>>2)&1)
    idx=offset+(i*width*width*width+j)*8+k
    keys[idx]=wp.int64(0x7fffffffffffffff)
    n=1<<depth
    if wp.min(c)>=-2*n and wp.max(c)<2*n:
        keys[idx]=tree_key(c)

@wp.kernel
def _field_keys(keys: wp.array(dtype=wp.int64), d: int, out: wp.array(dtype=wp.int64)):
    i=wp.tid()
    c=tree_coordinates(keys[i])
    out[i]=wp.int64(0x7fffffffffffffff)
    if wp.min(c)>=0 and wp.max(c)<=1<<d:
        out[i]=basis_key(d,c[0],c[1],c[2])

@dataclass
class LogicalTree:
    levels: list
    basis: list
    host_sync_seconds: float


def build_tree(normals,depth,full_depth,base_depth=None):
    if base_depth is None:
        base_depth=full_depth
    if not 0<=base_depth<=full_depth<=depth<=15:
        raise ValueError("Require 0 <= base_depth <= full_depth <= depth <= 15")
    device=normals.records.keys.device
    levels=[None]*(depth+1)
    basis=[None]*(depth+1)
    count=len(normals.records.keys)
    sync=0.0
    requests=None
    for d in range(depth,-1,-1):
        # Current correctness path emits neighborhoods from unique parent nodes,
        # so temporary storage is O(active nodes * fixed stencil), never dense D.
        radius=2
        fine=levels[d+1].keys if d<depth else wp.empty(0,dtype=wp.int64,device=device)
        # Deduplicate fine parents before expanding the support stencil.
        pk=wp.empty(2*len(fine),dtype=wp.int64,device=device)
        if len(fine):
            wp.launch(_parent_keys,len(fine),inputs=[fine,pk],device=device)
        parents=unique_keys(pk,len(fine))
        sync+=parents.sync_seconds

        if d>base_depth:
            requests=parents
        elif requests is not None:
            rk=wp.empty(2*len(requests.keys),dtype=wp.int64,device=device)
            wp.launch(_parent_keys,len(requests.keys),inputs=[requests.keys,rk],device=device)
            requests=unique_keys(rk,len(requests.keys))
            sync+=requests.sync_seconds
        else:
            requests=parents
        coarse_count=(4 if d==0 else (1<<d)+4) if d<=full_depth else 0
        nc=coarse_count**3
        ns=len(requests.keys)*27*8
        np=8*len(parents.keys)
        total=count*8+nc+ns+np
        keys=wp.empty(2*total,dtype=wp.int64,device=device)
        wp.launch(_seeds,(count,8),inputs=[normals.records.keys,normals.records.values,d,keys],device=device)
        if nc:
            wp.launch(_coarse,nc,inputs=[d,coarse_count,-2,count*8,keys],device=device)
        if ns:
            wp.launch(_support,(len(requests.keys),27,8),inputs=[requests.keys,2,count*8+nc,d,keys],device=device)
        if np:
            wp.launch(_support,(len(parents.keys),1,8),inputs=[parents.keys,0,count*8+nc+ns,d,keys],device=device)
        levels[d]=unique_keys(keys,total)
        sync+=levels[d].sync_seconds
        n=len(levels[d].keys)
        fk=wp.empty(2*n,dtype=wp.int64,device=device)
        wp.launch(_field_keys,n,inputs=[levels[d].keys,d,fk],device=device)
        basis[d]=unique_keys(fk,n)
        sync+=basis[d].sync_seconds
    return LogicalTree(levels,basis,sync)
