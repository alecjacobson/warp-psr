"""Nonoverlapping adaptive integration cells resolving every active spline knot."""
import warp as wp
import warp.fem as fem
from .field import basis_key, find_key
from .keyed import coordinates, unique_keys

@wp.kernel
def _basis_support(keys: wp.array(dtype=wp.int64), d: int, out: wp.array(dtype=wp.int64)):
    i,k=wp.tid()
    c=coordinates(keys[i])-wp.vec3i(k&1,(k>>1)&1,(k>>2)&1)
    out[i*8+k]=wp.int64(0x7fffffffffffffff)
    if wp.min(c)>=0 and wp.max(c)<1<<d:
        out[i*8+k]=basis_key(d,c[0],c[1],c[2])

@wp.kernel
def _normal_support(keys: wp.array(dtype=wp.int64), d: int, offset: int, out: wp.array(dtype=wp.int64)):
    i,k=wp.tid()
    idx=offset+i*27+k
    out[idx]=wp.int64(0x7fffffffffffffff)
    if int(keys[i]>>wp.int64(48))==d:
        c=coordinates(keys[i])+wp.vec3i(k//9-1,(k//3)%3-1,k%3-1)
        if wp.min(c)>=0 and wp.max(c)<1<<d:
            out[idx]=basis_key(d,c[0],c[1],c[2])

@wp.kernel
def _parent(keys: wp.array(dtype=wp.int64), d: int, offset: int, out: wp.array(dtype=wp.int64)):
    i=wp.tid()
    c=coordinates(keys[i])
    out[offset+i]=basis_key(d,c[0]//2,c[1]//2,c[2]//2)

@wp.kernel
def _children(parents: wp.array(dtype=wp.int64), refined: wp.array(dtype=wp.int64), d: int, out: wp.array(dtype=wp.int64)):
    i,k=wp.tid()
    c=coordinates(parents[i])*2+wp.vec3i(k&1,(k>>1)&1,(k>>2)&1)
    key=basis_key(d,c[0],c[1],c[2])
    out[i*8+k]=key
    if find_key(refined,key)>=0:
        out[i*8+k]=wp.int64(0x7fffffffffffffff)

@wp.kernel
def _pack(keys: wp.array(dtype=wp.int64), shift: int, offset: int, xyz: wp.array(dtype=wp.vec3i), levels: wp.array(dtype=wp.uint8)):
    i=wp.tid()
    c=coordinates(keys[i])
    xyz[offset+i]=c*(1<<shift)
    levels[offset+i]=wp.uint8(shift)

@wp.kernel
def _reorder(xyz: wp.array(dtype=wp.vec3i), levels: wp.array(dtype=wp.uint8), volume: wp.uint64, out: wp.array(dtype=wp.uint8)):
    i=wp.tid()
    c=xyz[i]
    index=wp.volume_lookup_index(volume,c[0],c[1],c[2])
    out[index]=levels[i]


def integration_grid(tree,normals,depth,dtype=wp.float32):
    """Resolve basis knots through depth, and optionally the normal-field knots."""
    device=tree.basis[0].keys.device
    normal_count=len(normals.records.keys) if normals is not None else 0
    required=[None]*(depth+1)
    refined=[None]*(depth+1)
    empty=wp.empty(0,dtype=wp.int64,device=device)
    for d in range(depth,-1,-1):
        fine=required[d+1].keys if d<depth else empty
        keys=wp.empty(2*len(fine),dtype=wp.int64,device=device)
        if len(fine):
            wp.launch(_parent,len(fine),inputs=[fine,d,0,keys],device=device)
        refined[d]=unique_keys(keys,len(fine))
        basis=tree.basis[d].keys
        count=8*len(basis)+27*normal_count+len(refined[d].keys)
        keys=wp.empty(2*count,dtype=wp.int64,device=device)
        if len(basis):
            wp.launch(_basis_support,(len(basis),8),inputs=[basis,d,keys],device=device)
        if normal_count:
            wp.launch(_normal_support,(normal_count,27),inputs=[normals.records.keys,d,8*len(basis),keys],device=device)
        wp.copy(keys,refined[d].keys,dest_offset=count-len(refined[d].keys),count=len(refined[d].keys))
        required[d]=unique_keys(keys,count)
    leaves=[]
    # The degree-1 complete root basis always resolves the entire reconstruction cube.
    for d in range(depth+1):
        if d==0:
            keys=wp.zeros(2,dtype=wp.int64,device=device)
            if len(refined[0].keys):
                leaves.append(empty)
            else:
                leaves.append(keys[:1])
        else:
            parents=refined[d-1].keys
            n=8*len(parents)
            keys=wp.empty(2*n,dtype=wp.int64,device=device)
            if n:
                wp.launch(_children,(len(parents),8),inputs=[parents,refined[d].keys,d,keys],device=device)
            leaves.append(unique_keys(keys,n).keys)
    count=sum(len(l) for l in leaves)
    xyz=wp.empty(count,dtype=wp.vec3i,device=device)
    levels=wp.empty(count,dtype=wp.uint8,device=device)
    offset=0
    for d,keys in enumerate(leaves):
        if len(keys):
            wp.launch(_pack,len(keys),inputs=[keys,depth-d,offset,xyz,levels],device=device)
        offset+=len(keys)
    h=1.0/(1<<depth)
    volume=wp.Volume.allocate_by_voxels(xyz,voxel_size=h,translation=(h/2,h/2,h/2),device=device)
    reordered=wp.empty(count,dtype=wp.uint8,device=device)
    wp.launch(_reorder,count,inputs=[xyz,levels,volume.id,reordered],device=device)
    coarsest=min(d for d,keys in enumerate(leaves) if len(keys))
    return fem.AdaptiveNanogrid(volume,cell_level=reordered,level_count=depth-coarsest+1,temporary_store=None,scalar_type=dtype)
