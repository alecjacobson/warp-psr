"""Pinned-reference normal-field splatting on GPU (including documented issue #12)."""
from dataclasses import dataclass
import warp as wp
import warp.utils
from .samples import point_div
from .field import basis_key
from .keyed import reduce_records, Records
from .density import quadratic

wp.set_module_options({"fuse_fp": False})

@wp.kernel
def _input_order(first: wp.array(dtype=int), keys: wp.array(dtype=int), ids: wp.array(dtype=int)):
    i=wp.tid()
    keys[i]=first[i]
    ids[i]=i

@wp.kernel
def _splats(sums: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3), weights: wp.array(dtype=float),
            dw: wp.array(dtype=wp.vec2), order: wp.array(dtype=int), minimum: int, maximum: int,
            keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.vec4)):
    i,k=wp.tid()
    src=order[i]
    dst=i*54+k
    keys[dst]=wp.int64(0x7fffffffffffffff)
    values[dst]=wp.vec4(0.0)
    w=weights[src]
    normal=point_div(normals[src],w)
    length=wp.length(normal)
    if length>0.0:
        normal=point_div(normal,length)*w
        p=point_div(sums[src],w)
        depth=wp.clamp(dw[src][0],float(minimum),float(maximum))
        top=int(wp.ceil(depth))
        dx=1.0-(float(top)-depth)
        if top<=minimum:
            top=minimum
            dx=1.0
        if top>maximum:
            top=maximum
            dx=1.0
        fraction=dx
        if k>=27:
            fraction=1.0-dx
        if k<27 or wp.abs(wp.float64(1.0)-wp.float64(dx))>wp.float64(1e-6):
            # The pinned reference's Splat lambda ignores its node argument and
            # uses captured temp for BOTH calls. Do not 'fix' this in compatibility.
            n=1<<top
            h=1.0/float(n)
            c=wp.vec3i(wp.max(0,int(wp.ceil(p[0]*float(n)))-1),wp.max(0,int(wp.ceil(p[1]*float(n)))-1),wp.max(0,int(wp.ceil(p[2]*float(n)))-1))
            t=(p-wp.vec3(float(c[0])*h,float(c[1])*h,float(c[2])*h))/h
            bx,by,bz=quadratic(wp.float64(t[0])),quadratic(wp.float64(t[1])),quadratic(wp.float64(t[2]))
            q=k%27
            x,y,z=q//9,(q//3)%3,q%3
            target=c+wp.vec3i(x-1,y-1,z-1)
            if wp.min(target)>=0 and wp.max(target)<n:
                v=normal*dw[src][1]/(h*h*h)*fraction
                v*=float(bx[x]*by[y]*bz[z])
                keys[dst]=basis_key(top,target[0],target[1],target[2])
                values[dst]=wp.vec4(v[0],v[1],v[2],0.0)

@wp.kernel
def _negative(values: wp.array(dtype=wp.vec4)):
    i=wp.tid()
    values[i]=-values[i]

@wp.kernel
def _average(normals: wp.array(dtype=wp.vec3), weights: wp.array(dtype=float),
             dw: wp.array(dtype=wp.vec2), order: wp.array(dtype=int), out: wp.array(dtype=wp.vec3)):
    total=wp.vec2(0.0)
    weight=float(0.0)
    for i in range(weights.shape[0]):
        s=order[i]
        if wp.length(point_div(normals[s],weights[s]))>0.0:
            total+=dw[s]*weights[s]
            weight+=weights[s]
    out[0]=wp.vec3(total[0]/weight,total[1]/weight,weight)

@dataclass
class NormalField:
    records: Records
    average: wp.array


def build_normal_field(samples,density,minimum,maximum):
    if not 0<=minimum<=maximum<=15:
        raise ValueError("Require 0 <= minimum <= maximum <= 15")
    device=samples.keys.device
    n=len(samples.keys)
    sortkeys=wp.empty(2*n,dtype=int,device=device)
    order=wp.empty(2*n,dtype=int,device=device)
    wp.launch(_input_order,n,inputs=[samples.first_indices,sortkeys,order],device=device)
    wp.utils.radix_sort_pairs(sortkeys,order,n)
    keys=wp.empty(2*54*n,dtype=wp.int64,device=device)
    vals=wp.empty(54*n,dtype=wp.vec4,device=device)
    wp.launch(_splats,(n,54),inputs=[samples.position_sums,samples.normal_sums,samples.weights,density.depths_and_weights,order,minimum,maximum,keys,vals],device=device)
    records=reduce_records(keys,vals)
    wp.launch(_negative,len(records.keys),inputs=[records.values],device=device)
    average=wp.empty(1,dtype=wp.vec3,device=device)
    wp.launch(_average,1,inputs=[samples.normal_sums,samples.weights,density.depths_and_weights,order,average],device=device)
    return NormalField(records,average)
