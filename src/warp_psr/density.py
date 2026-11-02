"""Reference degree-2 sample density and splat-depth selection on sparse GPU levels."""
from dataclasses import dataclass
import warp as wp
from .samples import point_div
from .field import basis_key, find_key
from .keyed import Records, coordinates, morton_order, reduce_records

wp.set_module_options({"fuse_fp": False})

@wp.func
def quadratic(x: wp.float64):
    one=wp.float64(1.0)
    half=wp.float64(0.5)
    return wp.vec3d((one-x)*(one-x)*half, ((one-x)*(x+one)+x*(wp.float64(2.0)-x))*half,x*x*half)

@wp.kernel
def _scale(out: wp.array(dtype=float)):
    # Reference averages 1000 float32 scale values. The two tangential axes
    # reproduce constants, so the hyperplane response is sum of squares in x.
    total=float(0.0)
    dx=float(0.1)
    for x in range(10):
        p=dx/2.0+dx*float(x)
        b=quadratic(wp.float64(p))
        scale=float(wp.float64(1.0)/wp.dot(b,b))
        for y in range(10):
            for z in range(10):
                total+=scale
    out[0]=total/1000.0

@wp.kernel
def _leaf_values(p: wp.array(dtype=wp.vec3), w: wp.array(dtype=float), out: wp.array(dtype=wp.vec4)):
    i=wp.tid()
    out[i]=wp.vec4(p[i][0],p[i][1],p[i][2],w[i])

@wp.kernel
def _parents(keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.vec4), order: wp.array(dtype=int),
             depth: int, out_keys: wp.array(dtype=wp.int64), out_values: wp.array(dtype=wp.vec4)):
    i=wp.tid()
    j=order[i]
    c=coordinates(keys[j])
    out_keys[i]=basis_key(depth-1,c[0]//2,c[1]//2,c[2]//2)
    out_values[i]=values[j]

@wp.kernel
def _splat(keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.vec4), order: wp.array(dtype=int),
           depth: int, scale: wp.array(dtype=float), out_keys: wp.array(dtype=wp.int64), out_values: wp.array(dtype=wp.vec4)):
    i,k=wp.tid()
    j=order[i]
    c=coordinates(keys[j])
    v=values[j]
    n=1<<depth
    p=point_div(wp.vec3(v[0],v[1],v[2]),v[3])
    h=1.0/float(n)
    t=(p-wp.vec3(float(c[0])*h,float(c[1])*h,float(c[2])*h))/h
    bx,by,bz=quadratic(wp.float64(t[0])),quadratic(wp.float64(t[1])),quadratic(wp.float64(t[2]))
    x,y,z=k//9,(k//3)%3,k%3
    target=c+wp.vec3i(x-1,y-1,z-1)
    idx=i*27+k
    out_keys[idx]=wp.int64(0x7fffffffffffffff)
    out_values[idx]=wp.vec4(0.0)
    if wp.min(target)>=0 and wp.max(target)<n:
        weight=v[3]*scale[0]
        val=float(wp.float64(weight)*bx[x]*by[y]*bz[z])
        out_keys[idx]=basis_key(depth,target[0],target[1],target[2])
        out_values[idx]=wp.vec4(val,0.0,0.0,0.0)

@wp.func
def density_value(keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.vec4), p: wp.vec3, depth: int):
    n=1<<depth
    c=wp.vec3i(wp.max(0,int(wp.ceil(p[0]*float(n)))-1),wp.max(0,int(wp.ceil(p[1]*float(n)))-1),wp.max(0,int(wp.ceil(p[2]*float(n)))-1))
    h=1.0/float(n)
    t=(p-wp.vec3(float(c[0])*h,float(c[1])*h,float(c[2])*h))/h
    bx,by,bz=quadratic(wp.float64(t[0])),quadratic(wp.float64(t[1])),quadratic(wp.float64(t[2]))
    value=float(0.0)
    for k in range(27):
        x,y,z=k//9,(k//3)%3,k%3
        target=c+wp.vec3i(x-1,y-1,z-1)
        if wp.min(target)>=0 and wp.max(target)<n:
            idx=find_key(keys,basis_key(depth,target[0],target[1],target[2]))
            if idx>=0:
                value+=float(bx[x]*by[y]*bz[z]*wp.float64(values[idx][0]))
    return value

@wp.kernel
def _evaluate(keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.vec4),
              sums: wp.array(dtype=wp.vec3), weights: wp.array(dtype=float), depth: int, out: wp.array2d(dtype=float)):
    i=wp.tid()
    out[i,depth]=density_value(keys,values,point_div(sums[i],weights[i]),depth)

@wp.kernel
def _depths(densities: wp.array2d(dtype=float), kernel_depth: int, target: float, out: wp.array(dtype=wp.vec2)):
    i=wp.tid()
    d=kernel_depth
    density=densities[i,d]
    raw=float(0.0)
    if density>=target:
        raw=float(wp.float64(d)+wp.log(wp.float64(density/target))/(wp.log(wp.float64(2.0))*wp.float64(2.0)))
    else:
        fine=density
        while density<target and d>0:
            fine=density
            d-=1
            density=densities[i,d]
        raw=float(wp.float64(d)+wp.log(wp.float64(density/target))/wp.log(wp.float64(density/fine)))
    h=1.0/float(1<<d)
    weight=h*h/density
    out[i]=wp.vec2(raw,weight)

@dataclass
class Density:
    levels: list
    aggregates: list
    depths_and_weights: wp.array
    scale: wp.array
    sync_seconds: float
    host_status_reads: int


def build_density(samples,depth,kernel_depth=None,samples_per_node=1.5):
    if kernel_depth is None:
        kernel_depth=max(depth-2,0)
    if not 0<=kernel_depth<=depth<=15 or samples_per_node<=0:
        raise ValueError("Require 0 <= kernel_depth <= depth <= 15 and positive samples_per_node")
    device=samples.keys.device
    count=len(samples.keys)
    values=wp.empty(count,dtype=wp.vec4,device=device)
    wp.launch(_leaf_values,count,inputs=[samples.position_sums,samples.weights,values],device=device)
    aggregates=[None]*(depth+1)
    aggregates[depth]=Records(samples.keys,values)
    sync=0.0
    for d in range(depth,0,-1):
        src=aggregates[d]
        order=morton_order(src.keys,d)
        keys=wp.empty(2*len(src.keys),dtype=wp.int64,device=device)
        vals=wp.empty_like(src.values)
        wp.launch(_parents,len(src.keys),inputs=[src.keys,src.values,order,d,keys,vals],device=device)
        aggregates[d-1]=reduce_records(keys,vals)
        sync+=aggregates[d-1].sync_seconds
    scale=wp.empty(1,dtype=float,device=device)
    wp.launch(_scale,1,inputs=[scale],device=device)
    levels=[]
    densities=wp.empty((count,kernel_depth+1),dtype=float,device=device)
    for d in range(kernel_depth+1):
        src=aggregates[d]
        order=morton_order(src.keys,d)
        keys=wp.empty(2*27*len(src.keys),dtype=wp.int64,device=device)
        vals=wp.empty(27*len(src.keys),dtype=wp.vec4,device=device)
        wp.launch(_splat,(len(src.keys),27),inputs=[src.keys,src.values,order,d,scale,keys,vals],device=device)
        level=reduce_records(keys,vals)
        levels.append(level)
        sync+=level.sync_seconds
        wp.launch(_evaluate,count,inputs=[level.keys,level.values,samples.position_sums,samples.weights,d,densities],device=device)
    dw=wp.empty(count,dtype=wp.vec2,device=device)
    wp.launch(_depths,count,inputs=[densities,kernel_depth,samples_per_node,dw],device=device)
    return Density(levels,aggregates,dw,scale,sync,depth+kernel_depth+1)
