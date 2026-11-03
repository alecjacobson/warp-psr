"""Evaluate the reference quadratic Dirichlet vector field for FEM quadrature."""
import warp as wp
import warp.fem as fem
from .forms import value
from .field import basis_key, find_key
from .density import quadratic

@wp.func
def normal_value(p: wp.vec3, keys: wp.array(dtype=wp.int64), values: wp.array(dtype=wp.vec4), depth: int):
    value=wp.vec3(0.0)
    for d in range(depth+1):
        n=1<<d
        q=p*float(n)
        c=wp.vec3i(wp.min(int(wp.floor(q[0])),n-1),wp.min(int(wp.floor(q[1])),n-1),wp.min(int(wp.floor(q[2])),n-1))
        t=q-wp.vec3(c)
        bx,by,bz=quadratic(wp.float64(t[0])),quadratic(wp.float64(t[1])),quadratic(wp.float64(t[2]))
        for k in range(27):
            x,y,z=k//9,(k//3)%3,k%3
            source=c+wp.vec3i(x-1,y-1,z-1)
            sign=float(1.0)
            for axis in range(3):
                if source[axis]<0:
                    source[axis]=-1-source[axis]
                    sign=-sign
                elif source[axis]>=n:
                    source[axis]=2*n-1-source[axis]
                    sign=-sign
            idx=find_key(keys,basis_key(d,source[0],source[1],source[2]))
            if idx>=0:
                v=values[idx]
                value+=wp.vec3(v[0],v[1],v[2])*float(bx[x]*by[y]*bz[z])*sign
    return value

@fem.integrand
def normal_load(s: fem.Sample, v: fem.Field, normal: fem.Field):
    return wp.dot(fem.grad(v,s),normal(s))


@wp.func
def _cached_normal(index: int, values: wp.array(dtype=wp.vec3)):
    return values[index]


def cache_normal_quadrature(geometry,normals,depth):
    domain=fem.Cells(geometry)
    quadrature=fem.RegularQuadrature(domain,order=3)
    normal=fem.ImplicitField(domain,normal_value,values={"keys":normals.records.keys,"values":normals.records.values,"depth":depth},degree=2)
    data=wp.empty(quadrature.total_point_count(),dtype=wp.vec3,device=normals.records.keys.device)
    fem.interpolate(value,dest=data,at=quadrature,fields={"u":normal})
    return quadrature,fem.ImplicitField(domain,_cached_normal,values={"values":data},degree=2)


def assemble_normal_load(space,normals,depth,cache=None):
    domain=fem.Cells(space.geometry)
    if cache is None:
        quadrature=fem.RegularQuadrature(domain,order=3)
        normal=fem.ImplicitField(domain,normal_value,values={"keys":normals.records.keys,"values":normals.records.values,"depth":depth},degree=2)
    else:
        quadrature,normal=cache
    v=fem.make_test(space,domain=domain)
    return fem.integrate(normal_load,quadrature=quadrature,fields={"v":v,"normal":normal},output_dtype=wp.float64,device=normals.records.keys.device)
