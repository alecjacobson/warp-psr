"""Reference exact and level-clustered point constraints, evaluated by PicQuadrature."""
from dataclasses import dataclass
import warp as wp
from .field import find_key
from .keyed import coordinates
from .tree import tree_key
from .samples import point_div

wp.set_module_options({"fuse_fp":False})

@wp.kernel
def _points(keys: wp.array(dtype=wp.int64), data: wp.array(dtype=wp.vec4), active: wp.array(dtype=wp.int64),
            average: wp.array(dtype=wp.vec3), point_weight: float, depth: int, source_depth: int,
            positions: wp.array(dtype=wp.vec3), weights: wp.array(dtype=float)):
    i=wp.tid()
    v=data[i]
    positions[i]=point_div(wp.vec3(v[0],v[1],v[2]),v[3])
    c=coordinates(keys[i])
    shift=source_depth-depth
    c=wp.vec3i(c[0]>>shift,c[1]>>shift,c[2]>>shift)
    w=v[3]*float(1<<source_depth)
    w*=point_weight*average[0][1]
    # Root padding relocates the original root's interpolation record outside
    # the valid depth-zero space cell. The pinned oracle has no points at d=0.
    if depth==0 or find_key(active,tree_key(c))<0:
        w=0.0
    weights[i]=w

@dataclass
class PointConstraints:
    positions: wp.array
    weights: wp.array


def point_constraints(density,normals,tree,depth,point_weight,exact=False):
    source_depth=len(density.aggregates)-1 if exact else depth
    src=density.aggregates[source_depth]
    device=src.keys.device
    p=wp.empty(len(src.keys),dtype=wp.vec3,device=device)
    w=wp.empty(len(src.keys),dtype=wp.float32,device=device)
    wp.launch(_points,len(src.keys),inputs=[src.keys,src.values,tree.levels[depth].keys,normals.average,point_weight,depth,source_depth,p,w],device=device)
    if exact:
        # PoissonRecon 262b0f5 initializes exact ranges before SPACE_FLAG exists;
        # _isValidSpaceNode rejects every sample. Preserve its observable run
        # behavior, verified against the official --exact CLI (issue #16).
        w.zero_()
    return PointConstraints(p,w)
