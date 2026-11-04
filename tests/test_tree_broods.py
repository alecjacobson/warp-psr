"""Brood compaction must preserve signed support keys and their exact order."""
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from warp_psr.keyed import Records
from warp_psr.tree import build_tree, expand_broods


def packed(points):
    return np.array(sorted(((int(x)+524288)<<40)|((int(y)+524288)<<20)|(int(z)+524288)
                           for x,y,z in points),dtype=np.int64)


def device():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return 'cuda'


@pytest.mark.cuda
@pytest.mark.parametrize('kind',['empty','single','row','plane','sparse','extremes'])
def test_sorted_brood_expansion(kind):
    rng=np.random.default_rng(82)
    points={tuple(p) for p in rng.integers(-12,13,size=(1000,3))}
    if kind=='empty':points=set()
    if kind=='single':points={(-1,-1,-1)}
    if kind=='row':points={(3,-1,z) for z in [-8,-3,0,4,19]}
    if kind=='plane':points={(x,y,4) for x in [-4,0,3] for y in [-6,0,3,9]}
    if kind=='extremes':points=set(product([-65536,-1,0,65535],repeat=3))
    keys=wp.array(packed(points),dtype=wp.int64,device=device())
    expected={tuple(2*p[a]+c[a] for a in range(3)) for p in points for c in product(range(2),repeat=3)}
    actual=expand_broods(Records(keys,None)).keys.numpy()
    np.testing.assert_array_equal(actual,packed(expected))


def expanded_reference(normal_levels,depth,full,base):
    """Independent child-set construction following the original tree rules."""
    levels=[None]*(depth+1)
    requests=None
    children=list(product(range(2),repeat=3))
    for d in range(depth,-1,-1):
        parents={tuple(v//2 for v in c) for c in levels[d+1]} if d<depth else set()
        if d>base or requests is None:requests=parents
        else:requests={tuple(v//2 for v in c) for c in requests}
        current=set()
        for c,value in normal_levels[d]:
            if np.dot(value,value)>0:
                current.update(tuple(2*(c[a]//2)+k[a] for a in range(3)) for k in children)
        if d<=full:
            end=2 if d==0 else (1<<d)+2
            current.update(product(range(-2,end),repeat=3))
        for source,radius in [(requests,1),(parents,0)]:
            for c in source:
                for delta in product(range(-radius,radius+1),repeat=3):
                    for k in children:
                        child=tuple(2*(c[a]//2+delta[a])+k[a] for a in range(3))
                        if min(child)>=-2*(1<<d) and max(child)<2*(1<<d):current.add(child)
        levels[d]=current
    return levels


@pytest.mark.cuda
@pytest.mark.parametrize('dtype',[wp.vec4,wp.vec4d])
@pytest.mark.parametrize('depth,full,base',[(0,0,0),(1,0,0),(4,2,1),(7,2,0),(15,1,1)])
def test_tree_matches_expanded_child_sets(dtype,depth,full,base):
    target=device()
    rng=np.random.default_rng(334)
    normal_levels=[]
    records=[]
    for d in range(depth+1):
        coords=rng.integers(0,1<<d,size=(3,3))
        coords[0]=0
        coords[1]=(1<<d)-1
        values=rng.normal(size=(3,4))
        values[2]=0  # A cancelled normal does not seed a brood.
        normal_levels.append(list(zip(coords,values)))
        keys=np.array([(d<<48)|(int(x)<<32)|(int(y)<<16)|int(z) for x,y,z in coords],dtype=np.int64)
        records.append(Records(wp.array(keys,dtype=wp.int64,device=target),wp.array(values,dtype=dtype,device=target)))
    actual=build_tree(SimpleNamespace(records=records[0],levels=records),depth,full,base)
    expected=expanded_reference(normal_levels,depth,full,base)
    for d,points in enumerate(expected):
        np.testing.assert_array_equal(actual.levels[d].keys.numpy(),packed(points))
        active=np.array(sorted((d<<48)|(x<<32)|(y<<16)|z for x,y,z in points
                               if min(x,y,z)>=0 and max(x,y,z)<=(1<<d)),dtype=np.int64)
        np.testing.assert_array_equal(actual.basis[d].keys.numpy(),active)
