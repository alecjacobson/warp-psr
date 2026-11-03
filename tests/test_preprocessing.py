from pathlib import Path
import numpy as np
import pytest
import warp as wp
from warp_psr.samples import prepare_samples
from warp_psr.density import build_density
from warp_psr.normals import build_normal_field
from warp_psr.tree import build_tree
from tools.make_fixture import read_output


def test_effective_floating_point_options():
    import warp_psr.samples, warp_psr.density, warp_psr.normals, warp_psr.keyed
    for module in [warp_psr.samples,warp_psr.density,warp_psr.normals,warp_psr.keyed]:
        assert wp.get_module_options(module.__name__)["fuse_fp"] is False
        assert wp.get_module_options(module.__name__)["fast_math"] is False


@pytest.mark.cuda
@pytest.mark.reference
@pytest.mark.parametrize("name,depth,full",[("sphere",4,2),("torus",6,3),("sparse",5,1),("sphere-psr-double",4,2),("torus-double",6,3),("sparse-double",5,1)])
def test_gpu_density_normal_tree(name,depth,full):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    path=Path('.build/fixtures')/name
    if not (path/'preprocess.txt').exists():
        pytest.skip("Regenerate reference preprocessing exports")
    double=name.endswith('double')
    atol,rtol=(1e-10,1e-9) if double else (1e-6,1e-5)
    vector=wp.vec3d if double else wp.vec3
    data=np.loadtxt(path/'input.pts').astype(np.float64 if double else np.float32)
    samples=prepare_samples(wp.array(data[:,:3],dtype=vector,device='cuda'),
                            wp.array(data[:,3:],dtype=vector,device='cuda'),depth=depth)
    assert samples.weights.dtype==(wp.float64 if double else wp.float32)
    density=build_density(samples,depth)
    normal=build_normal_field(samples,density,full,depth)
    tree=build_tree(normal,depth,full)
    pre=read_output(path/'preprocess.txt')
    stage=read_output(path/'stages.txt')
    order=np.argsort(samples.first_indices.numpy())
    np.testing.assert_allclose(samples.position_sums.numpy()[order],pre['sample'][:,5:8],atol=atol,rtol=rtol)
    np.testing.assert_allclose(samples.normal_sums.numpy()[order],pre['sample'][:,9:12],atol=atol,rtol=rtol)
    for d,level in enumerate(density.levels):
        expected=pre['density'][pre['density'][:,0]==d]
        expected=expected[np.lexsort(expected[:,:4].T[::-1])]
        keys=np.array([(int(d)<<48)|(int(x)<<32)|(int(y)<<16)|int(z) for d,x,y,z in expected[:,:4]],dtype=np.int64)
        np.testing.assert_array_equal(level.keys.numpy(),keys)
        np.testing.assert_allclose(level.values.numpy()[:,0],expected[:,4],atol=atol,rtol=rtol)
    expected=stage['normal'];expected=expected[np.lexsort(expected[:,:4].T[::-1])]
    keys=np.array([(int(d)<<48)|(int(x)<<32)|(int(y)<<16)|int(z) for d,x,y,z in expected[:,:4]],dtype=np.int64)
    np.testing.assert_array_equal(normal.records.keys.numpy(),keys)
    np.testing.assert_allclose(normal.records.values.numpy()[:,:3],expected[:,4:],atol=atol,rtol=rtol)
    np.testing.assert_allclose(normal.average.numpy()[0,:2],stage['density_scale'][0],atol=atol,rtol=rtol)
    splats=read_output(path/'splats.txt')['splat']
    np.testing.assert_allclose(density.depths_and_weights.numpy()[order],splats[:,3:5],atol=atol,rtol=rtol)
    nodes=read_output(path/'oracle_instrumented.txt')['node']
    for d,level in enumerate(tree.levels):
        expected=nodes[(nodes[:,0]==d)&(nodes[:,5]==0)]
        keys=np.sort(np.array([((int(x)+524288)<<40)|((int(y)+524288)<<20)|(int(z)+524288) for x,y,z in expected[:,1:4]],dtype=np.int64))
        np.testing.assert_array_equal(level.keys.numpy(),keys)
        expected=nodes[(nodes[:,0]==d)&((nodes[:,4].astype(int)&2)!=0)]
        keys=np.sort(np.array([(d<<48)|(int(x)<<32)|(int(y)<<16)|int(z) for x,y,z in expected[:,1:4]],dtype=np.int64))
        np.testing.assert_array_equal(tree.basis[d].keys.numpy(),keys)
