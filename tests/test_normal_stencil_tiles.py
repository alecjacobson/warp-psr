"""Cooperative loads retain the established double-precision stencil budget."""
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from warp_psr.normal_stencils import assemble_normal_stencil, load_kernel, parallel_load_kernel, _integrals


@pytest.mark.parametrize('vector',[wp.vec4,wp.vec4d])
def test_tile_module_preserves_floating_point_options(vector):
    options=wp.get_module_options(parallel_load_kernel(vector).module)
    assert options['fuse_fp'] is False
    assert options['fast_math'] is False


def keys(depth,xyz):
    xyz=xyz.astype(np.int64)
    return wp.array((depth<<48)|(xyz[:,0]<<32)|(xyz[:,1]<<16)|xyz[:,2],dtype=wp.int64)


@pytest.mark.cuda
@pytest.mark.parametrize('double',[False,True])
@pytest.mark.parametrize('normal_depth',[2,3,4])
@pytest.mark.parametrize('count',[1,259])
def test_cooperative_load_matches_serial_stencil(double,normal_depth,count):
    wp.init()
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(601)
    xyz=np.indices((9,9,9)).reshape(3,-1).T
    selected=np.sort(rng.choice(len(xyz),count,replace=False))
    normal_xyz=np.indices((1<<normal_depth,)*3).reshape(3,-1).T
    normal_xyz=normal_xyz[rng.random(len(normal_xyz))>.3]
    vector=wp.vec4d if double else wp.vec4
    with wp.ScopedDevice('cuda'):
        row_keys=keys(3,xyz[selected])
        records=SimpleNamespace(keys=keys(normal_depth,normal_xyz),
                                values=wp.array(rng.normal(size=(len(normal_xyz),4)),dtype=vector))
        actual=assemble_normal_stencil(row_keys,records,3,normal_depth).numpy()
        width,first=(6,-3) if normal_depth>3 else (4,-2)
        table=wp.empty((9,width),dtype=wp.vec2d)
        expected=wp.empty(count,dtype=wp.float64)
        wp.launch(_integrals,table.shape,inputs=[3,normal_depth,first,table])
        wp.launch(load_kernel(vector),count,inputs=[row_keys,records.keys,records.values,3,normal_depth,first,width,table,expected])
        np.testing.assert_allclose(actual,expected.numpy(),atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(actual,assemble_normal_stencil(row_keys,records,3,normal_depth).numpy())
