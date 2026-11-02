import numpy as np
import pytest
import warp as wp
from warp_psr.samples import prepare_samples


@pytest.fixture
def cuda():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return "cuda:0"


def test_multiplicity_rejections_and_split_ownership(cuda):
    # Bounding cube centered at origin; center maps exactly to split plane .5.
    points=np.array([[-1,-1,-1],[1,1,1],[0,0,0],[0,0,0],[.1,.1,.1]],dtype=np.float32)
    normals=np.array([[1,0,0],[0,1,0],[0,0,2],[0,0,-3],[0,0,0]],dtype=np.float32)
    result=prepare_samples(wp.array(points,dtype=wp.vec3,device=cuda),wp.array(normals,dtype=wp.vec3,device=cuda),depth=2)
    np.testing.assert_array_equal(result.weights.numpy(),[1,2,1])
    np.testing.assert_array_equal(result.keys.numpy(),[(2<<48), (2<<48)|(1<<32)|(1<<16)|1, (2<<48)|(3<<32)|(3<<16)|3])
    np.testing.assert_array_equal(result.normal_sums.numpy()[1],[0,0,0])
    assert result.host_status_reads==2
    assert result.host_sync_seconds>=0


def test_degenerate_rejected(cuda):
    points=wp.zeros(2,dtype=wp.vec3,device=cuda)
    normals=wp.array([[1,0,0],[1,0,0]],dtype=wp.vec3,device=cuda)
    with pytest.raises(ValueError,match="degenerate"):
        prepare_samples(points,normals)


@pytest.mark.reference
@pytest.mark.parametrize("name,depth",[("sphere",4),("torus",6),("sphere-exact",4)])
def test_aggregated_samples_against_reference(cuda,name,depth):
    from pathlib import Path
    from tools.make_fixture import read_output
    path=Path(".build/fixtures")/name
    if not (path/"stages.txt").exists():
        pytest.skip("Generate sphere fixture first")
    data=np.loadtxt(path/"input.pts").astype(np.float32)
    expected=read_output(path/"stages.txt")["sample"]
    expected=expected[np.lexsort(expected[:,:4].T[::-1])]
    result=prepare_samples(wp.array(data[:,:3],dtype=wp.vec3,device=cuda),wp.array(data[:,3:],dtype=wp.vec3,device=cuda),depth=depth)
    expected_keys=np.array([(int(d)<<48)|(int(x)<<32)|(int(y)<<16)|int(z) for d,x,y,z in expected[:,:4]],dtype=np.int64)
    np.testing.assert_array_equal(result.keys.numpy(),expected_keys)
    np.testing.assert_array_equal(result.weights.numpy(),expected[:,7])
    np.testing.assert_allclose(result.position_sums.numpy(),expected[:,4:7],atol=1e-6,rtol=1e-5)
