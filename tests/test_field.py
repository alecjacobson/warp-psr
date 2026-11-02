import numpy as np
import pytest
import warp as wp
from warp_psr.field import LevelField


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
def test_affine_gradient_and_domain(device, dtype):
    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    # f(cube)=x+2y+3z. Nontrivial similarity maps model to permuted cube axes.
    keys=np.array([(x<<32)|(y<<16)|z for x in range(2) for y in range(2) for z in range(2)],dtype=np.int64)
    coefficients=[x+2*y+3*z for x in range(2) for y in range(2) for z in range(2)]
    linear=np.array([[0,.25,0],[0,0,.25],[.25,0,0]])
    translation=np.array([-.2,.3,.1])
    cube=np.array([[.13,.27,.83],[0,0,0],[1,1,1],[1.01,.5,.5],[np.nan,0,0]])
    model=(cube-translation)@np.linalg.inv(linear).T
    vec=wp.types.vector(length=3,dtype=dtype)
    field=LevelField(wp.array(keys,dtype=wp.int64,device=device),wp.array(coefficients,dtype=dtype,device=device),
                     tuple(linear.reshape(-1)),tuple(translation),0,.5)
    value,gradient,valid=field.evaluate(wp.array(model,dtype=vec,device=device))
    np.testing.assert_array_equal(valid.numpy(),[1,1,1,0,0])
    atol=1e-6 if dtype==wp.float32 else 1e-14
    np.testing.assert_allclose(value.numpy()[:3],cube[:3]@np.array([1,2,3]),atol=atol)
    np.testing.assert_allclose(gradient.numpy()[:3],np.tile(np.array([1,2,3])@linear,(3,1)),atol=atol)
    assert np.isnan(value.numpy()[3:]).all()
    assert np.isnan(gradient.numpy()[3:]).all()
