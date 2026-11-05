"""Dense and sparse Warp MC vertex losses reaching reconstruction inputs."""
import numpy as np
import pytest
import warp as wp
from warp_psr import reconstruct
from warp_psr.reconstruction_adjoint import reconstruct_differentiable
from warp_psr.meshing import extract_isosurface, extract_mesh
from test_sample_adjoint import inputs
from test_reconstruction_adjoint import fixed_assignments


@pytest.mark.cuda
@pytest.mark.parametrize('backend',['dense','sparse'])
def test_mesh_vertex_loss_to_inputs(backend):
    wp.init()
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    p,n=inputs('cuda:0',wp.vec3,jitter=.01)
    options=dict(depth=3,full_depth=1,samples_per_node=.05,iters=2)
    mesh_options=dict(resolution=9,bounds=((-1.2,-1.2,-1.2),(1.2,1.2,1.2)),backend=backend)
    tape=wp.Tape();result=reconstruct_differentiable(p,n,tape=tape,**options)
    with tape:
        mesh=extract_isosurface(lambda q:result.evaluate(q,centered=True)[0],device=p.device,**mesh_options)
    reference=reconstruct(p,n,**options)
    expected=extract_mesh(reference,**mesh_options)
    assert len(mesh.vertices)>0
    np.testing.assert_array_equal(mesh.indices.numpy(),expected.indices.numpy())
    np.testing.assert_array_equal(mesh.vertices.numpy(),expected.vertices.numpy())
    rng=np.random.default_rng(372)
    seed=rng.normal(size=mesh.vertices.numpy().shape)/len(mesh.vertices)
    tape.backward(grads={mesh.vertices:wp.array(seed,dtype=wp.vec3,device=p.device)})
    for array in [p,n]:
        original,gradient=array.numpy().copy(),array.grad.numpy().copy()
        direction=rng.normal(size=original.shape).astype(original.dtype)
        losses=[];eps=1e-4
        for sign in [1,-1]:
            array.assign(original+sign*eps*direction)
            candidate=reconstruct(p,n,**options);fixed_assignments(reference,candidate)
            current=extract_mesh(candidate,**mesh_options)
            np.testing.assert_array_equal(current.indices.numpy(),mesh.indices.numpy())
            assert current.vertices.shape==mesh.vertices.shape
            losses.append(np.sum(current.vertices.numpy().astype(float)*seed))
        array.assign(original)
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(np.sum(gradient.astype(float)*direction),(losses[0]-losses[1])/(2*eps),rtol=3e-2,atol=2e-3)
