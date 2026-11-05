"""Device transform, cumulative and isovalue VJPs with primal parity."""
import numpy as np
import pytest
import warp as wp
from warp_psr.field import LevelField, cumulative_coefficients
from warp_psr.field_adjoint import (DifferentiableField, cube_field, record_cumulative_vjp, record_isovalue_vjp)
from warp_psr.reconstruction import _compute_isovalue
from warp_psr.transforms import precision_kernels
from test_assembly_adjoint import keys
from test_sample_adjoint import device


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
def test_device_field_and_cumulative_vjp(device, dtype):
    rng = np.random.default_rng(387)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    matrix = wp.mat44d if dtype == wp.float64 else wp.mat44
    k = wp.array(np.concatenate([keys(d,device).numpy() for d in range(3)]),dtype=wp.int64,device=device)
    coefficients = wp.array(rng.normal(size=len(k)),dtype=dtype,device=device,requires_grad=True)
    normalization = wp.array([.3,.2,.25,.3,0],dtype=dtype,device=device,requires_grad=True)
    cumulative = cumulative_coefficients(k,coefficients,2);cumulative.requires_grad=True
    transforms = wp.empty(2,dtype=matrix,device=device,requires_grad=True)
    candidates = rng.uniform(.1,.9,(100,3))
    local = candidates[:,[1,2,0]]*.3 + np.array([.2,.25,.3])
    keep = (np.abs(local*4-np.round(local*4)) > .04).all(axis=1)
    q = wp.array(candidates[keep][:13],dtype=vector,device=device,requires_grad=True)
    tape = wp.Tape()
    record_cumulative_vjp(k,coefficients,cumulative,2,tape=tape)
    with tape:
        wp.launch(precision_kernels(dtype).evaluation_transform,1,inputs=[normalization,transforms],device=device)
        field = DifferentiableField(cube_field(k,coefficients,cumulative,2),transforms,wp.zeros(1,dtype=dtype,device=device),tape)
        value,gradient,valid=field.evaluate(q)
    t=transforms.numpy()[1]
    original_cells=np.floor((q.numpy()@t[:3,:3].T+t[:3,3])*4)
    reference=LevelField(k,coefficients,tuple(t[:3,:3].ravel()),tuple(t[:3,3]),2,0,cumulative)
    rv,rg,rvalid=reference.evaluate(q)
    np.testing.assert_array_equal(value.numpy(),rv.numpy())
    np.testing.assert_array_equal(gradient.numpy(),rg.numpy())
    np.testing.assert_array_equal(valid.numpy(),rvalid.numpy())
    sv,sg=rng.normal(size=len(q)),rng.normal(size=(len(q),3))*.1
    tape.backward(grads={value:wp.array(sv,dtype=dtype,device=device),gradient:wp.array(sg,dtype=vector,device=device)})
    for array in [coefficients,normalization,q]:
        original,g=array.numpy().copy(),array.grad.numpy().copy()
        direction=rng.normal(size=original.shape).astype(original.dtype)
        eps=1e-4 if dtype==wp.float32 else 1e-6
        losses=[]
        for sign in [1,-1]:
            array.assign(original+sign*eps*direction)
            c=cumulative_coefficients(k,coefficients,2)
            wp.launch(precision_kernels(dtype).evaluation_transform,1,inputs=[normalization,transforms],device=device)
            t=transforms.numpy()[1]
            np.testing.assert_array_equal(np.floor((q.numpy()@t[:3,:3].T+t[:3,3])*4),original_cells)
            f=LevelField(k,coefficients,tuple(t[:3,:3].ravel()),tuple(t[:3,3]),2,0,c)
            v,gr,_=f.evaluate(q)
            losses.append(v.numpy().astype(float)@sv+np.sum(gr.numpy().astype(float)*sg))
        array.assign(original)
        np.testing.assert_allclose(np.sum(g.astype(float)*direction),(losses[0]-losses[1])/(2*eps),
            rtol=5e-3 if dtype==wp.float32 else 1e-6,atol=1e-3 if dtype==wp.float32 else 1e-7,err_msg="coefficients" if array is coefficients else "normalization" if array is normalization else "queries")


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('count',[19,1403])
def test_weighted_isovalue_vjp(device,dtype,count):
    rng=np.random.default_rng(344)
    values=wp.array(rng.normal(size=count),dtype=dtype,device=device,requires_grad=True)
    weights=wp.array(rng.uniform(.5,3,count),dtype=dtype,device=device,requires_grad=True)
    iso=_compute_isovalue(values,weights);iso.requires_grad=True
    tape=wp.Tape();record_isovalue_vjp(values,weights,iso,tape=tape)
    tape.backward(grads={iso:wp.ones_like(iso,requires_grad=False)})
    for array in [values,weights]:
        original,g=array.numpy().copy(),array.grad.numpy().copy()
        direction=rng.normal(size=original.shape).astype(original.dtype)
        eps=1e-3 if dtype==wp.float32 else 1e-6
        losses=[]
        for sign in [1,-1]:
            array.assign(original+sign*eps*direction)
            losses.append(float(_compute_isovalue(values,weights).numpy()[0]))
        array.assign(original)
        np.testing.assert_allclose(np.sum(g.astype(float)*direction),(losses[0]-losses[1])/(2*eps),
            rtol=1e-3 if dtype==wp.float32 else 1e-7,atol=1e-5 if dtype==wp.float32 else 1e-9)


def test_legacy_cumulative_field_tracking_fails_explicitly():
    field=LevelField(wp.array([0],dtype=wp.int64,device='cpu'),
        wp.ones(1,dtype=wp.float32,device='cpu',requires_grad=True),
        (1.,0.,0.,0.,1.,0.,0.,0.,1.),(0.,0.,0.),0,0.,wp.ones(1,dtype=wp.float32,device='cpu',requires_grad=True))
    with pytest.raises(ValueError,match='cumulative.*reconstruct_differentiable'):
        field.evaluate(wp.array([[.2,.3,.4]],dtype=wp.vec3,device='cpu'),requires_grad=True)
