"""Public density/normal forward parity and input-point VJPs of fixed splats."""
import numpy as np
import pytest
import warp as wp
from warp_psr.density import build_density
from warp_psr.normals import build_normal_field
from warp_psr.sample_adjoint import replay_samples
from warp_psr.splat_adjoint import replay_density, replay_normals
from test_sample_adjoint import inputs, frozen_layout, device


def parity(density,normal,reference_density,reference_normal):
    for actual,expected in zip(density.aggregates+density.levels,
                               reference_density.aggregates+reference_density.levels):
        np.testing.assert_array_equal(actual.keys.numpy(),expected.keys.numpy())
        np.testing.assert_array_equal(actual.values.numpy(),expected.values.numpy())
    np.testing.assert_array_equal(density.depths_and_weights.numpy(),reference_density.depths_and_weights.numpy())
    np.testing.assert_array_equal(normal.records.keys.numpy(),reference_normal.records.keys.numpy())
    np.testing.assert_array_equal(normal.records.values.numpy(),reference_normal.records.values.numpy())
    np.testing.assert_array_equal(normal.average.numpy(),reference_normal.average.numpy())


@pytest.mark.parametrize('dtype',[wp.vec3,wp.vec3d])
@pytest.mark.parametrize('kernel_depth,target',[(1,1.5),(3,20.0)])
def test_splat_replay_and_vjp(device,dtype,kernel_depth,target):
    p,n=inputs(device,dtype,jitter=1e-2);frozen=frozen_layout(p,n)
    sample=frozen if p.device.is_cuda else replay_samples(p,n,frozen,depth=3)
    density=build_density(sample,3,kernel_depth=kernel_depth,samples_per_node=target)
    normal=build_normal_field(sample,density,1,3)
    with wp.Tape() as tape:
        tracked_sample=replay_samples(p,n,frozen,depth=3)
        tracked_density=replay_density(tracked_sample,density,samples_per_node=target)
        tracked_normal=replay_normals(tracked_sample,tracked_density,normal,tape=tape,minimum=1,maximum=3)
    parity(tracked_density,tracked_normal,density,normal)
    rng=np.random.default_rng(58)
    outputs=[tracked_density.depths_and_weights,tracked_normal.records.values,tracked_normal.average]
    seeds=[rng.normal(size=o.numpy().shape)*scale for o,scale in zip(outputs,[.1,.001,.1])]
    tape.backward(grads={o:wp.array(s,dtype=o.dtype,device=device) for o,s in zip(outputs,seeds)})
    original=[a.numpy().copy() for a in [p,n]]
    gradients=[a.grad.numpy().copy() for a in [p,n]]
    epsilon=1e-4 if dtype==wp.vec3 else 1e-6
    for which,a in enumerate([p,n]):
        direction=rng.normal(size=original[which].shape).astype(original[which].dtype)
        losses=[]
        for sign in [1,-1]:
            a.assign(original[which]+sign*epsilon*direction)
            np.testing.assert_array_equal(np.argmin(p.numpy(),axis=0),np.argmin(original[0],axis=0))
            np.testing.assert_array_equal(np.argmax(p.numpy(),axis=0),np.argmax(original[0],axis=0))
            if p.device.is_cuda:
                current=frozen_layout(p,n)
                np.testing.assert_array_equal(current.input_group_indices.numpy(),frozen.input_group_indices.numpy())
            else:
                current=replay_samples(p,n,frozen,depth=3)
            d=build_density(current,3,kernel_depth=kernel_depth,samples_per_node=target)
            nf=build_normal_field(current,d,1,3)
            np.testing.assert_array_equal(nf.records.keys.numpy(),normal.records.keys.numpy())
            values=[d.depths_and_weights,nf.records.values,nf.average]
            losses.append(sum(np.sum(o.numpy().astype(float)*s) for o,s in zip(values,seeds)))
        a.assign(original[which])
        assert np.isfinite(gradients[which]).all()
        np.testing.assert_allclose(np.sum(gradients[which].astype(float)*direction),(losses[0]-losses[1])/(2*epsilon),
            rtol=1e-2 if dtype==wp.vec3 else 1e-6,atol=2e-3 if dtype==wp.vec3 else 2e-7)


@pytest.mark.parametrize('dtype',[wp.vec3,wp.vec3d])
@pytest.mark.parametrize('cancelled',[False,True])
def test_multiple_average_tiles_and_cancelled_groups(device,dtype,cancelled):
    from types import SimpleNamespace
    rng=np.random.default_rng(21)
    scalar=wp.float64 if dtype==wp.vec3d else wp.float32
    pair=wp.vec2d if dtype==wp.vec3d else wp.vec2
    count=277
    weights=rng.integers(1,5,size=count)
    positions=rng.uniform(.1,.9,size=(count,3))
    normal_values=np.zeros((count,3)) if cancelled else rng.normal(size=(count,3))
    sample=SimpleNamespace(keys=wp.array(np.arange(count),dtype=wp.int64,device=device),
        weights=wp.array(weights,dtype=scalar,device=device),
        first_indices=wp.array(rng.permutation(count),dtype=int,device=device),
        position_sums=wp.array(positions*weights[:,None],dtype=dtype,device=device,requires_grad=True),
        normal_sums=wp.array(normal_values,dtype=dtype,device=device,requires_grad=True),
        positions=wp.array(positions,dtype=dtype,device=device))
    dw=wp.array(np.column_stack([rng.uniform(2.3,2.7,count),rng.uniform(.3,.8,count)]),
                dtype=pair,device=device,requires_grad=True)
    density=SimpleNamespace(depths_and_weights=dw)
    reference=build_normal_field(sample,density,1,4)
    with wp.Tape() as tape:
        result=replay_normals(sample,density,reference,tape=tape,minimum=1,maximum=4)
    np.testing.assert_array_equal(result.average.numpy(),reference.average.numpy())
    np.testing.assert_array_equal(result.records.values.numpy(),reference.records.values.numpy())
    seed=np.array([[.7,-.3,.2]])
    tape.backward(grads={result.average:wp.array(seed,dtype=dtype,device=device)})
    assert np.isfinite(sample.normal_sums.grad.numpy()).all()
    gradient=dw.grad.numpy().copy()
    if cancelled:
        assert len(result.records.keys)==0
        np.testing.assert_array_equal(gradient,np.zeros_like(gradient))
        return
    direction=rng.normal(size=gradient.shape).astype(gradient.dtype)
    original=dw.numpy().copy()
    epsilon=1e-3 if dtype==wp.vec3 else 1e-6
    losses=[]
    for sign in [1,-1]:
        dw.assign(original+sign*epsilon*direction)
        nf=build_normal_field(sample,density,1,4)
        np.testing.assert_array_equal(nf.records.keys.numpy(),reference.records.keys.numpy())
        losses.append(np.sum(nf.average.numpy().astype(float)*seed))
    np.testing.assert_allclose(np.sum(gradient.astype(float)*direction),(losses[0]-losses[1])/(2*epsilon),
        rtol=1e-2 if dtype==wp.vec3 else 1e-6,atol=5e-4 if dtype==wp.vec3 else 1e-7)
