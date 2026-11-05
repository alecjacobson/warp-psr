"""Continuous sample VJPs with fixed membership and normalization decisions."""
import numpy as np
import pytest
import warp as wp
from warp_psr.sample_adjoint import replay_samples
from warp_psr.samples import SampleGroups, prepare_samples


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    wp.init()
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def inputs(device, dtype, jitter=1e-3):
    rng = np.random.default_rng(71)
    centers = np.array([[.1,.2,.3],[.8,1.5,.2],[-.7,-.6,-.2],[-.3,.7,.7],[.3,-1.7,0],[.5,.8,-.8]])
    p = np.repeat(centers,3,axis=0) + rng.normal(size=(18,3))*jitter
    n = rng.normal(size=p.shape)
    return wp.array(p,dtype=dtype,device=device,requires_grad=True), wp.array(n,dtype=dtype,device=device,requires_grad=True)


def frozen_layout(p, n):
    if p.device.is_cuda:
        return prepare_samples(p,n,depth=3)
    # Construct the same geometric grouping on CPU for derivative tests. Keys
    # must reflect normalized point positions when replaying downstream splats.
    dtype = wp.float64 if p.dtype == wp.vec3d else wp.float32
    arr = lambda a,t: wp.array(a,dtype=t,device=p.device)
    data=p.numpy();scalar=data.dtype.type
    low,high=data.min(axis=0),data.max(axis=0)
    width=scalar(np.max(high-low)*scalar(1.1));inv=scalar(1.0/float(width))
    origin=(high+low)/scalar(2)-width/scalar(2)
    normalized=data[:,[1,2,0]]*inv-origin[[1,2,0]]*inv
    cell=np.maximum(0,np.ceil(normalized*scalar(8)).astype(np.int64)-1)
    keys=(3<<48)|(cell[:,0]<<32)|(cell[:,1]<<16)|cell[:,2]
    order=np.argsort(keys,kind='stable');sorted_keys=keys[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_keys))+1]
    offsets=np.r_[starts,len(data)];count=len(starts)
    groups=np.empty(len(data),dtype=int)
    groups[order]=np.repeat(np.arange(count),np.diff(offsets))
    return SampleGroups(arr(sorted_keys[starts],wp.int64),wp.zeros(count,dtype=p.dtype,device=p.device),
        wp.zeros(count,dtype=p.dtype,device=p.device),arr(np.diff(offsets),dtype),
        wp.zeros(5,dtype=dtype,device=p.device),p,n,arr(order[starts],int),0.0,
        input_status=wp.zeros(len(data),dtype=int,device=p.device),
        input_group_indices=arr(groups,int),input_order=arr(order,int),group_offsets=arr(offsets,int))


@pytest.mark.parametrize('dtype',[wp.vec3,wp.vec3d])
def test_sample_replay_and_vjp(device,dtype):
    p,n=inputs(device,dtype);frozen=frozen_layout(p,n)
    with wp.Tape() as tape:
        result=replay_samples(p,n,frozen,depth=3)
    if p.device.is_cuda:
        for name in ['transform','positions','normals','position_sums','normal_sums']:
            np.testing.assert_array_equal(getattr(result,name).numpy(),getattr(frozen,name).numpy())
    rng=np.random.default_rng(79)
    names=['position_sums','normal_sums','transform']
    seeds={name:rng.normal(size=getattr(result,name).numpy().shape) for name in names}
    tape.backward(grads={getattr(result,name):wp.array(seed,dtype=getattr(result,name).dtype,device=device) for name,seed in seeds.items()})
    arrays=[p,n];original=[a.numpy().copy() for a in arrays];grads=[a.grad.numpy().copy() for a in arrays]
    eps=1e-4 if dtype==wp.vec3 else 1e-6
    for which,a in enumerate(arrays):
        direction=rng.normal(size=original[which].shape).astype(original[which].dtype)
        losses=[]
        for sign in [1,-1]:
            a.assign(original[which]+sign*eps*direction)
            np.testing.assert_array_equal(np.argmin(p.numpy(),axis=0),np.argmin(original[0],axis=0))
            np.testing.assert_array_equal(np.argmax(p.numpy(),axis=0),np.argmax(original[0],axis=0))
            value=prepare_samples(p,n,depth=3) if p.device.is_cuda else replay_samples(p,n,frozen,depth=3)
            np.testing.assert_array_equal(value.keys.numpy(),frozen.keys.numpy())
            np.testing.assert_array_equal(value.group_offsets.numpy(),frozen.group_offsets.numpy())
            losses.append(sum(np.sum(getattr(value,name).numpy().astype(float)*seeds[name]) for name in names))
        a.assign(original[which])
        np.testing.assert_allclose(np.sum(grads[which].astype(float)*direction),(losses[0]-losses[1])/(2*eps),
            rtol=3e-3 if dtype==wp.vec3 else 2e-7,atol=5e-4 if dtype==wp.vec3 else 2e-8)
    assert result.host_status_reads==0


def test_omitted_nan_normal_does_not_poison_bounds_gradient(device):
    if device=='cpu':pytest.skip('public classification uses CUDA')
    p,n=inputs(device,wp.vec3d)
    values=n.numpy();values[0]=np.nan;n.assign(values)
    frozen=prepare_samples(p,n,depth=3)
    with wp.Tape() as tape:
        result=replay_samples(p,n,frozen,depth=3)
    tape.backward(grads={result.normal_sums:wp.ones_like(result.normal_sums,requires_grad=False),
                         result.position_sums:wp.ones_like(result.position_sums,requires_grad=False)})
    assert np.isfinite(p.grad.numpy()).all() and np.isfinite(n.grad.numpy()).all()
    np.testing.assert_array_equal(n.grad.numpy()[0],np.zeros(3))
