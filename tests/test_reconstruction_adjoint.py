"""End-to-end finite differences of the actual public reconstruction."""
import numpy as np
import pytest
import warp as wp
from warp_psr import reconstruct, reconstruct_differentiable
from test_sample_adjoint import inputs


def solver_branches(result):
    from warp_psr.cascadic_adjoint import prepare_solve, replay_solve
    prepared=prepare_solve(result)
    with wp.Tape() as tape:
        _,trace=replay_solve(prepared,result.diagnostics['density'],result.diagnostics['normals'],tape=tape)
    return [(item['iterations'],item['outer_skip']) for item in trace]


def fixed_assignments(reference, current):
    a,b=reference.diagnostics,current.diagnostics
    for reducer in [np.argmin,np.argmax]:
        np.testing.assert_array_equal(reducer(a['samples'].positions.numpy(),axis=0),
                                      reducer(b['samples'].positions.numpy(),axis=0))
    if not hasattr(reference,'_test_branches'):
        reference._test_branches=solver_branches(reference)
    assert reference._test_branches==solver_branches(current)
    for key in ['input_group_indices','input_status','input_order','group_offsets']:
        np.testing.assert_array_equal(getattr(a['samples'],key).numpy(),getattr(b['samples'],key).numpy())
    for key in ['basis','levels']:
        for left,right in zip(getattr(a['tree'],key),getattr(b['tree'],key)):
            np.testing.assert_array_equal(left.keys.numpy(),right.keys.numpy())
    np.testing.assert_array_equal(a['normals'].records.keys.numpy(),b['normals'].records.keys.numpy())


@pytest.mark.cuda
@pytest.mark.parametrize('dtype',[wp.vec3,wp.vec3d])
@pytest.mark.parametrize('base,weight,extra',[(0,0.0,{}),(1,2.0,{}),(1,2.0,{'depth':4}),(1,2.0,{'base_v_cycles':2,'solve_depth':2}),(1,2.0,{'exact_interpolation':True})])
def test_reconstruction_point_and_normal_vjp(dtype,base,weight,extra):
    wp.init()
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    p,n=inputs('cuda:0',dtype,jitter=.01)
    options=dict(depth=3,full_depth=base,base_depth=base,point_weight=weight,iters=2,samples_per_node=.05)
    options.update(extra)
    tape=wp.Tape()
    result=reconstruct_differentiable(p,n,tape=tape,**options)
    assert result.diagnostics["levels"][-1]["depth"] == options.get("solve_depth",options["depth"])
    query=wp.array(np.array([[.1,.1,.1],[-.1,-.1,-.1],[.4,.7,.3],[.05,.3,-.2]]),dtype=dtype,device=p.device)
    with tape:
        value,_,valid=result.evaluate(query,centered=True)
    reference=reconstruct(p,n,**options)
    expected,_,_=reference.evaluate(query)
    np.testing.assert_array_equal(value.numpy(),expected.numpy()-reference.iso_value)
    assert np.all(valid.numpy())
    scalar=wp.float64 if dtype==wp.vec3d else wp.float32
    seed=np.array([.7,-.5,.9,.3])
    tape.backward(grads={value:wp.array(seed,dtype=scalar,device=p.device)})
    rng=np.random.default_rng(283)
    for array in [p,n]:
        original,gradient=array.numpy().copy(),array.grad.numpy().copy()
        direction=rng.normal(size=original.shape).astype(original.dtype)
        eps=1e-4 if dtype==wp.vec3 else 1e-6
        losses=[]
        for sign in [1,-1]:
            array.assign(original+sign*eps*direction)
            candidate=reconstruct(p,n,**options)
            fixed_assignments(reference,candidate)
            v,_,_=candidate.evaluate(query)
            losses.append((v.numpy().astype(float)-candidate.iso_value)@seed)
        array.assign(original)
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(np.sum(gradient.astype(float)*direction),(losses[0]-losses[1])/(2*eps),
            rtol=2e-2 if dtype==wp.vec3 else 3e-5,atol=2e-3 if dtype==wp.vec3 else 2e-7)


@pytest.mark.cuda
def test_repeated_backward_and_composed_calls():
    wp.init()
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    p,n=inputs('cuda:0',wp.vec3d,jitter=.01)
    original=[p.numpy().copy(),n.numpy().copy()]
    tape=wp.Tape()
    a=reconstruct_differentiable(p,n,tape=tape,depth=3,full_depth=1,samples_per_node=.05,iters=2)
    b=reconstruct_differentiable(p,n,tape=tape,depth=3,full_depth=1,samples_per_node=.05,iters=3)
    query=wp.array([[.1,.1,.1]],dtype=wp.vec3d,device=p.device,requires_grad=True)
    with tape:
        va,_,_=a.evaluate(query,centered=True)
        vb,_,_=b.evaluate(query,centered=True)
    seed=wp.ones(1,dtype=wp.float64,device=p.device)
    arrays=[p,n,query]
    tape.backward(grads={va:seed})
    first=[x.grad.numpy().copy() for x in arrays]
    tape.zero();tape.backward(grads={vb:seed})
    second=[x.grad.numpy().copy() for x in arrays]
    tape.zero();tape.backward(grads={va:seed,vb:seed})
    for x,ga,gb in zip(arrays,first,second):
        np.testing.assert_allclose(x.grad.numpy(),ga+gb,rtol=1e-10,atol=1e-11)
    again=[x.grad.numpy().copy() for x in arrays]
    tape.zero();tape.backward(grads={va:seed,vb:seed})
    for x,g in zip(arrays,again):np.testing.assert_allclose(x.grad.numpy(),g,rtol=1e-10,atol=1e-11)
    for x,v in zip([p,n],original):np.testing.assert_array_equal(x.numpy(),v)


@pytest.mark.cuda
def test_api_guard_and_cross_stream_backward():
    wp.init()
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    p,n=inputs('cuda:0',wp.vec3,jitter=.01)
    tape=wp.Tape()
    with tape:
        with pytest.raises(ValueError,match='outside'):
            reconstruct_differentiable(p,n,tape=tape,depth=3,full_depth=1)
    with pytest.raises(TypeError,match='wp.Tape'):
        reconstruct_differentiable(p,n,tape=None)
    forward,query_stream,back=wp.Stream(p.device),wp.Stream(p.device),wp.Stream(p.device)
    # Input construction was on the default stream.
    source_ready=wp.get_stream(p.device).record_event()
    with wp.ScopedStream(forward):
        forward.wait_event(source_ready)
        result=reconstruct_differentiable(p,n,tape=tape,depth=3,full_depth=1,samples_per_node=.05,iters=2)
    with wp.ScopedStream(query_stream):
        q=wp.array([[.1,.1,.1]],dtype=wp.vec3,device=p.device)
        with tape:v,_,_=result.evaluate(q,centered=True)
        loss_ready=query_stream.record_event()
    with wp.ScopedStream(back):
        back.wait_event(loss_ready)
        tape.backward(grads={v:wp.ones(1,dtype=wp.float32,device=p.device)})
        wp.synchronize_stream(back)
    assert np.isfinite(p.grad.numpy()).all() and np.linalg.norm(p.grad.numpy())>0
    assert np.isfinite(n.grad.numpy()).all() and np.linalg.norm(n.grad.numpy())>0


@pytest.mark.cuda
@pytest.mark.parametrize('case',['cancelled','omitted_nan','positions_only'])
def test_degenerate_and_single_input_gradients(case):
    wp.init()
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    p,n=inputs('cuda:0',wp.vec3d,jitter=.01)
    if case=='cancelled':
        pos=p.numpy()[::3];normal=n.numpy()[::3]
        p=wp.array(np.repeat(pos,2,axis=0),dtype=wp.vec3d,device=p.device,requires_grad=True)
        paired=np.empty((2*len(normal),3));paired[::2]=normal;paired[1::2]=-normal
        n=wp.array(paired,dtype=wp.vec3d,device=p.device,requires_grad=True)
    elif case=='omitted_nan':
        data=n.numpy();data[0]=np.nan;n.assign(data)
    else:
        n.requires_grad=False
    tape=wp.Tape();result=reconstruct_differentiable(p,n,tape=tape,depth=3,full_depth=1,iters=2)
    query=wp.array([[.1,.1,.1]],dtype=wp.vec3d,device=p.device)
    with tape:value,_,_=result.evaluate(query,centered=True)
    tape.backward(grads={value:wp.ones(1,dtype=wp.float64,device=p.device)})
    assert np.isfinite(p.grad.numpy()).all()
    if n.requires_grad:assert np.isfinite(n.grad.numpy()).all()
    if case=='cancelled':
        np.testing.assert_array_equal(value.numpy(),[0.])
        np.testing.assert_array_equal(p.grad.numpy(),np.zeros(p.shape+(3,)))
        np.testing.assert_array_equal(n.grad.numpy(),np.zeros(n.shape+(3,)))
    elif case=='omitted_nan':
        np.testing.assert_array_equal(n.grad.numpy()[0],np.zeros(3))
    else:
        assert n.grad is None
    with pytest.raises(ValueError,match='constant scalar'):
        reconstruct_differentiable(p,n,tape=wp.Tape(),point_weight=wp.ones(1,dtype=wp.float64,device=p.device,requires_grad=True))


def test_default_import_keeps_adjoint_modules_lazy():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"import sys,warp_psr; assert 'warp_psr.reconstruction_adjoint' not in sys.modules; assert 'warp_psr.field_adjoint' not in sys.modules"],check=True)
