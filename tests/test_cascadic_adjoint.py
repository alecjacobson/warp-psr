"""Cascadic replay through assembled parameters, compared to the public solve."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import warp as wp
from warp_psr import reconstruct
from warp_psr.keyed import Records
from warp_psr.cascadic_adjoint import prepare_solve, replay_solve
from warp_psr.solver import solve_levels
from warp_psr.constraints import point_constraints
from test_sample_adjoint import inputs


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [wp.vec3, wp.vec3d])
@pytest.mark.parametrize('depth,base,weight', [(3, 0, 0.0), (3, 1, 2.0), (4, 1, 2.0)])
def test_cascadic_assembly_forward_and_vjp(dtype, depth, base, weight):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    p, n = inputs('cuda:0', dtype, jitter=.01)
    result = reconstruct(p, n, depth=depth, full_depth=base, base_depth=base, iters=2, samples_per_node=.05)
    if weight == 0:
        result = reconstruct(p, n, depth=depth, full_depth=base, base_depth=base, point_weight=weight, iters=2, samples_per_node=.05)
    data = result.diagnostics
    assert data['levels'][-1]['depth'] == depth
    assert len(data['levels'][-1]['coefficients']) > 0
    normals, density, tree = data['normals'], data['density'], data['tree']
    normals.records.values.requires_grad = True
    normals.average.requires_grad = True
    begin = 0
    levels = []
    for level in normals.levels:
        size = len(level.keys)
        if size:
            levels.append(Records(level.keys, normals.records.values[begin:begin + size]))
        else:
            levels.append(Records(level.keys, wp.empty(0, dtype=normals.records.values.dtype, device=p.device, requires_grad=True)))
        begin += size
    normals.levels = levels
    for record in density.aggregates:
        record.values.requires_grad = True
    prepared = prepare_solve(result)
    with wp.Tape() as tape:
        corrections, trace = replay_solve(prepared, density, normals, tape=tape)
    for primal in data['levels']:
        np.testing.assert_array_equal(corrections[primal['depth']].numpy(), primal['coefficients'].numpy())
    rng = np.random.default_rng(17)
    seeds = [rng.normal(size=len(v)) for v in corrections]
    tape.backward(grads={v:wp.array(s,dtype=v.dtype,device=v.device) for v,s in zip(corrections,seeds)})
    arrays = [normals.records.values]
    if weight:
        arrays += [normals.average, density.aggregates[base].values, density.aggregates[depth].values]
    for which,array in enumerate(arrays):
        original, gradient = array.numpy().copy(), array.grad.numpy().copy()
        direction = rng.normal(size=original.shape).astype(original.dtype)
        # Positive aggregate masses and stable point bins are maintained by
        # these small perturbations; normal record keys are fixed explicitly.
        # Step sweep: f32 normal-load losses hit rounding noise at 1e-4;
        # fine aggregate positions need 1e-4 to stay inside their point bins.
        eps = (1e-4 if which == 3 else 1e-3) if dtype == wp.vec3 else 1e-6
        original_bins = []
        if weight:
            for d in range(base,depth+1):
                points=point_constraints(density,normals,tree,d,weight)
                original_bins.append(np.floor(points.positions.numpy()*(1<<d)))
        losses = []
        for sign in [1, -1]:
            array.assign(original + sign * eps * direction)
            if weight:
                for d, bins in zip(range(base,depth+1),original_bins):
                    points=point_constraints(density,normals,tree,d,weight)
                    np.testing.assert_array_equal(np.floor(points.positions.numpy()*(1<<d)),bins)
            value, perturbed_levels, _ = solve_levels(tree, None, normals, density, base_depth=base,
                solve_depth=depth, point_weight=weight, iters=2)
            losses.append(sum(v.numpy().astype(float) @ s for v,s in zip(value,seeds)))
            candidate=NS(metadata=result.metadata,diagnostics=dict(data,levels=perturbed_levels))
            candidate_prepared=prepare_solve(candidate)
            with wp.Tape() as candidate_tape:
                _, candidate_trace=replay_solve(candidate_prepared,density,normals,tape=candidate_tape)
            assert [(t['iterations'],t['outer_skip']) for t in trace] == [(t['iterations'],t['outer_skip']) for t in candidate_trace]
        array.assign(original)
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(np.sum(gradient.astype(float) * direction), (losses[0] - losses[1]) / (2 * eps),
            rtol=1e-2 if dtype == wp.vec3 else 2e-5, atol=5e-3 if dtype == wp.vec3 else 2e-6,err_msg=str(which))
