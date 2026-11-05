"""Screening derivatives checked against both public FEM and stencil assembly."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import warp as wp
import warp.sparse as sparse
from warp_psr.assembly_adjoint import record_screening_vjp, replay_constraints
from warp_psr.constraints import point_constraints
from warp_psr.stiffness import assemble_level_stiffness
from warp_psr.screening_stencils import add_screening_stencil
from warp_psr.forms import assemble_screening, assemble_point_load, point_quadrature
from warp_psr.spaces import make_level_space
from warp_psr.integration import level_integration_grid
from test_sample_adjoint import device


def keys(depth, device):
    xyz = np.indices(((1 << depth) + 1,) * 3).reshape(3, -1).T.astype(np.int64)
    return wp.array((depth << 48) | (xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2], dtype=wp.int64, device=device)


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('cross', [False, True])
@pytest.mark.parametrize('fem', [False, True])
def test_point_form_vjp_against_primal(device, dtype, cross, fem):
    if fem and device == 'cpu':
        pytest.skip('production Nanogrid geometry requires CUDA')
    rng = np.random.default_rng(971)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    depth, trial_depth = 2, 1 if cross else 2
    rows, cols = keys(depth, device), keys(trial_depth, device)
    # Interior points, comfortably away from all fine/coarse knots.
    cell = rng.integers(0, 4, (37, 3))
    positions = (cell + rng.uniform(.2, .8, (37, 3))) / 4
    points = NS(positions=wp.array(positions, dtype=vector, device=device, requires_grad=True),
                weights=wp.array(rng.uniform(.2, 1, 37), dtype=dtype, device=device, requires_grad=True))
    if fem:
        tree = NS(basis=[None, None, NS(keys=rows)])
        geometry = level_integration_grid(tree, depth, dtype)
        test = make_level_space(geometry, rows, depth, depth)
        trial = make_level_space(geometry, cols, trial_depth, depth)
    def primal():
        if fem:
            quadrature = point_quadrature(test, points.positions, points.weights)
            m = assemble_screening(test, points.positions, points.weights, trial_space=trial,
                                    accumulation_dtype=wp.float64, quadrature=quadrature)
            load = assemble_point_load(test, points.positions, points.weights, quadrature=quadrature)
        else:
            m = assemble_level_stiffness(rows, cols, depth, trial_depth)
            sparse.bsr_scale(m, 0)
            load = add_screening_stencil(m, rows, cols, points, depth, trial_depth)
        return m, load
    matrix, load = primal()
    matrix.values.requires_grad = True
    load.requires_grad = True
    tape = wp.Tape()
    record_screening_vjp(matrix, rows, cols, points, depth, trial_depth=trial_depth, tape=tape, load=load)
    count = int(matrix.offsets.numpy()[-1])
    seed_matrix = np.zeros(len(matrix.values))
    seed_matrix[:count] = rng.normal(size=count)
    seed_load = rng.normal(size=len(load))
    tape.backward(grads={matrix.values: wp.array(seed_matrix, dtype=matrix.values.dtype, device=device),
                         load: wp.array(seed_load, dtype=wp.float64, device=device)})
    original_topology = [matrix.offsets.numpy(), matrix.columns.numpy()[:count]]
    for array in [points.positions, points.weights]:
        original, gradient = array.numpy().copy(), array.grad.numpy().copy()
        direction = rng.normal(size=original.shape).astype(original.dtype)
        eps = 1e-3 if dtype == wp.float32 else 1e-6
        losses = []
        for sign in [1, -1]:
            array.assign(original + sign * eps * direction)
            m, b = primal()
            np.testing.assert_array_equal(m.offsets.numpy(), original_topology[0])
            np.testing.assert_array_equal(m.columns.numpy()[:count], original_topology[1])
            losses.append(np.dot(m.values.numpy()[:count], seed_matrix[:count]) + np.dot(b.numpy(), seed_load))
        array.assign(original)
        np.testing.assert_allclose(np.sum(gradient.astype(float) * direction), (losses[0] - losses[1]) / (2 * eps),
                                   rtol=2e-3 if dtype == wp.float32 else 1e-7, atol=2e-3 if dtype == wp.float32 else 1e-7)


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('depth,exact', [(0, False), (2, False), (2, True)])
def test_point_constraints_replay(device, dtype, depth, exact):
    rng = np.random.default_rng(512)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    vec4 = wp.vec4d if dtype == wp.float64 else wp.vec4
    source_depth = 3 if exact else depth
    xyz = np.indices((1 << source_depth,) * 3).reshape(3, -1).T.astype(np.int64)
    key = (source_depth << 48) | (xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2]
    weights = rng.uniform(.5, 4, len(key))
    positions = (xyz + rng.uniform(.2, .8, xyz.shape)) / (1 << source_depth)
    data = wp.array(np.column_stack([positions * weights[:, None], weights]), dtype=vec4, device=device, requires_grad=True)
    records = NS(keys=wp.array(key, dtype=wp.int64, device=device), values=data)
    active = np.unique(xyz >> (source_depth - depth), axis=0)[::2]
    active += 524288
    active_keys = (active[:, 0] << 40) | (active[:, 1] << 20) | active[:, 2]
    tree = NS(levels=[None] * depth + [NS(keys=wp.array(active_keys, dtype=wp.int64, device=device))])
    density = NS(aggregates=[None] * source_depth + [records], scale=wp.ones(1, dtype=dtype, device=device))
    normals = NS(average=wp.array([[2, .7, 4]], dtype=vector, device=device, requires_grad=True))
    expected = point_constraints(density, normals, tree, depth, 2.3, exact)
    with wp.Tape() as tape:
        actual = replay_constraints(density, normals, tree, depth, 2.3, exact)
    np.testing.assert_array_equal(actual.positions.numpy(), expected.positions.numpy())
    np.testing.assert_array_equal(actual.weights.numpy(), expected.weights.numpy())
    sp, sw = rng.normal(size=(len(key), 3)), rng.normal(size=len(key))
    tape.backward(grads={actual.positions: wp.array(sp, dtype=vector, device=device),
                         actual.weights: wp.array(sw, dtype=dtype, device=device)})
    for array in [data, normals.average]:
        original, gradient = array.numpy().copy(), array.grad.numpy().copy()
        direction = rng.normal(size=original.shape).astype(original.dtype)
        eps = 1e-3 if dtype == wp.float32 else 1e-6
        losses = []
        for sign in [1, -1]:
            array.assign(original + sign * eps * direction)
            value = point_constraints(density, normals, tree, depth, 2.3, exact)
            losses.append(np.sum(value.positions.numpy().astype(float) * sp) + np.dot(value.weights.numpy().astype(float), sw))
        array.assign(original)
        np.testing.assert_allclose(np.sum(gradient.astype(float) * direction), (losses[0] - losses[1]) / (2 * eps),
                                   rtol=2e-3 if dtype == wp.float32 else 1e-7, atol=2e-3 if dtype == wp.float32 else 1e-7)
