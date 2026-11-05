"""Finite MG and Galerkin VJPs against the original public solver."""
import numpy as np
import pytest
import warp as wp
import warp.sparse as sparse
from warp_psr.stiffness import assemble_level_stiffness
from warp_psr.solver import regular_multigrid, prolongation
from warp_psr.multigrid_adjoint import prepare_hierarchy, finite_multigrid
from test_assembly_adjoint import keys
from test_sample_adjoint import device


def system(device, dtype, base):
    levels = [keys(d, device) for d in range(base + 1)]
    matrix = sparse.bsr_copy(assemble_level_stiffness(levels[-1], levels[-1], base), scalar_type=dtype)
    values = matrix.values.numpy()
    offsets, columns = matrix.offsets.numpy(), matrix.columns.numpy()
    for i in range(matrix.nrow):
        for j in range(offsets[i], offsets[i + 1]):
            if columns[j] == i:
                values[j] += .4 + np.random.default_rng(100 + i).uniform(.1, 1.0)
    matrix.values.assign(values)
    matrix.values.requires_grad = True
    transfers = [prolongation(levels[d - 1], levels[d], d, dtype) for d in range(1, base + 1)]
    rhs = wp.array(np.random.default_rng(42).normal(size=len(levels[-1])), dtype=dtype, device=device, requires_grad=True)
    return matrix, levels, transfers, rhs


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('base,cycles,dc', [(0, 1, False), (0, 2, False), (0, 2, True), (1, 1, False), (2, 2, False)])
def test_multigrid_forward_and_vjp(device, dtype, base, cycles, dc):
    matrix, levels, transfers, rhs = system(device, dtype, base)
    hierarchy = prepare_hierarchy(matrix, levels, transfers, add_dc_term=dc)
    options = dict(iters=1, cycles=cycles, cg_accuracy=2e-4)
    reference, _ = regular_multigrid(matrix, rhs, levels, transfers, add_dc_term=dc, **options)
    with wp.Tape() as tape:
        out, trace = finite_multigrid(hierarchy, rhs, tape=tape, **options)
    np.testing.assert_array_equal(out.numpy(), reference.numpy())
    rng = np.random.default_rng(65)
    seed = rng.normal(size=len(rhs))
    tape.backward(grads={out: wp.array(seed, dtype=dtype, device=device)})
    for array in [matrix.values, rhs]:
        original, gradient = array.numpy().copy(), array.grad.numpy().copy()
        direction = rng.normal(size=original.shape).astype(original.dtype)
        if array is matrix.values:
            offsets, columns = matrix.offsets.numpy(), matrix.columns.numpy()
            dense = rng.normal(size=(matrix.nrow, matrix.ncol))
            dense = (dense + dense.T) * .5
            direction[:] = 0
            for row in range(matrix.nrow):
                for entry in range(offsets[row], offsets[row + 1]):
                    direction[entry] = dense[row, columns[entry]]
        eps = 1e-3 if dtype == wp.float32 else 1e-6
        losses = []
        for sign in [1, -1]:
            array.assign(original + sign * eps * direction)
            value, _ = regular_multigrid(matrix, rhs, levels, transfers, add_dc_term=dc, **options)
            losses.append(value.numpy().astype(float) @ seed)
            h = prepare_hierarchy(matrix, levels, transfers, add_dc_term=dc)
            with wp.Tape() as perturbed_tape:
                _, perturbed_trace = finite_multigrid(h, rhs, tape=perturbed_tape, **options)
            assert [(t['iterations'], t['outer_skip']) for t in trace] == [(t['iterations'], t['outer_skip']) for t in perturbed_trace]
        array.assign(original)
        assert np.isfinite(gradient).all()
        np.testing.assert_allclose(np.sum(gradient.astype(float) * direction), (losses[0] - losses[1]) / (2 * eps),
                                   rtol=3e-3 if dtype == wp.float32 else 2e-6, atol=3e-3 if dtype == wp.float32 else 1e-7)
