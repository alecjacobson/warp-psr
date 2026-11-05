"""Compare finite root solves and their VJPs with the pinned forward algorithm."""
import numpy as np
import pytest
import warp as wp
import warp.sparse as sparse
from warp.optim.linear import cg
from warp_psr.solver import constant_mode_operator, padded_root_system
from warp_psr.solver_adjoint import finite_root_cg, functional_spmv


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    wp.init()
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def system(device, dtype, n=8):
    rng = np.random.default_rng(19)
    a = rng.normal(size=(n, n))
    a = a.T @ a + np.eye(n) * 2
    i, j = np.indices(a.shape)
    matrix = sparse.bsr_from_triplets(n, n, wp.array(i.ravel(), dtype=int, device=device),
        wp.array(j.ravel(), dtype=int, device=device), wp.array(a.ravel(), dtype=dtype, device=device))
    matrix.values.requires_grad = True
    b = wp.array(rng.normal(size=n), dtype=dtype, device=device, requires_grad=True)
    initial = wp.array(rng.normal(size=n) * 0.1, dtype=dtype, device=device, requires_grad=True)
    return matrix, b, initial


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('steps', [1, 3, 8])
@pytest.mark.parametrize('dc', [False, True])
def test_root_forward_and_vjp(device, dtype, steps, dc):
    matrix, b, initial = system(device, dtype)
    options = dict(maxiter=steps, tol=1e-3, add_dc_term=dc)
    reference = wp.clone(initial, requires_grad=False)
    iterations, residual, tolerance = cg(constant_mode_operator(matrix) if dc else matrix,
        b, reference, tol=1e-3, maxiter=steps, check_every=1, use_cuda_graph=False)
    with wp.Tape() as tape:
        out, info = finite_root_cg(matrix, b, initial, **options)
    np.testing.assert_array_equal(out.numpy(), reference.numpy())
    assert info == dict(iterations=iterations, residual_norm=residual, absolute_tolerance=tolerance)
    seed = np.random.default_rng(61).normal(size=len(b))
    tape.backward(grads={out: wp.array(seed, dtype=dtype, device=device)})
    arrays = [matrix.values, b, initial]
    original = [a.numpy().copy() for a in arrays]
    gradients = [a.grad.numpy().copy() for a in arrays]
    rng = np.random.default_rng(32)
    # Near-converged float32 CG loses initial-state sensitivity below its
    # rounding noise at 1e-3; a step sweep resolves it at 1e-2. Float64 checks
    # retain the tight derivative budget, and every perturbation checks that
    # the stopping branch stayed fixed.
    eps = 1e-2 if dtype == wp.float32 else 1e-5
    for which, a in enumerate(arrays):
        direction = rng.normal(size=original[which].shape).astype(original[which].dtype)
        if which == 0:
            direction = (direction.reshape(8, 8) + direction.reshape(8, 8).T).ravel() * 0.5
        losses = []
        for sign in [1, -1]:
            a.assign(original[which] + sign * eps * direction)
            value, perturbed = finite_root_cg(matrix, b, initial, **options)
            assert perturbed['iterations'] == info['iterations'], 'perturbation crosses a stopping branch'
            losses.append(value.numpy().astype(np.float64) @ seed)
        a.assign(original[which])
        np.testing.assert_allclose(gradients[which].astype(np.float64) @ direction,
            (losses[0] - losses[1]) / (2 * eps),
            rtol=2e-2 if dtype == wp.float32 else 3e-6, atol=3e-4 if dtype == wp.float32 else 1e-8)


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
def test_spmv_alias_input_and_repeated_tape(device, dtype):
    matrix, x, _ = system(device, dtype)
    reference = wp.clone(x, requires_grad=False)
    sparse.bsr_mv(matrix, reference, reference, alpha=-0.4, beta=0.7, tile_size=-1)
    with wp.Tape() as tape:
        out = functional_spmv(matrix, x, x, alpha=-0.4, beta=0.7)
    np.testing.assert_array_equal(out.numpy(), reference.numpy())
    tape.backward(grads={out: wp.ones_like(out, requires_grad=False)})
    expected = -0.4 * matrix.values.numpy().reshape(8, 8).sum(axis=0) + 0.7
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-6, atol=1e-6)
    first = x.grad.numpy().copy()
    tape.zero()
    tape.backward(grads={out: wp.ones_like(out, requires_grad=False)})
    np.testing.assert_allclose(x.grad.numpy(), first, rtol=1e-6, atol=1e-6)


def test_zero_residual_retains_initial_gradient(device):
    matrix, b, initial = system(device, wp.float64)
    sparse.bsr_mv(matrix, initial, b)
    with wp.Tape() as tape:
        out, info = finite_root_cg(matrix, b, initial, maxiter=8, atol=1e-10)
    assert info['iterations'] == 0
    tape.backward(grads={out: wp.ones_like(out, requires_grad=False)})
    np.testing.assert_array_equal(initial.grad.numpy(), np.ones(len(b)))
    np.testing.assert_array_equal(b.grad.numpy(), np.zeros(len(b)))
    np.testing.assert_array_equal(matrix.values.grad.numpy(), np.zeros(matrix.values.shape))


def test_padded_root_constant_mode(device):
    matrix, rhs, _ = system(device, wp.float64)
    keys = wp.array([(x << 32) | (y << 16) | z for x in range(2) for y in range(2) for z in range(2)],
                    dtype=wp.int64, device=device)
    matrix, rhs = padded_root_system(matrix, rhs, keys)
    matrix.values.requires_grad = True
    rhs.requires_grad = True
    initial = wp.zeros(64, dtype=wp.float64, device=device, requires_grad=True)
    reference = wp.clone(initial, requires_grad=False)
    count, residual, tolerance = cg(constant_mode_operator(matrix), rhs, reference,
        maxiter=3, tol=1e-3, check_every=1, use_cuda_graph=False)
    with wp.Tape() as tape:
        out, info = finite_root_cg(matrix, rhs, initial, maxiter=3, tol=1e-3, add_dc_term=True)
    np.testing.assert_array_equal(out.numpy(), reference.numpy())
    assert info == dict(iterations=count, residual_norm=residual, absolute_tolerance=tolerance)
    rng = np.random.default_rng(28)
    seed = rng.normal(size=64)
    tape.backward(grads={out: wp.array(seed, dtype=wp.float64, device=device)})
    gradient = rhs.grad.numpy().copy()
    original = rhs.numpy().copy()
    direction = rng.normal(size=64) * (original != 0)
    losses = []
    for sign in [1, -1]:
        rhs.assign(original + sign * 1e-5 * direction)
        perturbed, _ = finite_root_cg(matrix, rhs, initial, maxiter=3, tol=1e-3, add_dc_term=True)
        losses.append(perturbed.numpy() @ seed)
    np.testing.assert_allclose(gradient @ direction, (losses[0] - losses[1]) / 2e-5, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('capacity_only', [False, True])
def test_tiled_spmv_forward_and_vjp(device, dtype, capacity_only):
    n = 8 if capacity_only else 133
    matrix, x, _ = system(device, dtype, n=n)
    if capacity_only:
        capacity = 200*n
        values = np.full(capacity,1e20)
        values[:n*n] = matrix.values.numpy()
        columns = np.full(capacity,2147483647,dtype=np.int32)
        columns[:n*n] = matrix.columns.numpy()
        matrix.values = wp.array(values,dtype=dtype,device=device,requires_grad=True)
        matrix.columns = wp.array(columns,dtype=int,device=device)
        matrix.nnz = capacity
    # Public Warp chooses 64-lane tiles on CUDA above 128 entries/row.
    reference = wp.clone(x, requires_grad=False)
    sparse.bsr_mv(matrix, reference, reference, alpha=-.4, beta=.7)
    with wp.Tape() as tape:
        out = functional_spmv(matrix, x, x, alpha=-.4, beta=.7)
    np.testing.assert_array_equal(out.numpy(), reference.numpy())
    rng = np.random.default_rng(529)
    seed = rng.normal(size=len(x))
    tape.backward(grads={out:wp.array(seed,dtype=dtype,device=device)})
    a = matrix.values.numpy()[:n*n].reshape(n,n).astype(float)
    expected_x = -.4*(a.T@seed)+.7*seed
    expected_a = -.4*seed[:,None]*x.numpy().astype(float)[None,:]
    np.testing.assert_allclose(x.grad.numpy(),expected_x,rtol=2e-5 if dtype==wp.float32 else 1e-12,
                               atol=1e-4 if dtype==wp.float32 else 1e-12)
    np.testing.assert_allclose(matrix.values.grad.numpy()[:n*n].reshape(n,n),expected_a,
                               rtol=1e-6 if dtype==wp.float32 else 1e-12,atol=1e-7 if dtype==wp.float32 else 1e-12)

    if capacity_only:
        np.testing.assert_array_equal(matrix.values.grad.numpy()[n*n:],np.zeros(matrix.nnz-n*n))
