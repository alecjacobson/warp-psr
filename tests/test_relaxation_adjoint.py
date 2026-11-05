"""VJPs of the actual finite ordered updates, including initial-state dependence."""
import itertools
import numpy as np
import pytest
import warp as wp
import warp.sparse as sparse
from warp_psr.relaxation import row_schedule, relax_scheduled, relax_sliced
from warp_psr.relaxation_adjoint import differentiable_relaxation


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    wp.init()
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def system(device, dtype, pattern='full', tracked=True, depth=1):
    xyz = list(itertools.product(range((1 << depth) + 1), repeat=3))
    if pattern == 'sparse':
        xyz = [p for i, p in enumerate(xyz) if i % 4]
    index = {p: i for i, p in enumerate(xyz)}
    rows, cols, values = [], [], []
    for p, i in index.items():
        diagonal = 1.0
        for delta in itertools.product((-1, 0, 1), repeat=3):
            j = index.get(tuple(x + y for x, y in zip(p, delta)))
            if j is None or i == j:
                continue
            v = np.sin(i * 13 + j * 3) * 0.2
            rows.append(i); cols.append(j); values.append(v)
            diagonal += abs(v)
        rows.append(i); cols.append(i); values.append(diagonal)
    A = sparse.bsr_from_triplets(len(xyz), len(xyz), wp.array(rows, dtype=int, device=device),
                                wp.array(cols, dtype=int, device=device), wp.array(values, dtype=dtype, device=device))
    A.values.requires_grad = tracked
    keys = wp.array([(depth << 48) | (x << 32) | (y << 16) | z for x, y, z in xyz], dtype=wp.int64, device=device)
    b = wp.array(np.sin(np.arange(len(xyz)) + 0.3), dtype=dtype, device=device, requires_grad=tracked)
    initial = wp.array(np.cos(np.arange(len(xyz)) - 0.7), dtype=dtype, device=device, requires_grad=tracked)
    return A, b, initial, keys


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('mode', ['forward', 'reverse', 'sliced'])
@pytest.mark.parametrize('iters', [0, 1, 3])
@pytest.mark.parametrize('pattern', ['full', 'sparse'])
def test_finite_iteration_vjp(device, dtype, mode, iters, pattern):
    check_vjp(device, dtype, mode, iters, pattern, depth=1)


def check_vjp(device, dtype, mode, iters, pattern, depth):
    A, b, initial, keys = system(device, dtype, pattern, depth=depth)
    schedule = row_schedule(keys, depth, sliced=mode == 'sliced')
    options = dict(iters=iters, forward=mode != 'reverse', sliced_depth=depth if mode == 'sliced' else None)
    reference = wp.clone(initial, requires_grad=False)
    if mode == 'sliced':
        relax_sliced(A, b, reference, schedule, depth, iters)
    else:
        relax_scheduled(A, b, reference, schedule, iters, forward=mode != 'reverse')
    with wp.Tape() as tape:
        out = differentiable_relaxation(A, b, initial, schedule, tape=tape, **options)
    expected = reference.numpy()
    np.testing.assert_array_equal(out.numpy(), expected)
    seed = np.random.default_rng(12).normal(size=len(b))
    tape.backward(grads={out: wp.array(seed, dtype=dtype, device=device)})
    arrays = [A.values, b, initial]
    original = [a.numpy().copy() for a in arrays]
    gradients = [a.grad.numpy().copy() for a in arrays]
    # Test each input independently; a near-converged implicit adjoint cannot
    # pass the initial-state derivative of an incomplete iteration map.
    random = np.random.default_rng(23)
    epsilon = 2e-3 if dtype == wp.float32 else 1e-5
    for which in range(3):
        direction = random.normal(size=original[which].shape).astype(original[which].dtype)
        estimates = []
        for sign in [1, -1]:
            arrays[which].assign(original[which] + sign * epsilon * direction)
            local = wp.Tape()
            result = differentiable_relaxation(A, b, initial, schedule, tape=local, **options)
            estimates.append(float(result.numpy().astype(np.float64) @ seed))
            arrays[which].assign(original[which])
        finite = (estimates[0] - estimates[1]) / (2 * epsilon)
        analytic = np.sum(gradients[which].astype(np.float64) * direction)
        np.testing.assert_allclose(analytic, finite, rtol=3e-3 if dtype == wp.float32 else 2e-7,
                                   atol=2e-4 if dtype == wp.float32 else 2e-9)
    np.testing.assert_array_equal(out.numpy(), expected)
    for a, data in zip(arrays, original):
        np.testing.assert_array_equal(a.numpy(), data)


@pytest.mark.parametrize('depth', [0, 2, 3])
@pytest.mark.parametrize('mode', ['forward', 'reverse', 'sliced'])
def test_larger_groups_and_slice_boundaries(device, depth, mode):
    check_vjp(device, wp.float64, mode, 3, 'sparse', depth)


@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
def test_composition_and_tape_reuse(device, dtype):
    A, b, initial, keys = system(device, dtype)
    schedule = row_schedule(keys, 1)
    with wp.Tape() as tape:
        first = differentiable_relaxation(A, b, initial, schedule, tape=tape, iters=1)
        second = differentiable_relaxation(A, b, first, schedule, tape=tape, iters=1, forward=False)
    saved = [first.numpy().copy(), second.numpy().copy()]
    seed = wp.full(len(b), 1., dtype=dtype, device=device)
    tape.backward(grads={second: seed})
    gradients = [a.grad.numpy().copy() for a in [A.values, b, initial]]
    tape.zero()
    tape.backward(grads={second: seed})
    for a, g in zip([A.values, b, initial], gradients):
        np.testing.assert_allclose(a.grad.numpy(), g, rtol=1e-6 if dtype == wp.float32 else 1e-14, atol=1e-7)
    for a, data in zip([first, second], saved):
        np.testing.assert_array_equal(a.numpy(), data)
    direction = np.random.default_rng(51).normal(size=len(b))
    original = initial.numpy().copy()
    eps = 1e-3 if dtype == wp.float32 else 1e-5
    values = []
    for sign in [1, -1]:
        initial.assign(original + sign * eps * direction)
        local = wp.Tape()
        x = differentiable_relaxation(A, b, initial, schedule, tape=local, iters=1)
        y = differentiable_relaxation(A, b, x, schedule, tape=local, iters=1, forward=False)
        values.append(y.numpy().astype(np.float64).sum())
    np.testing.assert_allclose(gradients[2].astype(np.float64) @ direction, (values[0] - values[1]) / (2 * eps),
                               rtol=1e-2, atol=2e-4 if dtype == wp.float32 else 1e-9)


def test_untracked_matrix_and_zero_diagonal(device):
    dtype = wp.float64
    A, b, initial, keys = system(device, dtype, tracked=False)
    b.requires_grad = True
    data = A.values.numpy()
    # Row zero has a zero diagonal, so its b[0] never affects its update.
    offsets, columns = A.offsets.numpy(), A.columns.numpy()
    data[np.flatnonzero(columns[:offsets[1]] == 0)[0]] = 0
    A.values.assign(data)
    schedule = row_schedule(keys, 1)
    with wp.Tape() as tape:
        out = differentiable_relaxation(A, b, initial, schedule, tape=tape, iters=2)
    tape.backward(grads={out: wp.ones_like(out, requires_grad=False)})
    assert b.grad.numpy()[0] == 0.0
    assert np.isfinite(b.grad.numpy()).all()
    assert A.values.grad is None and initial.grad is None
    np.testing.assert_equal(out.numpy()[0], initial.numpy()[0])


def test_independent_streams_and_backward_handoff(device):
    if device == 'cpu':
        pytest.skip('CUDA stream handoff')
    cases = []
    for depth in [1, 2]:
        A, b, initial, keys = system(device, wp.float64, depth=depth)
        schedule = row_schedule(keys, depth, sliced=True)
        # Finish input preparation before handing those inputs to a new stream.
        stream = wp.Stream(device)
        stream.wait_stream(wp.get_stream(device))
        tape = wp.Tape()
        with wp.ScopedStream(stream), tape:
            out = differentiable_relaxation(A, b, initial, schedule, tape=tape, iters=3, sliced_depth=depth)
        cases.append((A, b, initial, schedule, out, tape, depth))
    # Backward on another stream waits for the corresponding saved history.
    backward_stream = wp.Stream(device)
    with wp.ScopedStream(backward_stream):
        for A, b, initial, schedule, out, tape, depth in reversed(cases):
            tape.backward(grads={out: wp.ones_like(out, requires_grad=False)})
    wp.synchronize_stream(backward_stream)
    for A, b, initial, schedule, out, tape, depth in cases:
        expected = [a.grad.numpy().copy() for a in [A.values, b, initial]]
        tape.zero()
        with wp.Tape() as serial:
            repeated = differentiable_relaxation(A, b, initial, schedule, tape=serial, iters=3, sliced_depth=depth)
        serial.backward(grads={repeated: wp.ones_like(repeated, requires_grad=False)})
        np.testing.assert_array_equal(repeated.numpy(), out.numpy())
        for a, gradient in zip([A.values, b, initial], expected):
            np.testing.assert_allclose(a.grad.numpy(), gradient, rtol=1e-12, atol=1e-13)


def test_overlapping_parameter_gradient_views(device):
    A, b, initial, keys = system(device, wp.float64)
    schedule = row_schedule(keys, 1)
    with wp.Tape() as tape:
        out = differentiable_relaxation(A, b, initial, schedule, tape=tape, iters=3)
    tape.backward(grads={out: wp.ones_like(out, requires_grad=False)})
    expected = A.values.grad.numpy().copy()
    expected[:len(b)] += b.grad.numpy()
    tape.zero()
    # Callers may accumulate several logical parameter gradients into a shared
    # buffer. This case needs atomics despite each row owning its CSR entries.
    b.grad = A.values.grad[:len(b)]
    with wp.Tape() as shared:
        out = differentiable_relaxation(A, b, initial, schedule, tape=shared, iters=3)
    shared.backward(grads={out: wp.ones_like(out, requires_grad=False)})
    np.testing.assert_allclose(A.values.grad.numpy(), expected, rtol=1e-12, atol=1e-13)
