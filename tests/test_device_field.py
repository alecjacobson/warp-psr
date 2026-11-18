"""Exercise fused field queries, precision, validity, and first-order adjoints."""
from functools import lru_cache
import numpy as np
import pytest
import warp as wp
from warp_psr import Field, Field64, LevelField, evaluate_field
from warp_psr.field import cumulative_coefficients
from warp_psr.field_adjoint import record_cumulative_vjp
from test_sample_adjoint import device, inputs

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def query_kernel(dtype):
    field_type = Field64 if dtype == wp.float64 else Field
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3

    @wp.kernel
    def query_kernel(field: field_type, queries: wp.array[vector], values: wp.array[dtype],
                     gradients: wp.array[vector], valid: wp.array[int]):
        i = wp.tid()
        value, gradient, ok = evaluate_field(field, queries[i])
        values[i] = value
        gradients[i] = gradient
        valid[i] = ok
    return query_kernel


def query(field, queries, dtype, *, tracked=False):
    values = wp.empty(len(queries), dtype=dtype, device=queries.device, requires_grad=tracked)
    gradients = wp.empty_like(queries, requires_grad=tracked)
    valid = wp.empty(len(queries), dtype=int, device=queries.device)
    wp.launch(query_kernel(dtype), len(queries), inputs=[field.device_field(), queries, values, gradients, valid],
              device=queries.device)
    return values, gradients, valid


def adaptive_field(device, dtype, reference=True, tracked=False):
    keys = []
    for depth in range(3):
        for x in range((1 << depth) + 1):
            for y in range((1 << depth) + 1):
                for z in range((1 << depth) + 1):
                    if depth < 2 or (x <= 2 and y <= 2 and z <= 2):
                        keys.append((depth << 48) | (x << 32) | (y << 16) | z)
    keys = wp.array(keys, dtype=wp.int64, device=device)
    coefficients = wp.array(np.random.default_rng(271).normal(size=len(keys)), dtype=dtype, device=device,
                            requires_grad=tracked)
    cumulative = cumulative_coefficients(keys, coefficients, 2) if reference else None
    if cumulative is not None:
        cumulative.requires_grad = tracked
    return LevelField(keys, coefficients, (1.,0.,0.,0.,1.,0.,0.,0.,1.), (0.,0.,0.), 2, 0., cumulative)


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("reference", [False, True])
def test_fused_field_parity(device, dtype, reference):
    """Match batched queries at adaptive cells, knots, boundaries, and invalid points."""
    field = adaptive_field(device, dtype, reference)
    field.linear = (0., .5, 0., 0., 0., .5, .5, 0., 0.)
    field.translation = (.25, .25, .25)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    cube = np.concatenate([np.random.default_rng(2026).uniform(0, 1, (57, 3)),
        [[0, .3, .4], [1, .3, .4], [.5, .3, .4], [.25, .3, .4], [.75, .3, .4],
         [.5-1e-5, .3, .4], [.5+1e-5, .3, .4], [0, 0, 0], [1, 1, 1],
         [1.01, .3, .4], [np.nan, .3, .4], [np.inf, .3, .4], [-np.inf, .3, .4]]])
    model = ((cube - .25) * 2)[:, [2, 0, 1]]
    queries = wp.array(model, dtype=vector, device=device)
    actual, expected = query(field, queries, dtype), field.evaluate(queries)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a.numpy(), e.numpy())
    np.testing.assert_array_equal(actual[2].numpy()[-4:], 0)
    np.testing.assert_array_equal(np.isnan(actual[0].numpy()[-4:]), True)
    handle = field.device_field()
    np.testing.assert_equal(handle.coefficients.ptr, field.coefficients.ptr)
    np.testing.assert_equal(handle.keys.ptr, field.keys.ptr)
    empty = query(field, wp.empty(0, dtype=vector, device=device), dtype)
    np.testing.assert_equal(len(empty[0]), 0)


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
@pytest.mark.parametrize("reference", [False, True])
def test_fused_field_finite_differences(device, dtype, reference):
    """Differentiate mixed value and spatial-gradient losses through a fused query."""
    field = adaptive_field(device, dtype, reference, tracked=True)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    queries = wp.array([[.13, .21, .17], [.61, .71, .81], [.34, .43, .32]], dtype=vector,
                       device=device, requires_grad=True)
    tape = wp.Tape()
    if reference:
        record_cumulative_vjp(field.keys, field.coefficients, field.cumulative, field.depth, tape=tape)
    with tape:
        values, gradients, _ = query(field, queries, dtype, tracked=True)
    value_seed = np.array([.7, 0, 0])
    gradient_seed = np.array([[.03, -.02, .01], [-.04, .07, -.02], [0, 0, 0]])
    tape.backward(grads={values: wp.array(value_seed, dtype=dtype, device=device),
                         gradients: wp.array(gradient_seed, dtype=vector, device=device)})
    rng = np.random.default_rng(537)
    for array in (field.coefficients, queries):
        original, gradient = array.numpy().copy(), array.grad.numpy().copy()
        direction = rng.normal(size=original.shape).astype(original.dtype)
        epsilon = 1e-4 if dtype == wp.float32 else 1e-6
        losses = []
        for sign in (1, -1):
            array.assign(original + sign * epsilon * direction)
            if reference:
                field.cumulative = cumulative_coefficients(field.keys, field.coefficients, field.depth)
            value, spatial, _ = field.evaluate(queries)
            losses.append(value.numpy().astype(float) @ value_seed + np.sum(spatial.numpy().astype(float) * gradient_seed))
        array.assign(original)
        if reference:
            field.cumulative = cumulative_coefficients(field.keys, field.coefficients, field.depth)
        np.testing.assert_allclose(np.sum(gradient.astype(float) * direction), (losses[0]-losses[1])/(2*epsilon),
                                   rtol=5e-3 if dtype == wp.float32 else 1e-7,
                                   atol=2e-3 if dtype == wp.float32 else 1e-8,
                                   err_msg="coefficients" if array is field.coefficients else "queries")


@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
def test_fused_query_constant_coefficients(device, dtype):
    """Track queries alone and retain zero boundary and invalid-query adjoints."""
    field = adaptive_field(device, dtype)
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    queries = wp.array([[0, .31, .43], [1, .31, .43], [.13, .21, .17], [1.1, .3, .4]],
                       dtype=vector, device=device, requires_grad=True)
    with wp.Tape() as tape:
        values, gradients, _ = query(field, queries, dtype, tracked=True)
    tape.backward(grads={values: wp.array([1, 1, 1, 0], dtype=dtype, device=device)})
    expected = gradients.numpy()
    expected[:2, 0] = 0
    expected[-1] = 0
    np.testing.assert_allclose(queries.grad.numpy(), expected, rtol=2e-6, atol=1e-6)


def test_direct_batched_query_derivative(device):
    """Match finite differences for direct multilevel values and spatial gradients."""
    field = adaptive_field(device, wp.float64, reference=False)
    queries = wp.array([[.13, .21, .17], [.61, .71, .81]], dtype=wp.vec3d, device=device, requires_grad=True)
    value_seed = np.array([.7, 0])
    gradient_seed = np.array([[.03, -.02, .01], [-.04, .07, -.02]])
    with wp.Tape() as tape:
        values, gradients, _ = field.evaluate(queries, requires_grad=True)
    tape.backward(grads={values: wp.array(value_seed, dtype=wp.float64, device=device),
                         gradients: wp.array(gradient_seed, dtype=wp.vec3d, device=device)})
    original = queries.numpy().copy()
    direction = np.random.default_rng(537).normal(size=original.shape)
    losses = []
    for sign in (1, -1):
        queries.assign(original + sign * 1e-6 * direction)
        values, gradients, _ = field.evaluate(queries)
        losses.append(values.numpy() @ value_seed + np.sum(gradients.numpy() * gradient_seed))
    np.testing.assert_allclose(np.sum(queries.grad.numpy() * direction), (losses[0]-losses[1])/2e-6,
                               rtol=1e-7, atol=1e-8)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
def test_fused_reconstruction_adjoint(dtype):
    """Match batched reconstruction gradients and skip masked invalid queries."""
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    from warp_psr import reconstruct_differentiable
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    points, normals = inputs("cuda:0", vector, jitter=.01)
    queries = wp.array([[.1, .1, .1], [-.1, -.1, -.1], [100, 0, 0], [np.nan, 0, 0]],
                       dtype=vector, device=points.device, requires_grad=True)
    tape = wp.Tape()
    result = reconstruct_differentiable(points, normals, tape=tape, depth=3, full_depth=1,
                                        samples_per_node=.05, iters=2)
    with tape:
        fused = query(result, queries, dtype, tracked=True)
        batched = result.evaluate(queries)
    for actual, expected in zip(fused, batched):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    seeds = [wp.array([.7, -.3, 0, 0], dtype=dtype, device=points.device),
             wp.array([[.02, -.01, .03], [-.04, .07, -.01], [0, 0, 0], [0, 0, 0]],
                       dtype=vector, device=points.device)]
    tape.backward(grads=dict(zip(batched[:2], seeds)))
    expected = [array.grad.numpy().copy() for array in (points, normals, queries)]
    tape.zero()
    tape.backward(grads=dict(zip(fused[:2], seeds)))
    for array, gradient in zip((points, normals, queries), expected):
        np.testing.assert_allclose(array.grad.numpy(), gradient, rtol=2e-4 if dtype == wp.float32 else 1e-10,
                                   atol=2e-6 if dtype == wp.float32 else 1e-12)


@pytest.mark.cuda
def test_fused_query_graph_capture():
    """Replay a fused query inside a CUDA graph without allocations or host work."""
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    field = adaptive_field("cuda:0", wp.float32)
    queries = wp.array([[.13, .21, .17]], dtype=wp.vec3, device="cuda:0")
    outputs = query(field, queries, wp.float32)
    expected = outputs[0].numpy().copy()
    handle = field.device_field()
    with wp.ScopedCapture() as capture:
        wp.launch(query_kernel(wp.float32), len(queries), inputs=[handle, queries, *outputs], device="cuda:0")
    outputs[0].zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(outputs[0].numpy(), expected)


def test_invalid_handle_arrays():
    """Reject mismatched coefficient arrays before a device handle can be launched."""
    field = adaptive_field("cpu", wp.float32)
    field.cumulative = wp.empty(1, dtype=wp.float32, device="cpu")
    with pytest.raises(ValueError, match="precision and shape"):
        field.device_field()
    field.cumulative = wp.empty(len(field.keys), dtype=wp.float64, device="cpu")
    with pytest.raises(ValueError, match="precision and shape"):
        field.device_field()


@wp.func(inline=True)
def nested_query(field: Field, point: wp.vec3):
    value, gradient, valid = evaluate_field(field, point)
    return value + 0.1 * wp.length_sq(gradient)


@wp.kernel
def nested_query_kernel(field: Field, points: wp.array[wp.vec3], values: wp.array[float]):
    i = wp.tid()
    values[i] = nested_query(field, points[i]) + nested_query(field, points[i] + wp.vec3(.01, .02, .03))


@pytest.mark.cuda
def test_nested_device_function():
    """Compose two field queries in a caller's device function and differentiate."""
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    field = adaptive_field("cuda:0", wp.float32)
    queries = wp.array([[.13, .21, .17]], dtype=wp.vec3, device="cuda:0", requires_grad=True)
    values = wp.empty(1, dtype=float, device="cuda:0", requires_grad=True)
    handle = field.device_field()
    with wp.Tape() as tape:
        wp.launch(nested_query_kernel, 1, inputs=[handle, queries, values], device="cuda:0")
    tape.backward(grads={values: wp.ones_like(values, requires_grad=False)})
    original = queries.numpy().copy()
    gradient = queries.grad.numpy()[0]
    expected = []
    for axis in range(3):
        losses = []
        for sign in (1, -1):
            perturbed = original.copy()
            perturbed[0, axis] += sign * 1e-4
            queries.assign(perturbed)
            wp.launch(nested_query_kernel, 1, inputs=[handle, queries, values], device="cuda:0")
            losses.append(float(values.numpy()[0]))
        expected.append((losses[0] - losses[1]) / 2e-4)
    np.testing.assert_allclose(gradient, expected, rtol=3e-3, atol=3e-3)
