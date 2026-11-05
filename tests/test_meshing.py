"""Matched-grid coverage, topology, and actual mesh-loss directional derivatives."""
import numpy as np
import pytest
import warp as wp
from scipy.spatial import cKDTree

from warp_psr.field import LevelField
from warp_psr.meshing import extract_isosurface, extract_mesh


@wp.kernel
def analytic(points: wp.array(dtype=wp.vec3), kind: int, parameter: wp.array(dtype=float),
             values: wp.array(dtype=float)):
    i = wp.tid()
    p = points[i]
    value = wp.dot(p, p) - parameter[0]
    if kind == 1:
        a, b = p - wp.vec3(-0.48, 0.05, 0.0), p - wp.vec3(0.42, -0.03, 0.0)
        value = wp.min(wp.dot(a, a), wp.dot(b, b)) - parameter[0]
    elif kind == 2:
        value = p[0] + 0.31 * p[1] - 0.22 * p[2] - parameter[0]
    elif kind == 3:
        value = 1.0
    elif kind == 4:
        value = wp.nan
    elif kind == 5:
        if p[0] < -0.9 or p[0] > 0.95 or p[1] < -0.8 or p[1] > 0.85 or p[2] < -0.7 or p[2] > 0.75:
            value = wp.nan
    values[i] = value


@wp.kernel
def vertex_loss(vertices: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(vertices[i], vertices[i]) / float(vertices.shape[0]))


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    wp.init()
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def callback(kind, parameter):
    def evaluate(points):
        result = wp.empty(len(points), dtype=float, device=points.device, requires_grad=parameter.requires_grad)
        wp.launch(analytic, len(points), inputs=[points, kind, parameter, result], device=points.device)
        return result
    return evaluate


def canonical_faces(indices):
    faces = indices.reshape(-1, 3)
    faces = np.array([np.roll(face, -np.argmin(face)) for face in faces])
    return faces[np.lexsort(faces.T[::-1])]


def assert_same_mesh(dense, sparse):
    a, b = dense.vertices.numpy(), sparse.vertices.numpy()
    assert a.shape == b.shape
    assert dense.indices.shape == sparse.indices.shape
    if len(a):
        distance, mapping = cKDTree(a).query(b)
        assert distance.max() < 2e-6
        assert len(np.unique(mapping)) == len(a)
        np.testing.assert_array_equal(canonical_faces(dense.indices.numpy()),
                                      canonical_faces(mapping[sparse.indices.numpy()]))


@pytest.mark.parametrize('kind', [0, 1, 2, 3])
@pytest.mark.parametrize('slab_size', [1, 7])
def test_sparse_dense_same_cells_and_winding(device, kind, slab_size):
    parameter = wp.array([0.0837 if kind == 1 else 0.2941], dtype=float, device=device)
    f = callback(kind, parameter)
    settings = dict(resolution=(19, 22, 25), bounds=((-0.9, -0.8, -0.7), (0.95, 0.85, 0.75)), device=device)
    a = extract_isosurface(f, backend='dense', **settings)
    b = extract_isosurface(f, backend='sparse', slab_size=slab_size, **settings)
    assert_same_mesh(a, b)
    assert not b.vertices.requires_grad
    if kind in (0, 1):
        faces = b.indices.numpy().reshape(-1, 3)
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2), 'closed analytic surface must have no cracks'
    if kind == 3:
        assert len(b.vertices) == len(b.indices) == b.metadata['retained_cells'] == 0


def test_lipschitz_backend_requires_proven_bound_and_matches_dense(device):
    p = wp.array([0.2941], dtype=float, device=device)
    settings = dict(resolution=(19, 22, 25), bounds=((-0.9, -0.8, -0.7), (0.95, 0.85, 0.75)), device=device)
    # Also rejects queries outside this non-power-of-two extraction box.
    f = callback(5, p)
    with pytest.raises(ValueError, match='proven'):
        extract_isosurface(f, backend='lipschitz', **settings)
    # f=dot(p,p)-r², so ||grad f||=2||p|| <= this bound everywhere in the box.
    bound = 2 * np.linalg.norm([0.95, 0.85, 0.75])
    a = extract_isosurface(f, backend='dense', **settings)
    b = extract_isosurface(f, backend='lipschitz', lipschitz_bound=bound, **settings)
    assert_same_mesh(a, b)


@pytest.mark.parametrize('backend', ['dense', 'sparse', 'lipschitz'])
def test_mesh_parameter_gradient_matches_forward_finite_difference(device, backend):
    settings = dict(resolution=(13, 14, 15), bounds=((-0.9, -0.8, -0.7), (0.95, 0.85, 0.75)),
                    device=device, backend=backend, slab_size=4)
    if backend == 'lipschitz':
        settings['lipschitz_bound'] = 3.0

    def run(value, tracked=False):
        parameter = wp.array([value], dtype=float, device=device, requires_grad=tracked)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=tracked)
        mesh = extract_isosurface(callback(0, parameter), **settings)
        wp.launch(vertex_loss, len(mesh.vertices), inputs=[mesh.vertices, loss], device=device)
        return parameter, loss, mesh

    with wp.Tape() as tape:
        parameter, loss, mesh = run(0.2941, True)
    tape.backward(loss)
    actual = parameter.grad.numpy()[0]
    eps = 1e-4
    _, plus, mesh_plus = run(0.2941 + eps)
    _, minus, mesh_minus = run(0.2941 - eps)
    assert mesh.vertices.shape == mesh_plus.vertices.shape == mesh_minus.vertices.shape
    expected = (plus.numpy()[0] - minus.numpy()[0]) / (2 * eps)
    np.testing.assert_allclose(actual, expected, rtol=0.003, atol=0.001)
    # A second independent Tape must not reuse adjoints or mutable mesh buffers.
    with wp.Tape() as second:
        p2, l2, m2 = run(0.2941, True)
    second.backward(l2)
    np.testing.assert_allclose(p2.grad.numpy(), parameter.grad.numpy(), rtol=2e-6, atol=2e-6)
    assert m2.vertices.ptr != mesh.vertices.ptr


def direct_field(device, tracked=False, dtype=wp.float32):
    n = 4
    ijk = np.array([(x, y, z) for x in range(n + 1) for y in range(n + 1) for z in range(n + 1)])
    keys = (np.int64(2) << 48) | (ijk[:, 0].astype(np.int64) << 32) | (ijk[:, 1].astype(np.int64) << 16) | ijk[:, 2]
    values = np.sum((ijk / n - 0.5)**2, axis=1)
    return LevelField(wp.array(keys, dtype=wp.int64, device=device),
                      wp.array(values, dtype=dtype, device=device, requires_grad=tracked),
                      (1., 0., 0., 0., 1., 0., 0., 0., 1.), (0., 0., 0.), 2, 0.1231)


@pytest.mark.parametrize('backend', ['dense', 'sparse'])
def test_direct_field_coefficients_and_device_isovalue_gradients(device, backend):
    f = direct_field(device, True)
    iso = wp.array([f.iso_value], dtype=float, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    settings = dict(resolution=(17, 18, 19), bounds=((0.02, 0.03, 0.04), (0.98, 0.97, 0.96)), backend=backend)
    with wp.Tape() as tape:
        m = extract_mesh(f, iso_value=iso, differentiable=True, **settings)
        wp.launch(vertex_loss, len(m.vertices), inputs=[m.vertices, loss], device=device)
    tape.backward(loss)
    coefficient_grad = f.coefficients.grad.numpy()
    assert np.isfinite(coefficient_grad).all() and np.linalg.norm(coefficient_grad) > 0
    # Adding a constant to every coefficient is exactly opposite to shifting iso.
    np.testing.assert_allclose(coefficient_grad.sum(), -iso.grad.numpy()[0], rtol=3e-5, atol=1e-6)
    direction = np.random.default_rng(42).normal(size=len(coefficient_grad)).astype(np.float32)
    original = f.coefficients.numpy()
    eps = 2e-5
    results = []
    for sign in [1, -1]:
        other = direct_field(device)
        other.coefficients.assign(original + sign * eps * direction)
        mesh = extract_mesh(other, **settings)
        assert mesh.vertices.shape == m.vertices.shape
        results.append(np.mean(np.sum(mesh.vertices.numpy().astype(np.float64)**2, axis=1)))
    expected = (results[0] - results[1]) / (2 * eps)
    np.testing.assert_allclose(coefficient_grad @ direction, expected, atol=0.002, rtol=0.008)


@pytest.mark.parametrize('backend', ['dense', 'sparse', 'lipschitz'])
def test_nonfinite_field_rejected(device, backend):
    options = {'lipschitz_bound': 1.0} if backend == 'lipschitz' else {}
    with pytest.raises(ValueError, match='nonfinite'):
        extract_isosurface(callback(4, wp.array([0.0], dtype=float, device=device)), resolution=3,
                           bounds=((0., 0., 0.), (1., 1., 1.)), backend=backend, device=device, **options)


def test_unsupported_precision_and_reconstruction_gradients_rejected(device):
    settings = dict(resolution=3, bounds=((0., 0., 0.), (1., 1., 1.)))
    with pytest.raises(ValueError, match='float32'):
        extract_mesh(direct_field(device, dtype=wp.float64), **settings)
    with pytest.raises(ValueError, match='backward pass'):
        extract_mesh(direct_field(device), differentiable=True, **settings)


@pytest.mark.parametrize('resolution', [True, 1, (2, 3), (2, 2, 3.5), (2, 2, True), (3000, 3000, 3000)])
def test_bad_grid_rejected_before_sampling(resolution):
    with pytest.raises(ValueError):
        extract_isosurface(lambda _: pytest.fail('must validate before sampling'), resolution=resolution,
                           bounds=((0., 0., 0.), (1., 1., 1.)), device='cpu')


@wp.kernel
def single_cell(points: wp.array(dtype=wp.vec3), corners: wp.array(dtype=float), values: wp.array(dtype=float)):
    i = wp.tid()
    p = points[i]
    index = int(p[0]) * 4 + int(p[1]) * 2 + int(p[2])
    values[i] = corners[index]


def test_all_256_cases_and_exact_threshold_policy(device):
    settings = dict(resolution=2, bounds=((0., 0., 0.), (1., 1., 1.)), device=device)
    for code in range(256):
        data = np.array([1.0 if code & (1 << j) else -1.0 for j in range(8)], dtype=np.float32)
        corners = wp.array(data, dtype=float, device=device)

        def sample(points):
            values = wp.empty(len(points), dtype=float, device=device)
            wp.launch(single_cell, len(points), inputs=[points, corners, values], device=device)
            return values

        assert_same_mesh(extract_isosurface(sample, backend='dense', **settings),
                         extract_isosurface(sample, backend='sparse', slab_size=1, **settings))
    # Entirely on the threshold: all corners are classified >= 0; no triangles.
    corners.zero_()
    for backend in ('dense', 'sparse'):
        mesh = extract_isosurface(sample, backend=backend, **settings)
        assert len(mesh.indices) == len(mesh.vertices) == 0


def test_actual_reconstruction_dense_sparse_parity():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    from warp_psr import reconstruct
    from tools.make_fixture import points_for
    points, normals = points_for('disconnected', 256, 2026)
    r = reconstruct(wp.array(points, dtype=wp.vec3, device='cuda'),
                    wp.array(normals, dtype=wp.vec3, device='cuda'), depth=4, full_depth=2)
    matrix = np.array(r.field.linear).reshape(3, 3)
    inverse = np.linalg.inv(matrix)
    corners = np.array([(x, y, z) for x in (0.002, 0.998) for y in (0.002, 0.998) for z in (0.002, 0.998)])
    model = (corners - r.field.translation) @ inverse.T
    settings = dict(resolution=(25, 28, 31), bounds=(model.min(axis=0), model.max(axis=0)))
    a = extract_mesh(r, backend='dense', **settings)
    b = extract_mesh(r, backend='sparse', slab_size=5, **settings)
    assert_same_mesh(a, b)
    with pytest.raises(ValueError, match='backward pass'):
        extract_mesh(r, differentiable=True, **settings)
