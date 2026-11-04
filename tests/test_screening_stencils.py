"""Direct point forms must match the independently assembled public FEM route."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix
import warp as wp
import warp.sparse as sparse
from warp_psr.forms import assemble_screening, assemble_point_load, point_quadrature
from warp_psr.integration import level_integration_grid
from warp_psr.spaces import make_level_space
from warp_psr.stiffness import assemble_level_stiffness
from warp_psr.screening_stencils import add_screening_stencil


def keys(depth, xyz):
    xyz = xyz.astype(np.int64)
    return wp.array((depth << 48) | (xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2], dtype=wp.int64)


def csr(matrix):
    return csr_matrix((matrix.values.numpy(), matrix.columns.numpy(), matrix.offsets.numpy()),
                      shape=(matrix.nrow, matrix.ncol))


@pytest.mark.cuda
@pytest.mark.parametrize('double', [False, True])
@pytest.mark.parametrize('depth', [2, 4])
def test_clustered_point_matrix_and_load_against_fem(double, depth):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    dtype, vector, real = (wp.float64, wp.vec3d, np.float64) if double else (wp.float32, wp.vec3, np.float32)
    atol, rtol = (1e-10, 1e-9) if double else (1e-6, 1e-5)
    rng = np.random.default_rng(54)
    n = 1 << depth
    with wp.ScopedDevice('cuda'):
        row_keys = keys(depth, np.indices((n + 1,) * 3).reshape(3, -1).T)
        tree = SimpleNamespace(basis=[None] * depth + [SimpleNamespace(keys=row_keys)])
        geometry = level_integration_grid(tree, depth, dtype)
        test = make_level_space(geometry, row_keys, depth, depth)
        cells = np.indices((n,) * 3).reshape(3, -1).T
        cells = cells[rng.random(len(cells)) > .2]
        fraction = rng.uniform(.01, .99, (len(cells), 3)).astype(real)
        fraction[::11] = 0
        fraction[::17] = 1
        # Normalized constraints are interior to the reconstruction cube.
        # Exercise exact internal knots while respecting that precondition.
        for axis in range(3):
            fraction[(cells[:, axis] == 0) & (fraction[:, axis] == 0), axis] = .01
            fraction[(cells[:, axis] == n - 1) & (fraction[:, axis] == 1), axis] = .99
        positions = (cells.astype(real) + fraction) / real(n)
        weights = rng.uniform(.1, 4, len(cells)).astype(real)
        weights[::7] = 0
        # Multiple aggregates can land in the same cell after rounding.
        positions = np.concatenate([positions, positions[:3]])
        weights = np.r_[weights, weights[:3] * real(.7)]
        points = SimpleNamespace(positions=wp.array(positions, dtype=vector), weights=wp.array(weights, dtype=dtype))
        quadrature = point_quadrature(test, points.positions, points.weights)
        expected_load = assemble_point_load(test, points.positions, points.weights, quadrature=quadrature).numpy()
        for trial_depth in (depth, depth - 1):
            size = (1 << trial_depth) + 1
            col_keys = keys(trial_depth, np.indices((size,) * 3).reshape(3, -1).T)
            trial = make_level_space(geometry, col_keys, trial_depth, depth)
            reference = assemble_screening(test, points.positions, points.weights,
                trial_space=trial, accumulation_dtype=wp.float64, quadrature=quadrature)
            actual = assemble_level_stiffness(row_keys, col_keys, depth, trial_depth)
            sparse.bsr_scale(actual, 0)
            load = add_screening_stencil(actual, row_keys, col_keys, points, depth, trial_depth)
            x = rng.normal(size=(len(col_keys), 4))
            expected = csr(reference) @ x
            scale = np.max(np.abs(expected)) or 1
            np.testing.assert_allclose(csr(actual) @ x / scale, expected / scale, atol=atol, rtol=rtol)
            scale = np.max(np.abs(expected_load)) or 1
            np.testing.assert_allclose(load.numpy() / scale, expected_load / scale, atol=atol, rtol=rtol)
