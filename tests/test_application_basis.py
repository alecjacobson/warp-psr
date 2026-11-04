"""Exact uniform and one-octant gates for the production FEM basis in both precisions."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import warp as wp
import warp.fem as fem

from warp_psr.forms import (assemble_screening, assemble_stiffness, gradient,
                            point_quadrature, value)
from warp_psr.integration import integration_grid
from warp_psr.spaces import make_level_space
from warp_psr.stiffness import assemble_level_stiffness


ROOT = Path(__file__).resolve().parents[1]


def dense(matrix):
    return csr_matrix((matrix.values.numpy().ravel(), matrix.columns.numpy(),
                       matrix.offsets.numpy()), shape=(matrix.nrow, matrix.ncol)).toarray()


def tensor_stiffness(row_coords, col_coords, mass, derivative):
    result = np.zeros((len(row_coords), len(col_coords)))
    for axis in range(3):
        term = np.ones_like(result)
        for other in range(3):
            table = derivative if other == axis else mass
            term *= table[row_coords[:, other, None], col_coords[None, :, other]]
        result += term
    return result


@pytest.mark.cuda
@pytest.mark.reference
@pytest.mark.parametrize('double', [False, True])
@pytest.mark.parametrize('refinement', ['uniform', 'one_octant', 'grid3d'])
def test_application_basis_against_exact_reference(double, refinement):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    executable = ROOT/'.build/basis'
    if not executable.exists():
        pytest.skip('Build the pinned reference first')
    dtype, vector, real = (wp.float64, wp.vec3d, np.float64) if double else (wp.float32, wp.vec3, np.float32)
    atol, rtol = (1e-10, 1e-9) if double else (1e-6, 1e-5)
    mass, derivative = np.zeros((5, 5)), np.zeros((5, 5))
    cross_mass, cross_derivative = np.zeros((3, 5)), np.zeros((3, 5))
    for line in subprocess.check_output([str(executable), '2'], text=True).splitlines():
        tag, i, j, m, k = line.split()
        if tag == 'a':
            mass[int(i), int(j)], derivative[int(i), int(j)] = float(m), float(k)
        elif tag == 'c':
            cross_mass[int(i), int(j)], cross_derivative[int(i), int(j)] = float(m), float(k)
    coordinates = []
    for depth in range(3):
        size = 2 if depth == 2 and refinement == 'one_octant' else (1 << depth)+1
        coordinates.append(np.indices((size, size, size)).reshape(3, -1).T)
    with wp.ScopedDevice('cuda'):
        levels = []
        for depth, coords in enumerate(coordinates):
            x, y, z = coords.astype(np.int64).T
            levels.append(SimpleNamespace(keys=wp.array((depth << 48) | (x << 32) | (y << 16) | z,
                                                       dtype=wp.int64)))
        geometry = (fem.Grid3D(wp.vec3i(4), scalar_type=dtype) if refinement == 'grid3d'
                    else integration_grid(SimpleNamespace(basis=levels), None, 2, dtype=dtype))
        assert geometry.cell_count() == (15 if refinement == 'one_octant' else 64)
        coarse = make_level_space(geometry, levels[1].keys, 1, 2)
        fine = make_level_space(geometry, levels[2].keys, 2, 2)
        expected = tensor_stiffness(coordinates[2], coordinates[2], mass, derivative)
        actual = dense(assemble_stiffness(fine))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
        stencil = dense(assemble_level_stiffness(levels[2].keys, levels[2].keys, 2))
        np.testing.assert_allclose(stencil, expected, atol=atol, rtol=rtol)
        np.testing.assert_allclose(actual, actual.T, atol=atol, rtol=rtol)
        cross = dense(assemble_stiffness(fine, trial_space=coarse))
        expected = tensor_stiffness(coordinates[1], coordinates[2], cross_mass, cross_derivative).T
        np.testing.assert_allclose(cross, expected, atol=atol, rtol=rtol)
        stencil_cross = dense(assemble_level_stiffness(levels[2].keys, levels[1].keys, 2, 1))
        np.testing.assert_allclose(stencil_cross, expected, atol=atol, rtol=rtol)
        # Both sides of the hanging interface and near the cube boundary.
        epsilon = real(2.**(-32 if double else -16))
        positions = np.array([[.0001, .03125, .0625], [.27, .31, .43],
                              [.5-epsilon, .17, .21], [.5+epsilon, .17, .21],
                              [.83, .69, .9999]], dtype=real)
        weights = np.array([.31, 2.7, 4.1, .7, 1.3], dtype=real)
        points, measures = wp.array(positions, dtype=vector), wp.array(weights, dtype=dtype)
        quadrature = point_quadrature(fine, points, measures)
        dx = positions[:, None, :]*real(4)-coordinates[2][None, :, :]
        hats = np.maximum(1-np.abs(dx), 0)
        phi = np.prod(hats, axis=2)
        coarse_phi = np.prod(np.maximum(1-np.abs(positions[:, None, :]*real(2)
                                                 - coordinates[1][None, :, :]), 0), axis=2)
        screening = dense(assemble_screening(fine, points, measures, quadrature=quadrature))
        cross_screening = dense(assemble_screening(fine, points, measures,
                                                   trial_space=coarse, quadrature=quadrature))
        np.testing.assert_allclose(screening, phi.T@(weights[:, None]*phi), atol=atol, rtol=rtol)
        np.testing.assert_allclose(cross_screening, phi.T@(weights[:, None]*coarse_phi), atol=atol, rtol=rtol)
        coefficients = np.sin(np.arange(len(coordinates[2]))).astype(real)
        field = fine.make_field()
        field.dof_values = wp.array(coefficients, dtype=dtype)
        values, gradients = wp.empty(len(points), dtype=dtype), wp.empty(len(points), dtype=vector)
        fem.interpolate(value, dest=values, at=quadrature, fields={'u': field})
        fem.interpolate(gradient, dest=gradients, at=quadrature, fields={'u': field})
        expected_gradient = np.zeros((len(points), 3))
        for axis in range(3):
            partial = np.where(np.abs(dx[:, :, axis]) < 1, -4*np.sign(dx[:, :, axis]), 0)
            for other in range(3):
                if axis != other:
                    partial *= hats[:, :, other]
            expected_gradient[:, axis] = partial@coefficients
        np.testing.assert_allclose(values.numpy(), phi@coefficients, atol=atol, rtol=rtol)
        np.testing.assert_allclose(gradients.numpy(), expected_gradient, atol=atol, rtol=rtol)
