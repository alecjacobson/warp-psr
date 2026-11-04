"""Cancellation is valid input: the pinned reference returns a zero field."""

import math

import numpy as np
import pytest
import warp as wp

from warp_psr import reconstruct


@pytest.mark.cuda
@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("weight,exact", [(0.0, False), (2.0, False), (2.0, True)],
                         ids=["psr", "spsr", "exact"])
@pytest.mark.parametrize("partial", [False, True])
def test_cancelled_aggregate_normals(double, weight, exact, partial):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    dtype = wp.vec3d if double else wp.vec3
    points = [[-1, -1, -1], [-1, -1, -1], [1, 1, 1], [1, 1, 1]]
    normals = [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]]
    if partial:
        points.append([0, 0.1, 0])
        normals.append([0, 0, 1])
    p = wp.array(points, dtype=dtype, device="cuda")
    n = wp.array(normals, dtype=dtype, device="cuda")
    result = reconstruct(p, n, depth=2, full_depth=1, point_weight=weight,
                         exact_interpolation=exact)
    queries = wp.array(points + [[0, 0, 0], [-0.5, 0.25, 0.5]], dtype=dtype,
                       device="cuda")
    values, gradients, valid = result.evaluate(queries)
    average = result.diagnostics["normals"].average.numpy()
    assert math.isfinite(result.iso_value)
    assert valid.numpy().all()
    assert np.isfinite(values.numpy()).all()
    assert np.isfinite(gradients.numpy()).all()
    assert np.isfinite(average).all()
    if partial:
        assert len(result.diagnostics["normals"].records.keys) > 0
        assert average[0, 2] == 1
    else:
        assert len(result.diagnostics["normals"].records.keys) == 0
        np.testing.assert_array_equal(average, 0)
        np.testing.assert_array_equal(values.numpy(), 0)
        np.testing.assert_array_equal(gradients.numpy(), 0)
        assert result.iso_value == 0
