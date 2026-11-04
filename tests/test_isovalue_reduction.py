"""The parallel terminal mean must retain double accumulation accuracy."""
import math
import numpy as np
import pytest
import warp as wp
from warp_psr.reconstruction import _compute_isovalue


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('count', [1, 1023, 1024, 1025, 65537, 1048577])
def test_weighted_mean_matches_accurate_sum(dtype, count):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    scalar = np.float64 if dtype == wp.float64 else np.float32
    rng = np.random.default_rng(235)
    values = rng.normal(size=count).astype(scalar)
    weights = np.exp2(rng.uniform(-20, 20, size=count)).astype(scalar)
    weights[::13] = 0
    weights[0] = 1
    # Match the input-precision multiplication, then use an independent
    # compensated Python sum as the numerical reference for accumulation.
    products = values * weights
    expected = scalar(math.fsum(products) / math.fsum(weights))
    with wp.ScopedDevice('cuda'):
        v, w = wp.array(values, dtype=dtype), wp.array(weights, dtype=dtype)
        first = _compute_isovalue(v, w).numpy()[0]
        second = _compute_isovalue(v, w).numpy()[0]
    np.testing.assert_array_equal(first, second)
    if dtype == wp.float64:
        np.testing.assert_allclose(first, expected, atol=3e-15, rtol=3e-14)
    else:
        np.testing.assert_allclose(first, expected, atol=3e-8, rtol=1e-7)
