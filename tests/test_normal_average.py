"""Cooperative memory loads must preserve every serial rounding decision."""
import numpy as np
import pytest
import warp as wp
from warp_psr.normals import precision_kernels


@pytest.mark.cuda
@pytest.mark.parametrize('double', [False, True])
@pytest.mark.parametrize('count', [1, 31, 129, 8193])
def test_tiled_average_is_bitwise_serial(double, count):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    dtype = wp.float64 if double else wp.float32
    vec3 = wp.vec3d if double else wp.vec3
    vec2 = wp.vec2d if double else wp.vec2
    rng = np.random.default_rng(82)
    with wp.ScopedDevice('cuda'):
        normals = rng.normal(size=(count, 3))
        normals[::7] = 0
        n = wp.array(normals, dtype=vec3)
        w = wp.array(rng.integers(1, 20, size=count), dtype=dtype)
        dw = wp.array(rng.uniform(-2, 8, size=(count, 2)), dtype=vec2)
        order = wp.array(rng.permutation(count), dtype=int)
        reference = wp.empty(1, dtype=vec3)
        actual = wp.empty_like(reference)
        contributions = wp.empty(count, dtype=vec3)
        kernels = precision_kernels(dtype)
        wp.launch(kernels._average, 1, inputs=[n, w, dw, order, reference])
        wp.launch(kernels._average_contributions, count, inputs=[n, w, dw, order, contributions])
        wp.launch_tiled(kernels._average_tiled, dim=1, inputs=[contributions, actual], block_dim=32)
        np.testing.assert_array_equal(actual.numpy(), reference.numpy())


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('count', [1, 31, 129, 8193])
def test_tiled_isovalue_is_bitwise_serial(dtype, count):
    from warp_psr.reconstruction import precision_kernels as reconstruction_kernels
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(28)
    with wp.ScopedDevice('cuda'):
        values = wp.array(rng.normal(size=count), dtype=dtype)
        weights = wp.array(rng.integers(1, 100, size=count), dtype=dtype)
        reference = wp.empty(1, dtype=dtype)
        actual = wp.empty_like(reference)
        kernels = reconstruction_kernels(dtype)
        wp.launch(kernels._isovalue, 1, inputs=[values, weights, reference])
        wp.launch_tiled(kernels._isovalue_tiled, dim=1, inputs=[values, weights, actual], block_dim=32)
        np.testing.assert_array_equal(actual.numpy(), reference.numpy())
