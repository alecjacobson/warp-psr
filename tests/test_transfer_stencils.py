"""Direct transfers preserve sorted rows, values, actions and transposes exactly."""
import numpy as np
import pytest
import warp as wp
import warp.sparse as sparse
from warp_psr.solver import prolongation


def keys(depth, selection):
    xyz = np.indices(((1 << depth) + 1,) * 3).reshape(3, -1).T
    xyz = xyz[selection(len(xyz))]
    return wp.array((depth << 48) | (xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2], dtype=wp.int64)


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('depth', [1, 2, 4])
@pytest.mark.parametrize('subset', [False, True])
def test_direct_transfers(dtype, depth, subset):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(104)
    selection = lambda n: rng.random(n) > .4 if subset else np.ones(n, dtype=bool)
    with wp.ScopedDevice('cuda'):
        coarse, fine = keys(depth - 1, selection), keys(depth, selection)
        expected = prolongation(coarse, fine, depth, dtype, direct=False)
        actual = prolongation(coarse, fine, depth, dtype)
        for a, b in [(actual, expected), (sparse.bsr_transposed(actual), sparse.bsr_transposed(expected))]:
            np.testing.assert_array_equal(a.offsets.numpy(), b.offsets.numpy())
            count = int(a.offsets.numpy()[-1])
            np.testing.assert_array_equal(a.columns.numpy()[:count], b.columns.numpy()[:count])
            np.testing.assert_array_equal(a.values.numpy()[:count], b.values.numpy()[:count])
            x = wp.array(rng.normal(size=a.ncol), dtype=dtype)
            left, right = wp.zeros(a.nrow, dtype=dtype), wp.zeros(a.nrow, dtype=dtype)
            sparse.bsr_mv(a, x, left)
            sparse.bsr_mv(b, x, right)
            np.testing.assert_array_equal(left.numpy(), right.numpy())


@pytest.mark.cuda
@pytest.mark.parametrize('empty_coarse,empty_fine', [(True, False), (False, True), (True, True)])
def test_empty_transfers(empty_coarse, empty_fine):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    with wp.ScopedDevice('cuda'):
        coarse = keys(1, lambda n: np.full(n, not empty_coarse))
        fine = keys(2, lambda n: np.full(n, not empty_fine))
        expected = prolongation(coarse, fine, 2, direct=False)
        actual = prolongation(coarse, fine, 2)
        np.testing.assert_array_equal(actual.offsets.numpy(), expected.offsets.numpy())
        np.testing.assert_array_equal(sparse.bsr_transposed(actual).offsets.numpy(),
                                      sparse.bsr_transposed(expected).offsets.numpy())
