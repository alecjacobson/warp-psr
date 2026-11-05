"""Unused CSR capacity must never become active padded root triplets."""
import numpy as np
import pytest
import warp as wp
import warp.sparse as sparse
from warp_psr.solver import root_matrix_rows, padded_root_system
from warp_psr.stiffness import assemble_level_stiffness


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
def test_root_padding_excludes_unused_capacity(device, dtype):
    wp.init()
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    xyz = np.indices((2, 2, 2)).reshape(3, -1).T.astype(np.int64)
    keys = wp.array((xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2], dtype=wp.int64, device=device)
    matrix = sparse.bsr_copy(assemble_level_stiffness(keys, keys, 0), scalar_type=dtype)
    valid = int(matrix.offsets.numpy()[-1])
    assert matrix.nnz > valid, 'test requires unused capacity, without an nnz readback'
    values = matrix.values.numpy().copy()
    values[valid:] = 12345
    matrix.values.assign(values)
    # Pre-fill with valid-looking indices: leaving them untouched is a
    # deterministic regression, independent of allocator history.
    rows = wp.zeros(matrix.nnz, dtype=int, device=device)
    cols = wp.zeros_like(rows)
    wp.launch(root_matrix_rows, len(keys), inputs=[keys, matrix.offsets, matrix.columns, rows, cols], device=device)
    np.testing.assert_array_equal(rows.numpy()[valid:], -np.ones(matrix.nnz - valid))
    np.testing.assert_array_equal(cols.numpy()[valid:], -np.ones(matrix.nnz - valid))
    reference = sparse.bsr_from_triplets(64, 64, rows[:valid], cols[:valid], matrix.values[:valid])
    padded, rhs = padded_root_system(matrix, wp.ones(8, dtype=dtype, device=device), keys)
    vector = wp.array(np.random.default_rng(13).normal(size=64), dtype=dtype, device=device)
    expected, actual = wp.empty_like(vector), wp.empty_like(vector)
    sparse.bsr_mv(reference, vector, expected)
    sparse.bsr_mv(padded, vector, actual)
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert np.sum(rhs.numpy()) == 8
