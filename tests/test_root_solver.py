"""Reference root mean term and its padded-row arithmetic."""

import numpy as np
import pytest
import warp as wp
import warp.sparse as sparse
from warp_psr.solver import (
    constant_mode_operator,
    regular_multigrid,
    padded_root_system,
)


@pytest.mark.parametrize("double", [False, True])
def test_root_constant_mode_and_repeated_cycles(double):
    wp.init()
    dtype = wp.float64 if double else wp.float32
    numpy_dtype = np.float64 if double else np.float32
    # This exact rank-seven Laplacian has the same constant nullspace as
    # unscreened Neumann stiffness. Adding mean(x) makes the operator identity.
    matrix = np.eye(8, dtype=numpy_dtype) - numpy_dtype(1 / 8)
    rows, cols = np.indices(matrix.shape)
    devices = ["cpu"] + (["cuda"] if wp.is_cuda_available() else [])
    for device in devices:
        with wp.ScopedDevice(device):
            A = sparse.bsr_from_triplets(
                8,
                8,
                wp.array(rows.ravel(), dtype=int),
                wp.array(cols.ravel(), dtype=int),
                wp.array(matrix.ravel(), dtype=dtype),
            )
            vector = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=numpy_dtype)
            x = wp.array(vector, dtype=dtype)
            y = wp.ones(8, dtype=dtype)
            z = wp.empty_like(y)
            constant_mode_operator(A).matvec(x, y, z, 2.0, 3.0)
            np.testing.assert_array_equal(z.numpy(), 2 * vector + 3)
            keys = wp.array(
                [(i // 4 << 32) | ((i // 2 % 2) << 16) | (i % 2) for i in range(8)],
                dtype=wp.int64,
            )
            padded, rhs = padded_root_system(A, y, keys)
            output = wp.empty_like(rhs)
            constant_mode_operator(padded).matvec(rhs, rhs, output, 1.0, 0.0)
            np.testing.assert_array_equal(output.numpy(), np.full(64, 1 / 8))
            zero_mean = vector - vector.mean()
            x.assign(zero_mean)
            solution, _ = regular_multigrid(
                A, x, [keys], [], 8, 3, 1e-6, add_dc_term=True
            )
            np.testing.assert_allclose(
                solution.numpy(), zero_mean, atol=1e-6 if not double else 1e-12, rtol=0
            )
