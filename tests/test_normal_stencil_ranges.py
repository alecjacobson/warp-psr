"""Sorted normal ranges retain bitwise serial stencil arithmetic at boundaries."""
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from warp_psr.normal_stencils import assemble_normal_stencil, load_kernel, _integrals


def keys(depth, xyz):
    xyz = xyz.astype(np.int64)
    return wp.array((depth << 48) | (xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2], dtype=wp.int64)


@pytest.mark.cuda
@pytest.mark.parametrize('vector', [wp.vec4, wp.vec4d])
@pytest.mark.parametrize('test_depth,normal_depth', [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (3, 2), (3, 3), (3, 4)])
@pytest.mark.parametrize('layout', ['empty', 'dense', 'sparse'])
def test_normal_ranges_match_individual_searches_bitwise(vector, test_depth, normal_depth, layout):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(601)
    xyz = np.indices(((1 << test_depth) + 1,) * 3).reshape(3, -1).T
    normal_xyz = np.indices((1 << normal_depth,) * 3).reshape(3, -1).T
    if layout == 'empty':
        normal_xyz = normal_xyz[:0]
    elif layout == 'sparse':
        normal_xyz = normal_xyz[rng.random(len(normal_xyz)) > .7]
    with wp.ScopedDevice('cuda'):
        row_keys = keys(test_depth, xyz)
        values = rng.normal(size=(len(normal_xyz), 4))
        values[::5] = 0
        records = SimpleNamespace(keys=keys(normal_depth, normal_xyz), values=wp.array(values, dtype=vector))
        actual = assemble_normal_stencil(row_keys, records, test_depth, normal_depth).numpy()
        width, first = (6, -3) if normal_depth > test_depth else (4, -2)
        table = wp.empty(((1 << test_depth) + 1, width), dtype=wp.vec2d)
        expected = wp.empty(len(xyz), dtype=wp.float64)
        wp.launch(_integrals, table.shape, inputs=[test_depth, normal_depth, first, table])
        wp.launch(load_kernel(vector), len(xyz), inputs=[row_keys, records.keys, records.values, test_depth, normal_depth, first, width, table, expected])
        np.testing.assert_array_equal(actual.view(np.uint64), expected.numpy().view(np.uint64))
