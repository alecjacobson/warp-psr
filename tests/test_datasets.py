import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tools.datasets import load_input
from tools.benchmark_input import benchmark_input, reconstruction_parameters


def test_source_normals_and_reproducible_subsets(tmp_path):
    # Non-unit and zero normals must reach reconstruction unchanged.
    data = np.arange(600, dtype=np.float64).reshape(100, 6)
    data[3, 3:] = 0
    path = tmp_path/'points.npts'
    np.savetxt(path, data, fmt='%.17g')
    p, n, metadata = load_input(path)
    np.testing.assert_array_equal(np.column_stack([p, n]), data)
    assert metadata['original_count'] == metadata['selected_count'] == 100
    p, n, metadata = load_input(path, max_points=15, seed=42)
    indices = np.sort(np.random.default_rng(42).choice(100, 15, replace=False))
    np.testing.assert_array_equal(np.column_stack([p, n]), data[indices])
    assert metadata['selection_indices_sha256'] == hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
    with pytest.raises(ValueError, match='positive'):
        load_input(path, max_points=0)


def test_ply_attributes_and_required_normals(tmp_path):
    plyfile = pytest.importorskip('plyfile')
    dtype = [(name, 'f4') for name in ('x', 'y', 'z', 'nx', 'ny', 'nz')] + [('red', 'u1')]
    data = np.zeros(4, dtype=dtype)
    data['x'] = [1, 2, 3, 4]
    data['nz'] = [0, 2, -3, 1]
    path = tmp_path/'cloud.ply'
    plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(path)
    p, n, metadata = load_input(path)
    np.testing.assert_array_equal(p[:, 0], data['x'])
    np.testing.assert_array_equal(n[:, 2], data['nz'])
    assert metadata['ignored_properties'] == ['red']
    data = np.zeros(4, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(path)
    with pytest.raises(ValueError, match='nx'):
        load_input(path)


def test_benchmark_uses_fixture_parameters_and_rejects_changed_input(tmp_path):
    p, n, parameters, _ = benchmark_input(SimpleNamespace(fixture=None, count=10, depth=5, full_depth=2))
    path = tmp_path/'input.pts'
    np.savetxt(path, np.column_stack([p, n]), fmt='%.17g')
    parameters.update(weight=0., samples_per_node=3., base_depth=1, shape='test')
    manifest = {'name': 'test', 'parameters': parameters, 'precision': 'float',
                'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    # Contradictory synthetic defaults must not override fixture parameters.
    actual_p, actual_n, actual_parameters, _ = benchmark_input(SimpleNamespace(fixture=tmp_path, depth=9, full_depth=4))
    np.testing.assert_array_equal(actual_p, p)
    np.testing.assert_array_equal(actual_n, n)
    assert reconstruction_parameters(actual_parameters)['point_weight'] == 0.
    assert actual_parameters == parameters
    path.write_text(path.read_text()+'0 0 0 0 0 1\n')
    with pytest.raises(ValueError, match='checksum'):
        benchmark_input(SimpleNamespace(fixture=tmp_path))
