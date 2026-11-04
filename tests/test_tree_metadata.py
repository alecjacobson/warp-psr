"""Full retained node sets, flags and links agree with the pinned octree."""
from pathlib import Path
import json
import numpy as np
import pytest
import warp as wp
from tools.make_fixture import read_output
from tools.benchmark_input import reconstruction_parameters
from tools.check_geometry_profiles import GEOMETRY_CASES
from tools.check_profiles import ZERO_BASE_CASES
from warp_psr import reconstruct
from warp_psr.tree_metadata import build_tree_metadata

CASES = (['sphere', 'sparse', 'sphere-double', 'sparse-double',
          'sphere-exact', 'sphere-exact-double', 'sphere-psr', 'sphere-psr-double']
         + [f'geometry-{name}-{precision}' for name in GEOMETRY_CASES for precision in ('float', 'double')]
         + [f'profile-{name}-{precision}' for name in ZERO_BASE_CASES for precision in ('float', 'double')]
         + [f'threshold-{name}-{precision}' for name in ('density-above', 'density-below', 'ownership-psr', 'ownership-spsr') for precision in ('float', 'double')]
         + [f'{name}-10k-d7-spsr-float' for name in ('horse', 'bunny', 'eagle', 'torso')]
         + [f'topology-{name}-{precision}' for name in ('sphere', 'sparse') for precision in ('float', 'double')])


def decode(keys):
    return np.column_stack([((keys >> shift) & 1048575) - 524288 for shift in (40, 20, 0)])


@pytest.mark.cuda
@pytest.mark.reference
@pytest.mark.parametrize('name', CASES)
def test_complete_tree_metadata(name):
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    directory = Path('.build/fixtures')/name
    if not (directory/'oracle_instrumented.txt').exists():
        pytest.skip('Generate reference fixture first')
    manifest = json.loads((directory/'manifest.json').read_text())
    vector = wp.vec3d if manifest['precision'] == 'double' else wp.vec3
    data = np.loadtxt(directory/'input.pts')
    expected = read_output(directory/'oracle_instrumented.txt')['node']
    with wp.ScopedDevice('cuda'):
        result = reconstruct(wp.array(data[:, :3], dtype=vector), wp.array(data[:, 3:], dtype=vector),
                             **reconstruction_parameters(manifest['parameters']))
        diagnostic = result.diagnostics
        metadata = build_tree_metadata(diagnostic['tree'], diagnostic['samples'],
                                       diagnostic['density'], diagnostic['normals'])
        if 'topology.txt' in manifest['reference_file_sha256']:
            from tools.check_tree_metadata import check_tree_metadata
            check_tree_metadata(metadata, read_output(directory/'topology.txt')['topology'])
        coordinates = [decode(level.keys.numpy()) for level in metadata.levels]
        for g, level in enumerate(metadata.levels):
            d = level.local_depth
            rows = expected[expected[:, 0] == d]
            assert len(rows) == len(level.keys)
            if d >= 0:
                rows = rows[np.lexsort(rows[:, 1:4].T[::-1])]
                np.testing.assert_array_equal(coordinates[g] - (1 << (d + 1)), rows[:, 1:4])
            np.testing.assert_array_equal(level.flags.numpy(), rows[:, 4])
            np.testing.assert_array_equal(level.ghost.numpy(), rows[:, 5])
            parents, children = level.parents.numpy(), level.children.numpy()
            if g:
                assert np.all(parents >= 0)
                np.testing.assert_array_equal(coordinates[g - 1][parents], coordinates[g] // 2)
            else:
                np.testing.assert_array_equal(parents, -1)
            for k in range(8):
                present = children[:, k] >= 0
                if present.any():
                    np.testing.assert_array_equal(coordinates[g + 1][children[present, k]],
                        2 * coordinates[g][present] + [k & 1, (k >> 1) & 1, (k >> 2) & 1])
                    np.testing.assert_array_equal(metadata.levels[g + 1].parents.numpy()[children[present, k]],
                                                  np.flatnonzero(present))
            assert np.all((children >= 0).sum(axis=1) % 8 == 0)
