import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from tools.make_fixture import ROOT, points_for, read_output


@pytest.mark.reference
@pytest.mark.parametrize('precision', ['float', 'double'])
def test_binary_grid_matches_ordinary_query_evaluation(tmp_path, precision):
    if not (ROOT/'.build/oracle').exists():
        pytest.skip('Build the pinned reference first')
    dtype = np.float32 if precision == 'float' else np.float64
    p, n = points_for('sphere', 128, 2026)
    np.savetxt(tmp_path/'input.pts', np.column_stack([p, n]).astype(dtype), fmt='%.17g')
    low, high, resolution = -.9, .9, 5
    axis = low+(high-low)*(np.arange(resolution, dtype=np.float64)/(resolution-1))
    q = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    np.savetxt(tmp_path/'queries.txt', q, fmt='%.17g')
    (tmp_path/'spec.txt').write_text('5 -.9 -.9 -.9 .9 .9 .9\n')
    command = [str(ROOT/'.build/oracle'), str(tmp_path/'input.pts'), str(tmp_path/'queries.txt'),
               str(tmp_path/'output.txt'), '4', '2', '2', '0', '8', precision]
    env = {**os.environ, 'WARP_PSR_REFERENCE_PARALLELISM': 'none'}
    subprocess.run(command, env=env, check=True)
    before = read_output(tmp_path/'output.txt')
    env.update(WARP_PSR_GRID_SPEC=str(tmp_path/'spec.txt'), WARP_PSR_GRID_OUTPUT=str(tmp_path/'values.bin'))
    subprocess.run(command, env=env, check=True)
    after = read_output(tmp_path/'output.txt')
    for key in ('q', 'iso', 'transform'):
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(np.fromfile(tmp_path/'values.bin', dtype=dtype), before['q'][:, 0])
