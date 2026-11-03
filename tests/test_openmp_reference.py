import os
import subprocess
import numpy as np
import pytest
from tools.make_fixture import ROOT, points_for, read_output
from tools.check_repeatability import reference_keys
from tools.check_openmp import support


@pytest.mark.reference
@pytest.mark.parametrize('precision', ['float', 'double'])
def test_openmp_thread_limit_and_serial_topology(tmp_path, precision):
    if not (ROOT/'.build/oracle_instrumented_openmp').exists():
        pytest.skip('Build the pinned OpenMP reference first')
    p, n = points_for('sphere', 512, 2026)
    dtype = np.float64 if precision == 'double' else np.float32
    np.savetxt(tmp_path/'input.pts', np.column_stack([p, n]).astype(dtype), fmt='%.17g')
    queries = np.random.default_rng(2027).uniform(-.9, .9, (256, 3)).astype(dtype)
    np.savetxt(tmp_path/'queries.txt', queries, fmt='%.17g')
    results = []
    for mode, executable in [('none', 'oracle_instrumented'), ('openmp', 'oracle_instrumented_openmp')]:
        env = {**os.environ, 'WARP_PSR_REFERENCE_PARALLELISM': mode,
               'OMP_THREAD_LIMIT': '2', 'OMP_NUM_THREADS': '2', 'OMP_DYNAMIC': 'FALSE'}
        subprocess.run([str(ROOT/'.build'/executable), str(tmp_path/'input.pts'), str(tmp_path/'queries.txt'),
                        str(tmp_path/'output.txt'), '4', '2', '2', '0', '8', precision], env=env, check=True)
        results.append(read_output(tmp_path/'output.txt'))
    serial, parallel = results
    assert parallel['openmp_effective_threads'][0, 0] == 2
    np.testing.assert_array_equal(np.sort(reference_keys(serial['node'])), np.sort(reference_keys(parallel['node'])))
    np.testing.assert_array_equal(support(serial['node']), support(parallel['node']))
    np.testing.assert_array_equal(serial['transform'], parallel['transform'])
    atol, rtol = (1e-8, 1e-6) if precision == 'double' else (1e-5, 1e-4)
    for key in ('q', 'iso'):
        # Raw gradients at this small depth satisfy the unscaled budget as well.
        np.testing.assert_allclose(parallel[key], serial[key], atol=atol, rtol=rtol)
