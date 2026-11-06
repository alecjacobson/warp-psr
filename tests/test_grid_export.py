"""Read-only oracle exports, including regular grids and solver traces."""
import hashlib
import json
from tools.check_solver_trace import check
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from tools.make_fixture import ROOT, points_for, read_output


@pytest.mark.reference
def test_fixture_and_openmp_checks_support_explicit_scratch_directories(tmp_path, monkeypatch):
    from argparse import Namespace
    from tools.make_fixture import generate
    from tools.fixture_provenance import verify_fixture
    from tools.check_openmp import main as check_openmp
    if not all((ROOT/'.build'/name).exists() for name in ('oracle', 'oracle_instrumented', 'oracle_instrumented_openmp')):
        pytest.skip('Build the pinned reference first')
    check_output = subprocess.check_output
    def without_nvidia_smi(command, *args, **kwargs):
        if command[0] == 'nvidia-smi':
            raise FileNotFoundError('No GPU inventory utility on this host')
        return check_output(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'check_output', without_nvidia_smi)
    args = Namespace(name='sphere', dataset=None, input=None, max_points=None, shape='sphere',
                     input_scale=1., translation=[0.,0.,0.], count=64, seed=2026, variant='none',
                     depth=3, full_depth=2, base_depth=None, solve_depth=None, kernel_depth=None,
                     scale=1.1, samples_per_node=1.5, cg_accuracy=1e-3, base_v_cycles=1,
                     weight=2., exact=False, iters=8, precision='float', grid=3,
                     operators=False, topology=False, solver_trace=False)
    directory = tmp_path/'fixtures'/'sphere'
    generate(args, output_directory=directory)
    identity = verify_fixture(directory)
    assert json.loads((directory/'manifest.json').read_text())['environment']['gpu'] is None
    assert identity['parameters']['base_depth'] == 2
    assert 'output_directory' not in identity['parameters']
    report = tmp_path/'openmp.json'
    assert check_openmp(Namespace(fixtures=['sphere'], threads=2, repeats=1,
                                 fixture_root=directory.parent, work_directory=tmp_path/'reference-work',
                                 output=report)) == 0
    assert (tmp_path/'reference-work'/'sphere'/'run-0.txt').is_file()
    assert json.loads(report.read_text())['all_passed']


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


@pytest.mark.reference
@pytest.mark.parametrize('precision',['float','double'])
@pytest.mark.parametrize('accuracy',['loose','default','tight'])
def test_fresh_cpu_trace_neutrality_and_replay(tmp_path,precision,accuracy):
    if not all((ROOT/'.build'/name).exists() for name in ['oracle','oracle_instrumented']):
        pytest.skip('Build pinned reference executables')
    scalar=np.float64 if precision=='double' else np.float32
    p,n=points_for('sphere',128,2026)
    np.savetxt(tmp_path/'input.pts',np.column_stack([p,n]).astype(scalar),fmt='%.17g')
    np.savetxt(tmp_path/'queries.txt',p[:16]*.9,fmt='%.17g')
    env={k:v for k,v in os.environ.items() if not k.startswith('WARP_PSR_')}
    env['WARP_PSR_REFERENCE_PARALLELISM']='none'
    for name in ['oracle','oracle_instrumented']:
        if name=='oracle_instrumented':env['WARP_PSR_SOLVE_TRACE']=str(tmp_path/'solve_trace.txt')
        subprocess.run([str(ROOT/'.build'/name),str(tmp_path/'input.pts'),str(tmp_path/'queries.txt'),
                        str(tmp_path/(name+'.txt')),'3','2','2','0','3',precision,
                        '2','3','1','1.1','1.5',{'loose':'1e-1','default':'1e-3','tight':'1e-8'}[accuracy],'1'],
                       env=env,check=True,capture_output=True)
    manifest={'precision':precision,'reference_parallelism':'ThreadPool::NONE (serial)',
              'reference_file_sha256':{'solve_trace.txt':hashlib.sha256((tmp_path/'solve_trace.txt').read_bytes()).hexdigest()}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    report=check(tmp_path)
    assert report['gs_updates']>0
    if accuracy=='tight':assert any(s['iterations'] for s in report['cg_stops'])
    if accuracy=='loose':assert all(s['reason']=='initial' for s in report['cg_stops'])
