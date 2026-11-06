"""Collect the first five isolated fresh-process CPU solves after one warmup."""
from pathlib import Path
import copy
import json
import os
import statistics

from isolation import measured_process
from tools.benchmark_input import reconstruction_parameters
from tools.datasets import checksum

root = Path.cwd()
out = root / '.build/depth9'
case = out / 'matrix/torso-full-d9-spsr'
recovery = json.loads((case / 'recovery.json').read_text())
fixture = Path(recovery['scratch']) / 'fixtures' / recovery['case']
manifest = json.loads((fixture / 'manifest.json').read_text())
python = '/home/horde/projects/warp-psr/.venv/bin/python'
env = dict(os.environ, PYTHONPATH='src:.')
directory = case / 'cpu-processes'
directory.mkdir(exist_ok=True)
accepted, rejected = [], []
for index in range(6):
    attempt = 0
    while True:
        attempt += 1
        stem = f'run-{index}-attempt-{attempt}'
        target = directory / (stem + '.json')
        assert not target.exists(), 'Preserve prior attempts; explicitly resume their saved state'
        audit = measured_process([python, str(out / 'benchmark_cpu_single.py'),
                                  '--fixture', str(fixture), '--repeats', '0', '--output', str(target)],
                                 directory / (stem + '.log'), root=root, env=env)
        (directory / (stem + '-isolation.json')).write_text(json.dumps(audit, indent=2) + '\n')
        report = json.loads(target.read_text())
        assert report['input_sha256'] == manifest['input_sha256']
        assert reconstruction_parameters(report['parameters']) == reconstruction_parameters(manifest['parameters'])
        assert report['openmp_effective_threads'] == 4 and len(report['runs']) == 1
        record = {'index': index, 'attempt': attempt, 'benchmark': report, 'isolation': audit}
        if audit['foreign_work_observed']:
            rejected.append(record)
            print('REJECT CPU PROCESS', index, attempt, flush=True)
        else:
            accepted.append(record)
            print('ACCEPT CPU PROCESS', index, attempt, report['runs'][0]['seconds'], flush=True)
        (directory / 'progress.json').write_text(json.dumps({'accepted': accepted, 'rejected': rejected}, indent=2) + '\n')
        if not audit['foreign_work_observed']:
            break
report = copy.deepcopy(accepted[0]['benchmark'])
report['runs'] = [{'iteration': i, 'seconds': r['benchmark']['runs'][0]['seconds']}
                  for i, r in enumerate(accepted)]
report['warm_median_seconds'] = statistics.median(r['seconds'] for r in report['runs'][1:])
report['execution_environment_after'] = accepted[-1]['benchmark']['execution_environment_after']
report['benchmark_script_sha256'] = checksum(__file__)
report['single_process_script_sha256'] = checksum(out / 'benchmark_cpu_single.py')
report['measurement_policy'] = ('Same unchanged Solver::Solve timer and six fresh CPU processes as other rows: '
    'discard the first accepted invocation and report the median of the first five subsequent accepted invocations. '
    'Each process separately waits for idle and passes an activity audit. Rejected process attempts depend only '
    'on observed contention, never timing values; all are retained. Used after eight whole batches were contended.')
report['all_process_reports'] = [r['benchmark'] for r in accepted]
report['rejected_process_attempts'] = rejected
audit = {'gpu_before': [], 'foreign_work_observed': [],
         'accepted_process_audits': [r['isolation'] for r in accepted],
         'scope': 'Every accepted fresh CPU process independently passes the same idle/activity checks. '
                  'Six accepted invocations, first discarded; waiting between processes is excluded from Solver::Solve.'}
(case / 'openmp.json').write_text(json.dumps(report, indent=2) + '\n')
(case / 'openmp-isolation.json').write_text(json.dumps(audit, indent=2) + '\n')
print('CPU COMPLETE', report['warm_median_seconds'], flush=True)
