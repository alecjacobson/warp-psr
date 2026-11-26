"""Schedule an ABBA port comparison only after observing an idle shared machine."""
import argparse
import json
import os
from pathlib import Path
import sys

root = Path('/home/horde/projects/warp-psr')
destination = root/'.build/geometry-port'
sys.path.insert(0, str(root/'docs/results/depth9-audit-scripts'))
from isolation import measured_process

environment = dict(os.environ)
environment.update(PYTHONPATH='/home/horde/projects/warp-field-composition:'+str(root/'src')+':'+str(root),
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
parser = argparse.ArgumentParser()
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--gpu-lock', action='store_true')
args = parser.parse_args()
for index, variant in enumerate(('original', 'port', 'port', 'original'), start=args.offset):
    name = f'bench-{index}-{variant}'
    command = ['/home/horde/.local/bin/uv', 'run', '--no-project', '--python',
               '/home/horde/projects/warp-field-composition/.venv/bin/python',
               'python', str(destination/'benchmark.py'), '--variant', variant,
               '--output', str(destination/(name+'.json'))]
    if args.gpu_lock:
        command = ['bash', str(destination/'gpu_exclusive.sh'), *command]
    isolation = measured_process(command, destination/(name+'.log'), root=root, env=environment)
    isolation['gpu_lock'] = 'exclusive' if args.gpu_lock else 'not yet required at run launch'
    (destination/(name+'-isolation.json')).write_text(json.dumps(isolation, indent=2)+'\n')
    print('FINISHED', name, 'foreign work samples:', len(isolation['foreign_work_observed']), flush=True)
