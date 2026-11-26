"""Run one interleaved diagnostic under the machine-wide exclusive GPU lock."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('/home/horde/projects/warp-psr')
destination=root/'.build/geometry-port'
if '--locked' not in sys.argv:
    command=['bash',str(destination/'gpu_exclusive.sh'),'timeout','180',
             '/home/horde/.local/bin/uv','run','--no-project','--python',
             '/home/horde/projects/warp-field-composition/.venv/bin/python','python',str(Path(__file__).resolve()),'--locked']
    raise SystemExit(subprocess.run(command,timeout=310).returncode)
sys.path.insert(0,str(root/'docs/results/depth9-audit-scripts'))
from isolation import measured_process
environment=dict(os.environ)
environment.update(PYTHONPATH='/home/horde/projects/warp-field-composition:'+str(root/'src')+':'+str(root),
                   OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
command=['/home/horde/.local/bin/uv','run','--no-project','--python',
         '/home/horde/projects/warp-field-composition/.venv/bin/python','python',str(destination/'paired_benchmark.py')]
isolation=measured_process(command,destination/'paired.log',root=root,env=environment)
isolation['gpu_lock']='exclusive'
(destination/'paired-isolation.json').write_text(json.dumps(isolation,indent=2)+'\n')
print('Foreign work samples:',len(isolation['foreign_work_observed']))
