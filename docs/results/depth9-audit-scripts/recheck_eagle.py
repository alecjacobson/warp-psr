from pathlib import Path
import json, os, statistics, subprocess
from isolation import measured_process
from tools.datasets import checksum
from tools.fixture_provenance import production_hashes

root=Path.cwd();out=root/'.build/depth9';python=Path('/home/horde/projects/warp-psr/.venv/bin/python')
baseline=Path('/home/horde/projects/warp-psr-stiffness')
env=dict(os.environ,PYTHONPATH='src:.')
fixture=out/'eagle-input'
assert checksum(fixture/'input.pts')==json.loads((fixture/'manifest.json').read_text())['input_sha256']
assert not subprocess.check_output(['git','diff','f36218a470d06a2524550c8e9825d99de103e27f','--','src'],cwd=baseline)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=baseline)
records=[]
for round_index in range(2):
    for index,label in enumerate(['baseline','current','current','baseline']):
        directory=baseline if label=='baseline' else root
        name=f'eagle-r{round_index}-{index}-{label}'
        target=out/(name+'.json')
        print('START',name,flush=True)
        isolation=measured_process([str(python),'-m','tools.benchmark','--fixture',str(fixture),
                                    '--repeats','7','--output',str(target)],out/(name+'.log'),root=directory,env=env)
        report=json.loads(target.read_text())
        records.append({'round':round_index,'label':label,'benchmark':report,'isolation':isolation})
        (out/'eagle-gpu-recheck.json').write_text(json.dumps({'complete':len(records)==8,'records':records},indent=2)+'\n')
        print('DONE',name,report['warm_median_seconds'],'foreign',isolation['foreign_work_observed'],flush=True)
print('START CPU4',flush=True)
target=out/'eagle-cpu4-recheck.json'
isolation=measured_process([str(python),str(out/'benchmark_cpu.py'),'--fixture',str(fixture),
                           '--repeats','7','--output',str(target)],out/'eagle-cpu4-recheck.log',root=root,env=env)
cpu=json.loads(target.read_text())
summary={label:{'pooled_warm_median_seconds':statistics.median(run['seconds'] for r in records if r['label']==label for run in r['benchmark']['runs'][1:]),
               'process_medians_seconds':[r['benchmark']['warm_median_seconds'] for r in records if r['label']==label]}
         for label in ['baseline','current']}
summary['current_over_baseline']=summary['current']['pooled_warm_median_seconds']/summary['baseline']['pooled_warm_median_seconds']
summary['speedup_over_fresh_cpu4']=cpu['warm_median_seconds']/summary['current']['pooled_warm_median_seconds']
summary['no_regression_over_5_percent']=summary['current_over_baseline']<=1.05
payload={'scope':'Two order-balanced ABBA rounds: prior published GPU production revision versus current main, seven warm samples after first invocation in each fresh process. Fresh four-thread CPU baseline measured separately on the identical input and parameters. Regression threshold fixed at +5% pooled warm median; all process medians retained.',
         'complete':True,'all_isolated':not isolation['foreign_work_observed'] and not any(r['isolation']['foreign_work_observed'] for r in records),
         'production_source_sha256':production_hashes(),'gpu_records':records,'cpu4':cpu,'cpu_isolation':isolation,'summary':summary}
(out/'eagle-recheck.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
