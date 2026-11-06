"""Reproducible depth-9 matrix on the exact eight depth-8 point sets."""
from pathlib import Path
import copy, datetime, hashlib, json, os, shutil, subprocess, sys, tarfile, tempfile
from isolation import measured_process
from tools.benchmark_input import reconstruction_parameters
from tools.datasets import checksum
from tools.fixture_provenance import production_hashes, require_acceptance

root=Path.cwd();shared=Path('/home/horde/projects/warp-psr');out=root/'.build/depth9'
python=shared/'.venv/bin/python';env=dict(os.environ,PYTHONPATH='src:.')
inputs=out/'inputs';inputs.mkdir(exist_ok=True)
reports=out/'matrix';reports.mkdir(exist_ok=True)
old=json.loads((root/'docs/results/public-scaling.json').read_text())
added=json.loads((root/'docs/results/expanded-datasets.json').read_text())
sources={r['fixture']:r for r in old['profiles']+added['profiles']}
rows=[('eagle','full'),('drill','full'),('horse','full'),('bunny','full'),('torso','full'),
      ('armadillo','100k'),('dragon','100k'),('buddha','100k')]
frozen_sources=production_hashes();records=[]
summary_path=out/'matrix-summary.json'
if summary_path.exists():
    saved=json.loads(summary_path.read_text())
    assert saved['production_source_sha256']==frozen_sources
    assert saved['all_passed'] and saved['all_isolated']
    records=saved['profiles']
completed={r['fixture']:r for r in records}
recovery_path=out/'active-case.json'
resume=json.loads(recovery_path.read_text()) if recovery_path.exists() else None

def source_input(record):
    manifest=record['manifest'];digest=manifest['input_sha256']
    if 'archive' not in record:
        path=shared/'.build/archives/gpu-pipeline-b6c6988/reference-benchmark'/(digest[:16]+'-8-none')/'input.pts'
        assert checksum(path)==digest
        return path, None
    archive=Path('/home/horde/projects/warp-psr-cases')/record['archive']['path']
    assert checksum(archive)==record['archive']['sha256']
    path=inputs/(digest+'.pts')
    if not path.exists():
        with tarfile.open(archive,'r|gz') as stream:
            for member in stream:
                if member.name==record['fixture']+'/input.pts':
                    with stream.extractfile(member) as src,path.open('wb') as dest:
                        shutil.copyfileobj(src,dest)
                    break
            else:raise RuntimeError('Input missing from verified source archive')
    assert checksum(path)==digest
    return path,record['archive']['sha256']

def invoke(command, log_path):
    with log_path.open('w') as log:
        status=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if status:raise RuntimeError(f'Exit {status}: inspect {log_path}')

for dataset,size in rows:
    for method in (['spsr','psr'] if dataset=='eagle' else ['psr','spsr']):
        previous=f'{dataset}-{size}-d8-{method}'
        original=sources[previous]
        name=f'{dataset}-{size}-d9-{method}'
        if name in completed:
            assert completed[name]['source_input_sha256']==original['manifest']['input_sha256']
            print('RETAIN COMPLETED',name,flush=True)
            continue
        persistent=reports/name;persistent.mkdir(exist_ok=True)
        source,archive_digest=source_input(original)
        # Only disposable, deterministically regenerable fixture exports go here.
        # Inputs, code, Git data, measurements and acceptance reports are persistent.
        resuming=resume is not None and resume['case']==name and Path(resume['scratch']).exists()
        if resuming:
            assert resume['source_input']==str(source) and resume['persistent_results']==str(persistent)
            scratch=Path(resume['scratch'])
            assert scratch.parent==Path('/opt/warp-psr-depth9-scratch') and scratch.name.startswith('case-')
        else:
            scratch=Path(tempfile.mkdtemp(prefix='case-',dir='/opt/warp-psr-depth9-scratch'))
        fixture=scratch/'fixtures'/name
        recovery={'case':name,'source_input':str(source),'scratch':str(scratch),
                  'persistent_results':str(persistent),'state':'generating'}
        (out/'active-case.json').write_text(json.dumps(recovery,indent=2)+'\n')
        print('RESUME VALIDATED' if resuming else 'GENERATE',name,str(scratch),flush=True)
        generate=[str(python),'tools/make_fixture.py','--name',name,'--input',str(source),
                  '--depth','9','--full-depth','4','--base-depth','4','--solve-depth','9','--kernel-depth','7',
                  '--weight','0' if method=='psr' else '2','--grid','24','--precision','float',
                  '--output-directory',str(fixture)]
        if not resuming:
            invoke(generate,persistent/'generation.log')
        manifest=json.loads((fixture/'manifest.json').read_text())
        assert manifest['input_sha256']==original['manifest']['input_sha256']
        assert manifest['parameters']['count']==original['manifest']['parameters']['count']
        expected=reconstruction_parameters(original['manifest']['parameters'])
        expected.update(depth=9,solve_depth=9,kernel_depth=7)
        assert reconstruction_parameters(manifest['parameters'])==expected
        (persistent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('VALIDATE',name,flush=True)
        if not resuming:
            invoke([str(python),'-m','tools.check_reconstruction',str(fixture)],persistent/'gpu-validation.log')
        acceptance=require_acceptance(fixture)
        (persistent/'reconstruction_report.json').write_text(json.dumps(acceptance,indent=2)+'\n')
        print('OPENMP CHECK',name,flush=True)
        if not resuming:
            invoke([str(python),'-m','tools.check_openmp','--fixtures',name,'--threads','4','--repeats','1',
                    '--fixture-root',str(fixture.parent),'--work-directory',str(scratch/'openmp'),
                    '--output',str(persistent/'openmp-validation.json')],persistent/'openmp-validation.log')
        openmp=json.loads((persistent/'openmp-validation.json').read_text())
        assert openmp['all_passed']
        assert openmp['fixtures'][0]['input_sha256']==manifest['input_sha256']
        assert reconstruction_parameters(openmp['fixtures'][0]['parameters'])==expected
        entry={'fixture':name,'dataset':dataset,'subset':size=='100k','source_fixture':previous,
               'source_input_sha256':original['manifest']['input_sha256'],
               'source_metadata':original['manifest']['input_source'],'source_archive_sha256':archive_digest,
               'manifest':manifest,'reconstruction':acceptance,'openmp_validation':openmp,
               'benchmarks':{},'isolation':{}}
        if name=='eagle-full-d9-spsr':
            recheck=json.loads((out/'eagle-recheck.json').read_text())
            assert recheck['complete'] and recheck['all_isolated']
            assert recheck['production_source_sha256']==frozen_sources
            current=[r for r in recheck['gpu_records'] if r['label']=='current']
            for r in current:
                assert r['benchmark']['input_sha256']==manifest['input_sha256']
                assert reconstruction_parameters(r['benchmark']['parameters'])==expected
            gpu=copy.deepcopy(current[0]['benchmark'])
            gpu['warm_median_seconds']=recheck['summary']['current']['pooled_warm_median_seconds']
            gpu['measurement_policy']='Preplanned pooled median of all28warm samples from four current-main processes in the two ABBA rounds; all process medians and samples are in eagle-recheck.json.'
            gpu['all_process_reports']=[r['benchmark'] for r in current]
            cpu=recheck['cpu4']
            assert cpu['input_sha256']==manifest['input_sha256'] and reconstruction_parameters(cpu['parameters'])==expected
            entry['benchmarks']={'gpu':gpu,'openmp':cpu}
            entry['isolation']={'gpu':[r['isolation'] for r in current],'openmp':recheck['cpu_isolation']}
            entry['all_isolated']=True
        else:
            print('BENCHMARK',name,flush=True)
            for kind,command in [('gpu',[str(python),'-m','tools.benchmark']),
                                  ('openmp',[str(python),str(out/'benchmark_cpu.py')])]:
                target=persistent/(kind+'.json')
                isolation=measured_process([*command,'--fixture',str(fixture),'--repeats','5','--output',str(target)],
                                           persistent/(kind+'.log'),root=root,env=env)
                result=json.loads(target.read_text())
                assert result['input_sha256']==manifest['input_sha256']
                assert reconstruction_parameters(result['parameters'])==expected
                entry['benchmarks'][kind]=result;entry['isolation'][kind]=isolation
            entry['all_isolated']=not any(r['foreign_work_observed'] for r in entry['isolation'].values())
        assert entry['all_isolated'], f'Contention observed; preserve fixture and all attempts for {name}'
        entry['speedup_over_openmp4']=entry['benchmarks']['openmp']['warm_median_seconds']/entry['benchmarks']['gpu']['warm_median_seconds']
        # Capture hashes for every disposable reference export and numerical result
        # before deleting scratch. All inputs and regeneration settings are retained.
        inventory={str(p.relative_to(scratch)):{'sha256':checksum(p),'bytes':p.stat().st_size}
                   for p in scratch.rglob('*') if p.is_file()}
        entry['scratch_export_inventory']=inventory
        entry['raw_export_policy']='Reproducible scratch exports are not retained; exact source input, parameter manifest, query/reference/output hashes, full acceptance reports and every timing sample are retained on persistent storage.'
        assert production_hashes()==frozen_sources
        (persistent/'result.json').write_text(json.dumps(entry,indent=2)+'\n')
        records.append(entry)
        summary={'scope':'Fresh float32 depth9 PSR/SPSR comparisons on exactly the point sets from the depth8 eight-dataset table. Full/base depth4, kernel depth7, same remaining numerical parameters. Fresh C++ serial/GPU and four-thread OpenMP correctness gates precede isolated timings; eagle timing reused from the preplanned same-source recheck.',
                 'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                 'production_source_sha256':frozen_sources,'complete':len(records)==16,
                 'all_passed':all(r['reconstruction']['reconstruction_compatibility'] and r['openmp_validation']['all_passed'] for r in records),
                 'all_isolated':all(r['all_isolated'] for r in records),'profiles':records,
                 'runner_sha256':checksum(__file__),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (out/'matrix-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        assert scratch.parent==Path('/opt/warp-psr-depth9-scratch') and scratch.name.startswith('case-')
        shutil.rmtree(scratch)
        recovery.update(state='complete',scratch_removed=True)
        (out/'active-case.json').write_text(json.dumps(recovery,indent=2)+'\n')
        print('PASS',name,'GPU',entry['benchmarks']['gpu']['warm_median_seconds'],
              'CPU4',entry['benchmarks']['openmp']['warm_median_seconds'],
              'speedup',entry['speedup_over_openmp4'],'isolated',entry['all_isolated'],flush=True)
assert len(records)==16
print('COMPLETE',flush=True)
