"""Measure GPU repeatability and serial/default-async reference variability."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import warp as wp
from warp_psr import reconstruct
from reference.cells import query_cell_widths
from reference.adapter import load_field
from tools.make_fixture import ROOT, oracle_parameters, read_output


def digest_keys(keys):
    return hashlib.sha256(np.sort(np.asarray(keys,dtype='<i8')).tobytes()).hexdigest()


def reference_keys(nodes):
    active=nodes[(nodes[:,0]>=0)&((nodes[:,4].astype(np.int64)&2)!=0),:4].astype(np.int64)
    return (active[:,0]<<48)|(active[:,1]<<32)|(active[:,2]<<16)|active[:,3]


def errors(actual,iso,reference,reference_iso,inverse_linear,widths,atol,rtol):
    if (actual.shape!=reference.shape or not np.isfinite(actual).all()
            or not np.isfinite(reference).all() or not np.isfinite([iso,reference_iso]).all()):
        raise ValueError('Repeatability fields must be finite arrays with matching shapes')
    delta=actual[:,0]-reference[:,0]
    centered=(actual[:,0]-iso)-(reference[:,0]-reference_iso)
    gradient=actual[:,1:]@inverse_linear
    ref_gradient=reference[:,1:]@inverse_linear
    scaled=np.abs(gradient-ref_gradient)*widths[:,None]
    ratios={
        'value':float(np.max(np.abs(delta)/(atol+rtol*np.abs(reference[:,0])))),
        'centered_value':float(np.max(np.abs(centered)/(atol+rtol*np.abs(reference[:,0]-reference_iso)))),
        'scaled_gradient':float(np.max(scaled/(atol+rtol*np.abs(ref_gradient)*widths[:,None]))),
        'isovalue':float(abs(iso-reference_iso)/(atol+rtol*abs(reference_iso))),
    }
    return {'bitwise_equal':bool(np.array_equal(actual,reference) and iso==reference_iso),
            'within_budget':max(ratios.values())<=1.,'budget_ratios':ratios,
            'max_abs_value_error':float(np.max(np.abs(delta))),
            'rms_value_error':float(np.sqrt(np.mean(delta**2))),
            'max_abs_centered_value_error':float(np.max(np.abs(centered))),
            'max_abs_cube_gradient_error':float(np.max(np.abs(gradient-ref_gradient))),
            'isovalue_error':float(abs(iso-reference_iso))}


def reference_run(directory,parameters,output,parallelism,instrumented=True):
    output.mkdir(parents=True,exist_ok=True)
    exe='oracle_instrumented' if instrumented else 'oracle'
    environment={k:v for k,v in os.environ.items() if not k.startswith('WARP_PSR_')}
    environment.update(WARP_PSR_REFERENCE_PARALLELISM=parallelism,
                       WARP_PSR_ORACLE_DIR=str(output),
                       WARP_PSR_TIMING_FILE=str(output/'solve-seconds.txt'))
    command=[str(ROOT/'.build'/exe),str(directory/'input.pts'),str(directory/'queries.txt'),
             str(output/'oracle.txt'),*oracle_parameters(parameters)]
    with (output/'process.log').open('w') as log:
        subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    data=read_output(output/'oracle.txt')
    metadata={'solve_seconds':float((output/'solve-seconds.txt').read_text()),
              'parallel_async':bool(data['parallel_async'][0,0]),
              'configured_threads':int(data['threads'][0,0]),
              'schedule':int(data['schedule'][0,0]),'chunk_size':int(data['chunk_size'][0,0])}
    return data,metadata


def check_fixture(name,repeats):
    directory=(ROOT/'.build/fixtures'/name).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    parameters=manifest['parameters']
    for filename,key in [('input.pts','input_sha256'),('queries.txt','queries_sha256')]:
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=manifest[key]:
            raise ValueError(f'{name}: fixture checksum mismatch for {filename}')
    destination=(ROOT/'.build/repeatability'/name).resolve()
    baseline,first=reference_run(directory,parameters,destination/'serial-0','none')
    original,_=reference_run(directory,parameters,destination/'original','none',instrumented=False)
    for key in ('q','iso','transform'):
        np.testing.assert_array_equal(baseline[key],original[key])
    key_hash=digest_keys(reference_keys(baseline['node']))
    double=manifest['precision']=='double'
    numpy_dtype=np.float64 if double else np.float32
    vector=wp.vec3d if double else wp.vec3
    atol,rtol=(1e-8,1e-6) if double else (1e-5,1e-4)
    queries=np.loadtxt(directory/'queries.txt').astype(numpy_dtype)
    field=load_field(destination/'serial-0/oracle.txt',dtype=wp.float64 if double else wp.float32)
    linear=np.array(field.linear).reshape(3,3)
    cube=queries@linear.T+field.translation
    normal=read_output(destination/'serial-0/stages.txt')['normal']
    widths=query_cell_widths(baseline['node'],normal,cube,parameters['depth'])
    inverse_linear=np.linalg.inv(linear)
    del field
    def compare(data,reference=baseline):
        return errors(data['q'],float(data['iso'][0,0]),reference['q'],float(reference['iso'][0,0]),inverse_linear,widths,atol,rtol)
    reference_runs={}
    for mode in ('none','async'):
        runs=[]
        first_data=None
        for index in range(repeats):
            if mode=='none' and index==0:data,metadata=baseline,first
            else:data,metadata=reference_run(directory,parameters,destination/f'{mode}-{index}',mode)
            if first_data is None:first_data=data
            runs.append({**metadata,'active_basis_sha256':digest_keys(reference_keys(data['node'])),
                         'against_serial':compare(data),'against_first_in_mode':compare(data,first_data)})
        reference_runs[mode]=runs
        print(name,mode,'complete',flush=True)
    inputs=np.loadtxt(directory/'input.pts').astype(numpy_dtype)
    gpu=[]
    first_gpu=None
    with wp.ScopedDevice('cuda'):
        p,n=wp.array(inputs[:,:3],dtype=vector),wp.array(inputs[:,3:],dtype=vector)
        q=wp.array(queries,dtype=vector)
        for index in range(repeats):
            gc.collect();wp.synchronize()
            start=time.perf_counter()
            result=reconstruct(p,n,depth=parameters['depth'],full_depth=parameters['full_depth'],
                base_depth=parameters['base_depth'],solve_depth=parameters['solve_depth'],
                kernel_depth=parameters['kernel_depth'],samples_per_node=parameters['samples_per_node'],
                scale=parameters['scale'],point_weight=parameters['weight'],exact_interpolation=parameters['exact'],
                iters=parameters['iters'],base_v_cycles=parameters['base_v_cycles'],cg_accuracy=parameters['cg_accuracy'])
            wp.synchronize()
            reconstruction_seconds=time.perf_counter()-start
            start=time.perf_counter()
            value,gradient,valid=result.evaluate(q)
            if not valid.numpy().all():raise ValueError('Invalid GPU repeatability query')
            data={'q':np.column_stack([value.numpy(),gradient.numpy()]),'iso':np.array([[result.iso_value]])}
            query_seconds=time.perf_counter()-start
            if first_gpu is None:first_gpu=data
            gpu.append({'reconstruction_seconds':reconstruction_seconds,
                        'query_and_readback_seconds':query_seconds,
                        'active_basis_sha256':digest_keys(result.field.keys.numpy()),
                        'against_serial':compare(data),'against_first_gpu':compare(data,first_gpu)})
            del result
    serial_repeatable=all(r['against_first_in_mode']['bitwise_equal'] and r['active_basis_sha256']==key_hash for r in reference_runs['none'])
    gpu_compatible=all(r['against_serial']['within_budget'] and r['active_basis_sha256']==key_hash for r in gpu)
    gpu_repeatable=all(r['against_first_gpu']['within_budget'] and r['active_basis_sha256']==gpu[0]['active_basis_sha256'] for r in gpu)
    return {'fixture':name,'precision':manifest['precision'],'input_sha256':manifest['input_sha256'],
            'queries_sha256':manifest['queries_sha256'],'queries':len(queries),'parameters':parameters,
            'tolerances':{'absolute':atol,'relative':rtol},'serial_active_basis_sha256':key_hash,
            'serial_bitwise_repeatable':serial_repeatable,'gpu_matches_serial':gpu_compatible,
            'gpu_repeatable_within_budget':gpu_repeatable,
            'parallel_topology_matches_serial':all(r['active_basis_sha256']==key_hash for r in reference_runs['async']),
            'passed':serial_repeatable and gpu_compatible and gpu_repeatable,
            'reference_runs':reference_runs,'gpu_runs':gpu}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',nargs='+',default=['sphere','geometry-cluster-double'])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'.build/repeatability.json')
    args=parser.parse_args()
    if args.repeats<2:parser.error('At least two repeats are required')
    wp.init()
    if not wp.is_cuda_available():raise RuntimeError('Repeatability measurement requires CUDA')
    fixtures=[check_fixture(name,args.repeats) for name in args.fixtures]
    source_paths=sorted([*ROOT.glob('src/**/*.py'),*ROOT.glob('reference/*.cpp'),*ROOT.glob('reference/*.inc'),*ROOT.glob('reference/*.py'),*ROOT.glob('tools/*.py'),*ROOT.glob('tests/*.py')])
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    report={'scope':'Separate serial/default-async official reference and GPU repeatability measurements',
            'parallel_policy':'Parallel variability is reported separately; serial compatibility budgets stay fixed.',
            'reference_parallel_mode':'Default CLI backend for the pinned build without OpenMP: ASYNC, dynamic schedule, hardware thread count',
            'timing_scope':'Reference times cover Solver::Solve only; GPU timings separately cover reconstruction and queries with readback; first invocations may compile kernels',
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
            'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            'cpu':cpu,'python':platform.python_version(),'warp':wp.__version__,
            'build':json.loads((ROOT/'.build/build.json').read_text()),
            'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
            'all_passed':all(r['passed'] for r in fixtures),'repeats':args.repeats,'fixtures':fixtures}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
