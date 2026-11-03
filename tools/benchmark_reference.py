"""Time the pinned C++ solve with an explicit serial or default-async backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import numpy as np
from tools.make_fixture import ROOT, points_for, oracle_parameters, read_output


def benchmark(args):
    directory=ROOT/'.build/reference-benchmark'/f'{args.count}-{args.depth}-{args.full_depth}-{args.parallelism}'
    directory.mkdir(parents=True,exist_ok=True)
    p,n=points_for('torus',args.count,2026)
    np.savetxt(directory/'input.pts',np.column_stack([p,n]).astype('f'),fmt='%.17g')
    (directory/'queries.txt').write_text('')
    parameters={'depth':args.depth,'full_depth':args.full_depth,'base_depth':args.full_depth,
                'solve_depth':args.depth,'kernel_depth':max(args.depth-2,0),'scale':1.1,
                'samples_per_node':1.5,'weight':2.,'exact':False,'iters':8,'cg_accuracy':1e-3,
                'base_v_cycles':1,'precision':'float'}
    command=[str(ROOT/'.build/oracle'),str(directory/'input.pts'),str(directory/'queries.txt'),str(directory/'output.txt'),
             *oracle_parameters(parameters)]
    environment={**os.environ,'WARP_PSR_TIMING_FILE':str(directory/'solve-seconds.txt'),
                 'WARP_PSR_REFERENCE_PARALLELISM':args.parallelism}
    runs=[]
    for i in range(args.repeats+1):
        subprocess.run(command,env=environment,check=True,capture_output=True)
        seconds=float((directory/'solve-seconds.txt').read_text())
        runs.append({'iteration':i,'seconds':seconds})
        print(f'Iteration {i}: {seconds:.4f} s',flush=True)
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    settings=read_output(directory/'output.txt')
    report={'count':args.count,'shape':'torus','seed':2026,'parameters':parameters,'cpu':cpu,
            'parallelism':'ThreadPool::ASYNC (default CLI backend without OpenMP)' if args.parallelism=='async' else 'ThreadPool::NONE (serial), matching the correctness fixtures',
            'configured_threads':int(settings['threads'][0,0]),'schedule':int(settings['schedule'][0,0]),'chunk_size':int(settings['chunk_size'][0,0]),
            'timing_scope':'Unchanged official Solver::Solve; includes normalization, tree, assembly, solve, and isovalue; excludes input parsing, queries, export, and process startup',
            'comparison_limit':'Explicit library backend; input parsing, mesh extraction and CLI startup are excluded',
            'input_sha256':hashlib.sha256((directory/'input.pts').read_bytes()).hexdigest(),
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
            'build':json.loads((ROOT/'.build/build.json').read_text()),
            'warm_median_seconds':statistics.median(r['seconds'] for r in runs[1:]),'runs':runs}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=100000)
    parser.add_argument('--depth',type=int,default=8)
    parser.add_argument('--full-depth',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--parallelism',choices=['none','async'],default='none')
    parser.add_argument('--output',type=Path,default=ROOT/'.build/benchmark-reference.json')
    args=parser.parse_args()
    if args.count<1 or args.repeats<1:parser.error('count and repeats must be positive')
    benchmark(args)
