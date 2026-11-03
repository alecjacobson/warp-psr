"""Measure first-in-process and warm GPU reconstruction; correctness is a separate gate."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
import warp as wp
from warp_psr import reconstruct
from tools.make_fixture import ROOT
from tools.benchmark_input import benchmark_input, reconstruction_parameters


def benchmark(args):
    wp.init()
    p,n,parameters,provenance=benchmark_input(args)
    vector=wp.vec3d if parameters['precision']=='double' else wp.vec3
    records=[]
    with wp.ScopedDevice('cuda'):
        p=wp.array(p,dtype=vector)
        n=wp.array(n,dtype=vector)
        for i in range(args.repeats+1):
            gc.collect();wp.synchronize()
            start=time.perf_counter()
            result=reconstruct(p,n,**reconstruction_parameters(parameters))
            wp.synchronize()
            record={'iteration':i,'seconds':time.perf_counter()-start,'metadata':result.metadata,
                    'peak_device_pool_bytes':wp.get_mempool_used_mem_high('cuda'),
                    'support_counts':[len(level.keys) for level in result.diagnostics['tree'].levels]}
            records.append(record)
            print(f"Iteration {i}: {record['seconds']:.4f} s; peak pool {record['peak_device_pool_bytes']/2**30:.2f} GiB",flush=True)
            del result
    report={**provenance,'count':len(p),'depth':parameters['depth'],'full_depth':parameters['full_depth'],'parameters':parameters,
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
            'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
            'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
            'warp':wp.__version__,'timing_scope':'GPU reconstruction, including Python orchestration; excludes input upload and queries',
            'first_run_note':'First invocation in process; existing disk kernel cache may be reused. Not a clean-cache compilation measurement.',
            'warm_median_seconds':statistics.median(r['seconds'] for r in records[1:]),'runs':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,help='Use the exact input and all parameters of a validated fixture')
    parser.add_argument('--count',type=int,default=8192)
    parser.add_argument('--depth',type=int,default=6)
    parser.add_argument('--full-depth',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('.build/benchmark.json'))
    args=parser.parse_args()
    if args.count<1 or args.repeats<1:parser.error('count and repeats must be positive')
    benchmark(args)
