"""Measure opt-in forward and checkpointed backward separately, without meshing."""
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
from warp_psr.reconstruction_adjoint import reconstruct_differentiable
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.benchmark_meshing import gpu_processes
from tools.cpu_environment import cpu_environment
from tools.fixture_provenance import production_hashes


def benchmark(args):
    wp.init()
    device=wp.get_device('cuda:0')
    p,n,parameters,provenance=benchmark_input(args)
    vector=wp.vec3d if parameters['precision']=='double' else wp.vec3
    scalar=wp.float64 if vector==wp.vec3d else wp.float32
    tracked=args.mode=='differentiable'
    queries=wp.array(p[np.linspace(0,len(p)-1,min(32,len(p)),dtype=int)],dtype=vector,device=device)
    positions=wp.array(p,dtype=vector,device=device,requires_grad=tracked)
    normals=wp.array(n,dtype=vector,device=device,requires_grad=tracked)
    seed=wp.full(len(queries),1/len(queries),dtype=scalar,device=device)
    options=reconstruction_parameters(parameters)
    records=[]
    for iteration in range(args.repeats+1):
        gc.collect();gpu_processes();wp.synchronize_device(device)
        if tracked:
            positions.grad.zero_();normals.grad.zero_()
        begin=time.perf_counter()
        tape=wp.Tape() if tracked else None
        result=(reconstruct_differentiable(positions,normals,tape=tape,**options) if tracked else reconstruct(positions,normals,**options))
        wp.synchronize_device(device)
        forward=time.perf_counter()-begin
        forward_pool=wp.get_mempool_used_mem_current(device)
        if tracked:
            with tape:values,spatial,valid=result.evaluate(queries,centered=True)
        else:
            values,spatial,valid=result.evaluate(queries)
        wp.synchronize_device(device)
        backward=None
        if tracked:
            begin=time.perf_counter();tape.backward(grads={values:seed});wp.synchronize_device(device)
            backward=time.perf_counter()-begin
        record={'iteration':iteration,'forward_seconds':forward,'backward_seconds':backward,
                'retained_pool_after_forward_bytes':forward_pool,'peak_pool_bytes':wp.get_mempool_used_mem_high(device),
                'gpu_processes_after':gpu_processes()}
        if not valid.numpy().all():raise AssertionError('benchmark queries must be valid')
        if tracked:
            gp,gn=positions.grad.numpy(),normals.grad.numpy()
            if not np.isfinite(gp).all() or not np.isfinite(gn).all():raise AssertionError('nonfinite reconstruction gradient')
            translation_actual=gp.astype(float).sum(axis=0)
            translation_expected=-spatial.numpy().astype(float).mean(axis=0)
            normal_scale=float(np.sum(gn.astype(float)*n))
            np.testing.assert_allclose(translation_actual,translation_expected,rtol=1e-3,atol=5e-5)
            np.testing.assert_allclose(normal_scale,0.,atol=1e-6)
            record['invariants']={'translation_actual':translation_actual.tolist(),'translation_expected':translation_expected.tolist(),
                                  'uniform_normal_scale_derivative':normal_scale}
            record['gradient_norms']={'positions':float(np.linalg.norm(gp.astype(float))),
                                      'normals':float(np.linalg.norm(gn.astype(float)))}
        records.append(record);print(json.dumps(record),flush=True)
        del result,tape,values,spatial,valid
    report={'mode':args.mode,'input':provenance,'count':len(positions),'parameters':parameters,'runs':records,
        'warm_forward_median':statistics.median(r['forward_seconds'] for r in records[1:]),
        'warm_backward_median':statistics.median(r['backward_seconds'] for r in records[1:]) if tracked else None,
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'])),
        'production_source_sha256':production_hashes(),'warp':wp.__version__,'environment':cpu_environment(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        'scope':'Wall time with GPU completion. Upload and 32-query centered field evaluation excluded; backward includes stage replay and input gradients. Pool high watermark is process-wide, native CUDA graph storage is outside the pool.',
        'first_call':'First invocation in this process, may reuse disk kernel cache.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--mode',choices=['forward','differentiable'],default='differentiable')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    benchmark(args)
