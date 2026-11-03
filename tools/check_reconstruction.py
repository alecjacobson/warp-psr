"""Validate independent GPU reconstruction against pinned reference fields."""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from warp_psr import reconstruct
from tools.make_fixture import read_output
from reference.cells import query_cell_widths


def check(directory,device='cuda:0'):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['precision']!='float':
        raise ValueError('End-to-end float64 preprocessing is not implemented yet')
    p=manifest['parameters']
    samples=np.loadtxt(directory/'input.pts').astype(np.float32)
    queries=np.loadtxt(directory/'queries.txt').astype(np.float32)
    with wp.ScopedDevice(device):
        result=reconstruct(wp.array(samples[:,:3],dtype=wp.vec3),wp.array(samples[:,3:],dtype=wp.vec3),
            depth=p['depth'],full_depth=p['full_depth'],point_weight=p['weight'],exact_interpolation=p['exact'],iters=p['iters'])
        v,g,valid=result.evaluate(wp.array(queries,dtype=wp.vec3))
        assert valid.numpy().all()
        actual=np.column_stack([v.numpy(),g.numpy()])
    expected=read_output(directory/'oracle.txt')
    error=actual-expected['q']
    inverse=np.linalg.inv(np.array(result.field.linear).reshape(3,3))
    cubegrad=actual[:,1:]@inverse
    refgrad=expected['q'][:,1:]@inverse
    np.testing.assert_allclose(actual[:,0],expected['q'][:,0],atol=1e-5,rtol=1e-4)
    tree=read_output(directory/'oracle_instrumented.txt')['node']
    normal=read_output(directory/'stages.txt')['normal']
    cube=queries@np.array(result.field.linear).reshape(3,3).T+result.field.translation
    widths=query_cell_widths(tree,normal,cube,p['depth'])
    np.testing.assert_allclose(cubegrad*widths[:,None],refgrad*widths[:,None],atol=1e-5,rtol=1e-4)
    np.testing.assert_allclose(result.iso_value,expected['iso'][0,0],atol=1e-5,rtol=1e-4)
    report={"test":"Independent GPU reconstruction from oriented points","reconstruction_compatibility":True,
        "device":device,"queries":len(queries),"coefficients":len(result.field.keys),
        "max_abs_value_error":float(np.max(np.abs(error[:,0]))),
        "rms_value_error":float(np.sqrt(np.mean(error[:,0]**2))),
        "max_abs_cube_gradient_error":float(np.max(np.abs(cubegrad-refgrad))),
        "max_scaled_gradient_budget_ratio":float(np.max(np.abs(cubegrad-refgrad)*widths[:,None]/(1e-5+1e-4*np.abs(refgrad)*widths[:,None]))),
        "max_unscaled_gradient_budget_ratio":float(np.max(np.abs(cubegrad-refgrad)/(1e-5+1e-4*np.abs(refgrad)))),
        "gradient_gate":"Design budget, scaled by canonical reference integration-cell width",
        "gradient_errors_by_depth":{str(d):float(np.max(np.abs(cubegrad[widths==2.**(-d)]-refgrad[widths==2.**(-d)]))) for d in range(p['depth']+1) if np.any(widths==2.**(-d))},
        "iso_value":result.iso_value,"reference_iso_value":float(expected['iso'][0,0]),
        "metadata":result.metadata}
    np.save(directory/'warp_reconstruction.npy',actual)
    (directory/'reconstruction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();wp.init();check(args.directory,args.device)
