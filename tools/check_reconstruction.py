"""Validate independent GPU reconstruction against pinned reference fields."""
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import warp as wp
from warp_psr import reconstruct
from tools.make_fixture import read_output, ROOT
from reference.cells import query_cell_widths


def check(directory,device='cuda:0'):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    double=manifest['precision']=='double'
    numpy_dtype=np.float64 if double else np.float32
    vector=wp.vec3d if double else wp.vec3
    atol,rtol=(1e-8,1e-6) if double else (1e-5,1e-4)
    p=manifest['parameters']
    samples=np.loadtxt(directory/'input.pts').astype(numpy_dtype)
    queries=np.loadtxt(directory/'queries.txt').astype(numpy_dtype)
    with wp.ScopedDevice(device):
        result=reconstruct(wp.array(samples[:,:3],dtype=vector),wp.array(samples[:,3:],dtype=vector),
            depth=p['depth'],full_depth=p['full_depth'],base_depth=p['base_depth'],solve_depth=p['solve_depth'],kernel_depth=p['kernel_depth'],
            scale=p['scale'],samples_per_node=p['samples_per_node'],point_weight=p['weight'],exact_interpolation=p['exact'],
            iters=p['iters'],base_v_cycles=p['base_v_cycles'],cg_accuracy=p['cg_accuracy'])
        v,g,valid=result.evaluate(wp.array(queries,dtype=vector))
        assert valid.numpy().all()
        actual=np.column_stack([v.numpy(),g.numpy()])
    expected=read_output(directory/'oracle.txt')
    error=actual-expected['q']
    inverse=np.linalg.inv(np.array(result.field.linear).reshape(3,3))
    cubegrad=actual[:,1:]@inverse
    refgrad=expected['q'][:,1:]@inverse
    np.testing.assert_allclose(actual[:,0],expected['q'][:,0],atol=atol,rtol=rtol)
    centered_actual=actual[:,0]-result.iso_value
    centered_expected=expected['q'][:,0]-expected['iso'][0,0]
    np.testing.assert_allclose(centered_actual,centered_expected,atol=atol,rtol=rtol)
    tree=read_output(directory/'oracle_instrumented.txt')['node']
    normal=read_output(directory/'stages.txt')['normal']
    cube=queries@np.array(result.field.linear).reshape(3,3).T+result.field.translation
    widths=query_cell_widths(tree,normal,cube,p['depth'])
    np.testing.assert_allclose(cubegrad*widths[:,None],refgrad*widths[:,None],atol=atol,rtol=rtol)
    np.testing.assert_allclose(result.iso_value,expected['iso'][0,0],atol=atol,rtol=rtol)
    def errors(selection):
        delta=error[selection,0]
        centered_delta=(centered_actual-centered_expected)[selection]
        gradient_delta=(cubegrad-refgrad)[selection]
        h=widths[selection,None]
        return {'queries':len(delta),
                'max_abs_value_error':float(np.max(np.abs(delta))),
                'rms_value_error':float(np.sqrt(np.mean(delta**2))),
                'max_abs_centered_value_error':float(np.max(np.abs(centered_delta))),
                'rms_centered_value_error':float(np.sqrt(np.mean(centered_delta**2))),
                'max_abs_cube_gradient_error':float(np.max(np.abs(gradient_delta))),
                'rms_cube_gradient_error':float(np.sqrt(np.mean(gradient_delta**2))),
                'max_scaled_gradient_budget_ratio':float(np.max(np.abs(gradient_delta)*h/(atol+rtol*np.abs(refgrad[selection])*h)))}
    report={"test":"Independent GPU reconstruction from oriented points","reconstruction_compatibility":True,
        "source_revision":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        "source_dirty":bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
        "production_source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ROOT.glob('src/**/*.py'))},
        "device":device,"queries":len(queries),"coefficients":len(result.field.keys),
        "field_tolerances":{"absolute":atol,"relative":rtol},
        "max_abs_value_error":float(np.max(np.abs(error[:,0]))),
        "max_abs_centered_value_error":float(np.max(np.abs(centered_actual-centered_expected))),
        "rms_value_error":float(np.sqrt(np.mean(error[:,0]**2))),
        "max_abs_cube_gradient_error":float(np.max(np.abs(cubegrad-refgrad))),
        "max_scaled_gradient_budget_ratio":float(np.max(np.abs(cubegrad-refgrad)*widths[:,None]/(atol+rtol*np.abs(refgrad)*widths[:,None]))),
        "max_unscaled_gradient_budget_ratio":float(np.max(np.abs(cubegrad-refgrad)/(atol+rtol*np.abs(refgrad)))),
        "gradient_gate":"Design budget, scaled by canonical reference integration-cell width",
        "gradient_errors_by_depth":{str(d):float(np.max(np.abs(cubegrad[widths==2.**(-d)]-refgrad[widths==2.**(-d)]))) for d in range(p['depth']+1) if np.any(widths==2.**(-d))},
        "errors_by_depth":{str(d):errors(widths==2.**(-d)) for d in range(p['depth']+1) if np.any(widths==2.**(-d))},
        "errors_by_query_family":{name:errors(slice(*bounds)) for name,bounds in manifest.get('query_families',{}).items()},
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
