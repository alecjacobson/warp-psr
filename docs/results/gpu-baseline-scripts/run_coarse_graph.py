import argparse,gc,hashlib,importlib.util,json,statistics,time
from pathlib import Path
import sys
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())))
import numpy as np
import warp as wp
import warp_psr.solver as solver
from warp_psr import reconstruct
from tools.benchmark_input import benchmark_input,reconstruction_parameters
from tools.benchmark_gpu_baseline import gpu_processes
from tools.cpu_environment import cpu_environment
parser=argparse.ArgumentParser();parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--depth',type=int,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
spec=importlib.util.spec_from_file_location('coarse_graph',Path(__file__).with_name('coarse_graph.py'));candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
wp.init();wp.set_device('cuda:0')
p,n,params,provenance=benchmark_input(args);params.update(depth=args.depth,solve_depth=args.depth,kernel_depth=args.depth-2)
p,n=(wp.array(a,dtype=wp.vec3) for a in (p,n));options=reconstruction_parameters(params)
original=solver.relax_scheduled;methods={'production':original,'coarse_graph':candidate.relax_scheduled}
records=[];reference=None;environment=cpu_environment()
try:
 for i,name in enumerate(list(methods)*2+['production','coarse_graph','coarse_graph','production']*7):
  solver.relax_scheduled=methods[name];gc.collect();gpu_processes();wp.synchronize();start=time.perf_counter()
  result=reconstruct(p,n,**options);wp.synchronize();elapsed=time.perf_counter()-start
  arrays=(result.field.keys.numpy(),result.field.coefficients.numpy(),result.iso_value)
  if reference is None:reference=arrays
  else:
   for a,b in zip(reference,arrays):np.testing.assert_array_equal(a,b)
  record=dict(iteration=i,method=name,warm=i>=4,seconds=elapsed,bitwise_equal=True);records.append(record);print(json.dumps(record),flush=True);del result
finally:solver.relax_scheduled=original
report=dict(**provenance,parameters=params,environment=environment,environment_after=cpu_environment(),records=records,candidate_sha256=hashlib.sha256(Path(__file__).with_name('coarse_graph.py').read_bytes()).hexdigest(),medians={name:statistics.median(r['seconds'] for r in records if r['warm'] and r['method']==name) for name in methods})
args.output.write_text(json.dumps(report,indent=2)+'\n')
