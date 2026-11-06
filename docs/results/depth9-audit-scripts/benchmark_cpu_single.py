"""Time the unchanged C++ solver, reading the immutable fixture input directly."""
import argparse, json, os, statistics, subprocess
from pathlib import Path
from tools.cpu_environment import cpu_environment
from tools.datasets import checksum
from tools.make_fixture import ROOT, oracle_parameters, read_output

parser=argparse.ArgumentParser()
parser.add_argument('--fixture',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--repeats',type=int,default=0)
args=parser.parse_args()
manifest=json.loads((args.fixture/'manifest.json').read_text())
input_path=args.fixture/'input.pts'
assert checksum(input_path)==manifest['input_sha256']
parameters=manifest['parameters']
scratch=args.output.parent/(args.output.stem+'-scratch')
scratch.mkdir(exist_ok=True)
(scratch/'queries.txt').write_text('')
binary=ROOT/'.build/oracle_openmp'
command=[str(binary),str(input_path),str(scratch/'queries.txt'),str(scratch/'output.txt'),*oracle_parameters(parameters)]
environment={k:v for k,v in os.environ.items() if not k.startswith('WARP_PSR_')}
environment.update(WARP_PSR_REFERENCE_PARALLELISM='openmp',OMP_THREAD_LIMIT='4',OMP_NUM_THREADS='4',OMP_DYNAMIC='FALSE',
                   WARP_PSR_TIMING_FILE=str(scratch/'solve-seconds.txt'))
before=cpu_environment();runs=[]
for i in range(args.repeats+1):
    subprocess.run(command,env=environment,check=True,capture_output=True)
    seconds=float((scratch/'solve-seconds.txt').read_text())
    runs.append({'iteration':i,'seconds':seconds})
    print(i,seconds,flush=True)
after=cpu_environment()
settings=read_output(scratch/'output.txt')
assert int(settings['openmp_effective_threads'][0,0])==4
report={'fixture':manifest['name'],'input_sha256':manifest['input_sha256'],'count':parameters['count'],'parameters':parameters,
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)),
        'cpu':next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        'execution_environment':before,'execution_environment_after':after,'openmp_thread_limit':4,'openmp_effective_threads':4,
        'configured_threads':int(settings['threads'][0,0]),'schedule':int(settings['schedule'][0,0]),'chunk_size':int(settings['chunk_size'][0,0]),
        'timing_scope':'Unchanged official Solver::Solve; normalization, tree, assembly, solve and isovalue. Excludes parsing, queries, export and process startup.',
        'input_policy':'Read the exact immutable fixture ASCII file directly; Samples<Real> performs the same float32 cast as the fixture benchmark.',
        'build':json.loads((ROOT/'.build/build.json').read_text()),'reference_executable_sha256':checksum(binary),
        'benchmark_script_sha256':checksum(__file__),'warm_median_seconds':runs[0]['seconds'],'runs':runs}
assert len(runs)==1
report['measurement_policy']='One fresh-process CPU solve; the outer scheduler discards its first accepted invocation and takes the median of five subsequent accepted invocations.'
args.output.write_text(json.dumps(report,indent=2)+'\n')
