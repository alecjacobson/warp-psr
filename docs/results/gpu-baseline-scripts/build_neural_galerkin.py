"""Reproduce the optional external extension build in the active PyTorch environment."""
import argparse,os,subprocess,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--cache',type=Path,required=True)
parser.add_argument('--cuda-home',type=Path,required=True)
parser.add_argument('--arch',default='8.9')
args=parser.parse_args()
source=args.source.resolve();cache=args.cache.resolve();cache.mkdir(parents=True,exist_ok=True)
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert revision=='5ec184004edbbf7492e49167d5063ac16712f629',revision
p=source/'csrc/sparse_solve/solve_pcg.cu';text=p.read_text()
if 'CUSPARSE_MV_ALG_DEFAULT' in text:
 assert text.count('CUSPARSE_MV_ALG_DEFAULT')==4
 p.write_text(text.replace('CUSPARSE_MV_ALG_DEFAULT','CUSPARSE_SPMV_ALG_DEFAULT'))
import torch
site=Path(torch.__file__).resolve().parent.parent/'nvidia'
lib=cache/'link-libraries';lib.mkdir(exist_ok=True)
for package,soname in [('cusolver','libcusolver.so.11'),('cusparse','libcusparse.so.12')]:
 target=site/package/'lib'/soname;assert target.exists(),target
 link=lib/soname.rsplit('.',1)[0]
 if not link.exists():link.symlink_to(target)
env=dict(os.environ,CUDA_HOME=str(args.cuda_home.resolve()),TORCH_CUDA_ARCH_LIST=args.arch,MAX_JOBS='2',TMPDIR=str(cache),CPATH=':'.join(str(p) for p in site.glob('*/include')),LIBRARY_PATH=':'.join([str(lib)]+[str(p) for p in site.glob('*/lib')]))
subprocess.run([sys.executable,'setup.py','build_ext','--build-temp',str(cache/'temp'),'--build-lib',str(cache/'lib'),'--inplace'],cwd=source,env=env,check=True)
