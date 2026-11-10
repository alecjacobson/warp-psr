import sys,json
from pathlib import Path
project = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0, str(project))
import numpy as np
import torch
from tools.benchmark_cached_fft import original_loader,cached_class
from tools.benchmark_gpu_baseline import digest
base=project/'.build/gpu-baselines';root=base/'vendor/shape_as_points';fixture=base/'inputs/eagle-full-d8-spsr'
torch.set_num_threads(4);torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True)
data=np.loadtxt(fixture/'input.pts',dtype=np.float32);p,n=data[:,:3],data[:,3:];lo,hi=p.min(0),p.max(0);side=np.float32(np.max(hi-lo)*np.float32(1.1));origin=(lo+hi)*np.float32(.5)-side*np.float32(.5)
p=torch.tensor((p-origin)/side,device='cuda')[None];n=torch.tensor(n.copy(),device='cuda')[None]
records=[]
for depth in (8,9):
 resolution=2**depth
 original=original_loader(root)((resolution,)*3,sig=10).cuda();cached=cached_class(root)((resolution,)*3,sig=10).cuda()
 expected=original(p,n);actual=cached(p,n)
 equal=torch.equal(expected,actual);error=(expected-actual).abs().max().item();assert equal,(depth,error)
 row=dict(depth=depth,resolution=resolution,values=expected.numel(),bitwise_equal=equal,max_absolute_error=error)
 records.append(row);print(json.dumps(row),flush=True)
 del original,cached,expected,actual;torch.cuda.empty_cache()
(base/'cached-fft-full-field-validation.json').write_text(json.dumps(dict(input_sha256=digest(fixture/'input.pts'),points=p.shape[1],deterministic_scatter=True,records=records),indent=2)+'\n')
