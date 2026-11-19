"""Separate GPU kernel costs and Python work for the horse comparison."""
import argparse
from collections import defaultdict
import cProfile
import json
from pathlib import Path
import pstats
from types import SimpleNamespace

import warp as wp
import warp._src.codegen as codegen

parser=argparse.ArgumentParser()
parser.add_argument("--variant",choices=("original","fixed"),required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
if args.variant=="original":
    codegen.get_func_inline_hint=lambda hint,arg_types,device:hint
from warp_psr import reconstruct
from tools.benchmark_input import benchmark_input,reconstruction_parameters

wp.init()
positions,normals,parameters,provenance=benchmark_input(SimpleNamespace(fixture=Path('.build/gpu-baselines/inputs/horse-full-d8-spsr')))
options=reconstruction_parameters(parameters)
points=wp.array(positions,dtype=wp.vec3,device='cuda:0');normals=wp.array(normals,dtype=wp.vec3,device=points.device)
for _ in range(5):result=reconstruct(points,normals,**options)
wp.synchronize_device()
records=[]
for _ in range(3):
    with wp.ScopedTimer('Reconstruction',cuda_filter=wp.TIMING_ALL,print=False) as timer:
        result=reconstruct(points,normals,**options)
    totals=defaultdict(float)
    for item in timer.timing_results:totals[item.name]+=item.elapsed
    records.append(dict(totals))
profile=cProfile.Profile()
profile.enable()
for _ in range(5):result=reconstruct(points,normals,**options)
wp.synchronize_device()
profile.disable()
profile.dump_stats(str(args.output.with_suffix('.pstats')))
stats=pstats.Stats(profile)
rows=sorted(((key,data) for key,data in stats.stats.items()),key=lambda row:row[1][3],reverse=True)
args.output.write_text(json.dumps({'variant':args.variant,'gpu_ms':records,'cpu_profile':[{'file':key[0],'line':key[1],'function':key[2],'primitive_calls':data[0],'total_calls':data[1],'self_seconds':data[2],'cumulative_seconds':data[3]} for key,data in rows[:70]]},indent=2)+'\n')
print('GPU total ms:',[sum(r.values()) for r in records],flush=True)
