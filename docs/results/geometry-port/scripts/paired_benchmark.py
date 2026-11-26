"""Separate API cost from GPU code by interleaving variants on identical inputs."""
import gc
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import warp as wp
from warp.geometry import poisson_reconstruct
from warp._src.geometry.poisson.field import reference_evaluation_kernel as port_kernel
from warp_psr import reconstruct
from warp_psr.field import reference_evaluation_kernel as original_kernel
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.benchmark_device_field import timed_graph
from tools.cpu_environment import cpu_environment

wp.init()
report = dict(environment_before=cpu_environment(), cases=[])
for dataset, depth, precision in (("bunny",8,"float"), ("horse",8,"float"),
                                  ("eagle",8,"float"), ("eagle",9,"float"), ("bunny",8,"double")):
    positions, normals, parameters, provenance = benchmark_input(SimpleNamespace(
        fixture=Path(f".build/gpu-baselines/inputs/{dataset}-full-d8-spsr")))
    options = reconstruction_parameters(parameters)
    options.update(depth=depth, solve_depth=depth, kernel_depth=depth-2)
    port_options = dict(options)
    if port_options.pop("exact_interpolation", False):
        raise ValueError("Unsupported exact interpolation")
    port_options["iterations"] = port_options.pop("iters")
    dtype = wp.float64 if precision == "double" else wp.float32
    vector = wp.vec3d if precision == "double" else wp.vec3
    matrix = wp.mat33d if precision == "double" else wp.mat33
    points = wp.array(positions, dtype=vector, device="cuda:0")
    normal_array = wp.array(normals, dtype=vector, device=points.device)
    functions = {"original":lambda:reconstruct(points,normal_array,**options),
                 "port":lambda:poisson_reconstruct(points,normal_array,**port_options)}
    last = {name:function() for name,function in functions.items()}
    np.testing.assert_array_equal(last['original'].field.coefficients.numpy(), last['port']._result.field.coefficients.numpy())
    row = dict(dataset=dataset, depth=depth, precision=precision, points=len(points),
               reconstruction_seconds={name:[] for name in functions}, queries=[])
    for iteration in range(14):
        order = ("original","port") if iteration%2 == 0 else ("port","original")
        for name in order:
            del last[name]
            gc.collect()
            wp.synchronize_device(points.device)
            start=time.perf_counter()
            last[name] = functions[name]()
            wp.synchronize_device(points.device)
            if iteration>=2:
                row['reconstruction_seconds'][name].append(time.perf_counter()-start)
    field = last['port']._result.field
    for count in (4096, len(points)):
        queries = wp.array(positions[np.linspace(0,len(points)-1,count,dtype=int)],dtype=vector,device=points.device)
        outputs = [wp.empty(count,dtype=kind,device=points.device) for kind in (dtype,vector,int)]
        inputs = [field.keys,field.coefficients,field.cumulative,queries,matrix(*field.linear),vector(*field.translation),field.depth]
        graphs={}
        checks={}
        for name,factory in (("original",original_kernel),("port",port_kernel)):
            kernel=factory(dtype)
            wp.launch(kernel,count,inputs=inputs,outputs=outputs,device=points.device)
            checks[name]=[array.numpy().copy() for array in outputs]
            with wp.ScopedCapture() as capture:
                for _ in range(10):
                    wp.launch(kernel,count,inputs=inputs,outputs=outputs,device=points.device)
            graphs[name]=capture.graph
        for left,right in zip(checks['original'],checks['port']):
            np.testing.assert_array_equal(left,right)
        query_row=dict(count=count, gpu_ms={name:[] for name in graphs}, wrapper_seconds={name:[] for name in ('raw','public')})
        for iteration in range(16):
            order=("original","port") if iteration%2==0 else ("port","original")
            for name in order:
                sample=timed_graph(graphs[name],10)/10
                if iteration>=2: query_row['gpu_ms'][name].append(sample)
        methods={'raw':last['port']._result.evaluate,'public':last['port'].evaluate}
        for iteration in range(32):
            order=('raw','public') if iteration%2==0 else ('public','raw')
            for name in order:
                wp.synchronize_device(points.device)
                start=time.perf_counter()
                returned=methods[name](queries)
                wp.synchronize_device(points.device)
                if iteration>=2:query_row['wrapper_seconds'][name].append(time.perf_counter()-start)
        row['queries'].append(query_row)
    report['cases'].append(row)
    print(dataset,depth,precision,{name:1000*statistics.median(values) for name,values in row['reconstruction_seconds'].items()},flush=True)
report['environment_after']=cpu_environment()
Path('.build/geometry-port/paired.json').write_text(json.dumps(report,indent=2)+'\n')
