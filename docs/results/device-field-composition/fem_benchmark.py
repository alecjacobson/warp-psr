"""Measure FEM assembly outside PSR for the struct-inlining policy."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import warp as wp
import warp._src.codegen as codegen

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("original", "fixed"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.variant == "original":
    codegen.get_func_inline_hint = lambda hint, arg_types, device: hint

import warp.fem as fem
from tools.cpu_environment import cpu_environment


@fem.integrand
def stiffness_form(sample: fem.Sample, trial: fem.Field, test: fem.Field):
    return wp.dot(fem.grad(trial, sample), fem.grad(test, sample))


wp.init()
wp.set_module_options({"fuse_fp": False})
report = dict(variant=args.variant, environment_before=cpu_environment(), cases=[])
with wp.ScopedDevice("cuda:0"):
    for degree, resolution in ((1, 32), (2, 16)):
        geometry = fem.Grid3D(res=wp.vec3i(resolution))
        space = fem.make_polynomial_space(geometry, degree=degree)
        domain = fem.Cells(geometry)
        trial, test = fem.make_trial(space, domain=domain), fem.make_test(space, domain=domain)
        times = []
        for iteration in range(10):
            wp.synchronize_device()
            start = time.perf_counter()
            matrix = fem.integrate(stiffness_form, fields={"trial":trial,"test":test},
                                   kernel_options={"fuse_fp":False})
            wp.synchronize_device()
            if iteration >= 3:
                times.append(time.perf_counter()-start)
        arrays = [matrix.offsets.numpy(),matrix.columns.numpy(),matrix.values.numpy()]
        digest = hashlib.sha256()
        for array in arrays:
            digest.update(array.tobytes())
        row = dict(degree=degree, resolution=resolution, dofs=space.node_count(), seconds=times,
                   median_ms=statistics.median(times)*1000, sha256=digest.hexdigest(),
                   value_sum=float(np.sum(arrays[-1].astype(float))))
        report["cases"].append(row)
        print(json.dumps(row),flush=True)
report["environment_after"] = cpu_environment()
args.output.write_text(json.dumps(report,indent=2)+"\n")
