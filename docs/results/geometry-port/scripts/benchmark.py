"""Compare the public geometry port with the established implementation on one Warp build."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import warp as wp
import warp._src.codegen as codegen

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("original", "port"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

if args.variant == "original":
    from warp_psr import Field, Field64, evaluate_field, reconstruct
    from warp_psr.field import reference_evaluation_kernel
else:
    from warp.geometry import PoissonField as Field, PoissonField64 as Field64, poisson_evaluate as evaluate_field
    from warp.geometry import poisson_reconstruct
    from warp._src.geometry.poisson.field import reference_evaluation_kernel
    def reconstruct(points, normals, **options):
        if options.pop("exact_interpolation", False):
            raise ValueError("Exact interpolation is outside the public port profile")
        options["iterations"] = options.pop("iters")
        return poisson_reconstruct(points, normals, **options)
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.benchmark_device_field import timed_graph
from tools.cpu_environment import cpu_environment

wp.set_module_options({"fuse_fp": False})


def composed_kernel(dtype):
    field_type = Field64 if dtype == wp.float64 else Field
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    hint = None

    @wp.func(inline=hint)
    def composed(field: field_type, point: vector):
        value, gradient, valid = evaluate_field(field, point)
        return value + dtype(.1) * wp.length_sq(gradient)

    @wp.kernel(module="unique", module_options={"fuse_fp": False})
    def composed_kernel(field: field_type, points: wp.array[vector], output: wp.array[dtype]):
        index = wp.tid()
        output[index] = composed(field, points[index])
    return composed_kernel


def digest(arrays):
    result = hashlib.sha256()
    for array in arrays:
        result.update(np.asarray(array).tobytes())
    return result.hexdigest()


wp.init()
report = dict(variant=args.variant, force_first_compile=args.compile, warp=wp.__version__, environment_before=cpu_environment(), cases=[])
for dataset, depth, precision in (("bunny", 8, "float"), ("horse", 8, "float"),
                                  ("eagle", 8, "float"), ("eagle", 9, "float"),
                                  ("bunny", 8, "double")):
    fixture = Path(f".build/gpu-baselines/inputs/{dataset}-full-d8-spsr")
    positions, normals, parameters, provenance = benchmark_input(SimpleNamespace(fixture=fixture))
    options = reconstruction_parameters(parameters)
    options.update(depth=depth, solve_depth=depth, kernel_depth=depth-2)
    dtype = wp.float64 if precision == "double" else wp.float32
    vector = wp.vec3d if precision == "double" else wp.vec3
    points = wp.array(positions, dtype=vector, device="cuda:0")
    normal_array = wp.array(normals, dtype=vector, device=points.device)
    wp.synchronize_device()
    wp.config.cache_kernels = not args.compile
    start = time.perf_counter()
    result = reconstruct(points, normal_array, **options)
    wp.synchronize_device()
    row = dict(dataset=dataset, depth=depth, precision=precision, points=len(points), input=provenance,
               first_reconstruction_seconds=time.perf_counter()-start, reconstruction_seconds=[], queries=[])
    wp.config.cache_kernels = True
    for iteration in range(9):
        del result
        gc.collect()
        wp.synchronize_device()
        start = time.perf_counter()
        result = reconstruct(points, normal_array, **options)
        wp.synchronize_device()
        if iteration >= 2:
            row["reconstruction_seconds"].append(time.perf_counter()-start)
    row["reconstruction_median_ms"] = statistics.median(row["reconstruction_seconds"]) * 1000
    raw = result._result if args.variant == "port" else result
    row["reconstruction_sha256"] = digest([raw.field.keys.numpy(), raw.field.coefficients.numpy(),
                                            np.asarray([result.iso_value])])
    kernel = composed_kernel(dtype)
    handle = result.device_field()
    for count in sorted(set((1, min(4096, len(points)), len(points)))):
        queries = wp.array(positions[np.linspace(0, len(points)-1, count, dtype=int)], dtype=vector, device=points.device)
        output = wp.empty(count, dtype=dtype, device=points.device)
        wp.config.cache_kernels = not args.compile
        start = time.perf_counter()
        wp.launch(kernel, count, inputs=[handle, queries, output], device=points.device)
        values, gradients, valid = result.evaluate(queries)
        actual = output.numpy()
        expected = values.numpy() + .1 * np.sum(gradients.numpy() ** 2, axis=1)
        np.testing.assert_allclose(actual, expected, rtol=2e-6 if precision=="float" else 2e-14,
                                   atol=2e-6 if precision=="float" else 2e-14)
        query_row = dict(count=count, first_query_seconds=time.perf_counter()-start,
                         sha256=digest([actual, values.numpy(), gradients.numpy(), valid.numpy()]), kernel_ms={})
        wp.config.cache_kernels = True
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                wp.launch(kernel, count, inputs=[handle, queries, output], device=points.device)
        composed_graph = capture.graph
        field = raw.field
        matrix = wp.mat33d if precision == "double" else wp.mat33
        batch_inputs = [field.keys, field.coefficients, field.cumulative, queries,
                        matrix(*field.linear), vector(*field.translation), field.depth]
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                wp.launch(reference_evaluation_kernel(dtype), count, inputs=batch_inputs,
                          outputs=[values, gradients, valid], device=points.device)
        batch_graph = capture.graph
        # Event timing includes only GPU graph execution, not Python dispatch.
        for name, graph in (("batch", batch_graph), ("composed", composed_graph)):
            query_row["kernel_ms"][name] = [timed_graph(graph, 10)/10 for _ in range(7)]
        query_row["median_ms"] = {name:statistics.median(samples) for name,samples in query_row["kernel_ms"].items()}
        host_samples = []
        for iteration in range(9):
            wp.synchronize_device(points.device)
            start = time.perf_counter()
            batch_result = result.evaluate(queries)
            wp.synchronize_device(points.device)
            if iteration >= 2:
                host_samples.append(time.perf_counter() - start)
        query_row["batch_host_seconds"] = host_samples
        query_row["batch_host_median_ms"] = statistics.median(host_samples) * 1000
        row["queries"].append(query_row)
    report["cases"].append(row)
    print(json.dumps(row), flush=True)
report["environment_after"] = cpu_environment()
args.output.write_text(json.dumps(report, indent=2)+"\n")
