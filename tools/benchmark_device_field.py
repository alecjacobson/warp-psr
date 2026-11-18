"""Compare warmed field queries and reconstruction against a Git baseline."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np
import warp as wp
from warp_psr import Field, Field64, evaluate_field, reconstruct
import warp_psr.field as field_module
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.benchmark_meshing import gpu_processes
from tools.cpu_environment import cpu_environment
from tools.fixture_provenance import production_hashes

wp.set_module_options({"fuse_fp": False})


def fused_kernel(dtype):
    field_type = Field64 if dtype == wp.float64 else Field
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3

    @wp.kernel
    def fused_query_kernel(field: field_type, queries: wp.array[vector], values: wp.array[dtype],
                           gradients: wp.array[vector], valid: wp.array[int]):
        i = wp.tid()
        value, gradient, ok = evaluate_field(field, queries[i])
        values[i] = value
        gradients[i] = gradient
        valid[i] = ok
    return fused_query_kernel


def timed_graph(graph, repetitions):
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    wp.record_event(start)
    for _ in range(repetitions):
        wp.capture_launch(graph)
    wp.record_event(end)
    wp.synchronize_event(end)
    return wp.get_event_elapsed_time(start, end) / repetitions


def run(args, baseline, baseline_adjoint):
    wp.init()
    before = cpu_environment()
    positions, normals, parameters, provenance = benchmark_input(SimpleNamespace(fixture=args.fixture))
    options = reconstruction_parameters(parameters)
    if args.depth is not None:
        options.update(depth=args.depth, solve_depth=args.depth, kernel_depth=max(args.depth-2, 0))
    dtype = wp.float64 if args.precision == "double" else wp.float32
    vector = wp.vec3d if dtype == wp.float64 else wp.vec3
    matrix = wp.mat33d if dtype == wp.float64 else wp.mat33
    p = wp.array(positions, dtype=vector, device="cuda:0")
    n = wp.array(normals, dtype=vector, device=p.device)
    current = field_module.reference_evaluation_kernel
    variants = {"baseline": baseline.reference_evaluation_kernel, "current": current}
    report = dict(input=provenance, points=len(p), options=options, precision=args.precision,
                  baseline_revision=args.baseline, production_source_sha256=production_hashes(),
                  environment_before=before, warp=wp.__version__, queries=[], reconstruction=[])
    try:
        # Warm both kernels, solver, allocations, and caches before measurement.
        for variant in ("baseline", "current", "baseline", "current"):
            field_module.reference_evaluation_kernel = variants[variant]
            result = reconstruct(p, n, **options)
        field = result.field
        common = [field.keys, field.coefficients, field.cumulative]
        transform = [matrix(*field.linear), vector(*field.translation), field.depth]
        fused = fused_kernel(dtype)
        handle = result.device_field()
        for count in sorted(set((1, min(4096, len(p)), len(p)))):
            queries = wp.array(positions[np.linspace(0, len(p)-1, count, dtype=int)], dtype=vector, device=p.device)
            outputs = [wp.empty(count, dtype=dtype, device=p.device), wp.empty(count, dtype=vector, device=p.device),
                       wp.empty(count, dtype=int, device=p.device)]
            graphs, expected = {}, None
            for variant in ("baseline", "current", "fused"):
                kernel = fused if variant == "fused" else variants[variant](dtype)
                inputs = [handle, queries] if variant == "fused" else common + [queries] + transform
                wp.launch(kernel, count, inputs=inputs, outputs=outputs, device=p.device)
                actual = [array.numpy() for array in outputs]
                if expected is None:
                    expected = actual
                else:
                    for a, e in zip(actual, expected):
                        np.testing.assert_array_equal(a, e)
                with wp.ScopedCapture(device=p.device) as capture:
                    for _ in range(10):
                        wp.launch(kernel, count, inputs=inputs, outputs=outputs, device=p.device)
                graphs[variant] = capture.graph
            timings = {variant: [] for variant in graphs}
            for _ in range(args.rounds):
                for variant in ("baseline", "current", "fused", "fused", "current", "baseline"):
                    gpu_processes()
                    timings[variant].append(timed_graph(graphs[variant], 10) / 10)
            medians = {variant: statistics.median(values) for variant, values in timings.items()}
            row = dict(count=count, kernel_ms=timings, median_ms=medians, bitwise_equal=True)
            report["queries"].append(row)
            print(json.dumps(dict(count=count, median_ms=medians)), flush=True)
        # The batched backward evaluator also shares the new per-query VJP.
        from warp_psr.field_adjoint import precision_kernels
        cube_queries = wp.array(positions @ np.array(field.linear).reshape(3, 3).T + np.array(field.translation),
                                dtype=vector, device=p.device)
        value_seed = wp.ones(len(p), dtype=dtype, device=p.device)
        # Nonnegative coefficient contributions avoid cancellation-dependent
        # atomic summation noise in the large-cloud comparison. Mixed losses
        # are checked against finite differences in the correctness suite.
        spatial_seed = wp.zeros(len(p), dtype=vector, device=p.device)
        adjoint_outputs = [wp.zeros_like(field.coefficients), wp.zeros_like(field.cumulative), wp.zeros_like(p)]
        adjoint_graphs, expected = {}, None
        for variant, kernels in (("baseline", baseline_adjoint.precision_kernels(dtype)),
                                 ("current", precision_kernels(dtype))):
            for array in adjoint_outputs:
                array.zero_()
            inputs = common + [cube_queries, field.depth, value_seed, spatial_seed] + adjoint_outputs
            wp.launch(kernels.evaluate_backward, len(p), inputs=inputs, device=p.device)
            actual = [array.numpy() for array in adjoint_outputs]
            if expected is None:
                expected = actual
            else:
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a, e, rtol=3e-5 if dtype == wp.float32 else 1e-11,
                                               atol=3e-5 if dtype == wp.float32 else 1e-11)
            with wp.ScopedCapture(device=p.device) as capture:
                for _ in range(10):
                    wp.launch(kernels.evaluate_backward, len(p), inputs=inputs, device=p.device)
            adjoint_graphs[variant] = capture.graph
        adjoint_timings = {variant: [] for variant in adjoint_graphs}
        for _ in range(args.rounds):
            for variant in ("baseline", "current", "current", "baseline"):
                gpu_processes()
                adjoint_timings[variant].append(timed_graph(adjoint_graphs[variant], 10) / 10)
        report["batched_backward"] = dict(
            count=len(p), kernel_ms=adjoint_timings,
            median_ms={variant: statistics.median(values) for variant, values in adjoint_timings.items()},
            gradient_agreement=True, loss_seeds=dict(value=1, spatial=[0, 0, 0]))
        print(json.dumps(dict(batched_backward=report["batched_backward"]["median_ms"])), flush=True)
        # Alternate in ABBA order, including allocation and host dispatch costs.
        expected = None
        for _ in range(args.rounds):
            for variant in ("baseline", "current", "current", "baseline"):
                field_module.reference_evaluation_kernel = variants[variant]
                gc.collect()
                gpu_processes()
                wp.synchronize_device(p.device)
                start = time.perf_counter()
                result = reconstruct(p, n, **options)
                wp.synchronize_device(p.device)
                elapsed = time.perf_counter() - start
                report["reconstruction"].append(dict(variant=variant, seconds=elapsed))
                actual = [result.field.keys.numpy(), result.field.coefficients.numpy(), np.array([result.iso_value])]
                if expected is None:
                    expected = actual
                else:
                    for a, e in zip(actual, expected):
                        np.testing.assert_array_equal(a, e)
                del result
        report["reconstruction_median_ms"] = {
            variant: 1000 * statistics.median(row["seconds"] for row in report["reconstruction"] if row["variant"] == variant)
            for variant in variants}
        report["reconstruction_bitwise_equal"] = True
        report["environment_after"] = cpu_environment()
        report["methodology"] = (
            "Only the field evaluation kernel is exchanged: the baseline is loaded from Git and the remaining "
            "forward reconstruction code is unchanged. Query event timings use 10 graph replays of 10 launches, "
            "in baseline/current/fused/fused/current/baseline order. Reconstruction uses synchronized wall time "
            "in ABBA order. Batched backward kernel timings use ABBA order and the same graph repetition count. "
            "Compilation, uploads, validation, and garbage collection are outside timing. "
            "No other GPU processes are permitted; cgroup counters record CPU throttling.")
        print(json.dumps(report["reconstruction_median_ms"]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        field_module.reference_evaluation_kernel = current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--precision", choices=["float", "double"], default="float")
    parser.add_argument("--baseline", default="7a6a6190786aeacce9775d0575e9b13e611d5aae")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    source = subprocess.check_output(["git", "show", f"{args.baseline}:src/warp_psr/field.py"], text=True, timeout=30)
    adjoint_source = subprocess.check_output(["git", "show", f"{args.baseline}:src/warp_psr/field_adjoint.py"],
                                             text=True, timeout=30)
    Path(".build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="device-field-baseline-", dir=".build") as directory:
        path = Path(directory) / "baseline_field.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("device_field_baseline", path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        adjoint_path = Path(directory) / "baseline_field_adjoint.py"
        adjoint_path.write_text(adjoint_source)
        spec = importlib.util.spec_from_file_location("warp_psr.device_field_adjoint_baseline", adjoint_path)
        baseline_adjoint = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline_adjoint
        spec.loader.exec_module(baseline_adjoint)
        run(args, baseline, baseline_adjoint)
