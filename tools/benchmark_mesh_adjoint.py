"""Measure a Warp MC vertex loss backward through the full reconstruction."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import warp as wp
from warp_psr import reconstruct_differentiable
import warp_psr.meshing as meshing
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.benchmark_meshing import gpu_processes
from tools.cpu_environment import cpu_environment
from tools.fixture_provenance import production_hashes


def benchmark(args):
    wp.init()
    device = wp.get_device('cuda:0')
    points, normals, parameters, provenance = benchmark_input(args)
    if parameters['precision'] != 'float':
        raise ValueError('Warp MC requires float32 reconstruction')
    p = wp.array(points, dtype=wp.vec3, device=device, requires_grad=True)
    n = wp.array(normals, dtype=wp.vec3, device=device, requires_grad=True)
    options = reconstruction_parameters(parameters)
    bounds, records, profile = None, [], {}

    # The final extra invocation is instrumented; warm wall times are unmodified.
    for iteration in range(args.repeats + 2):
        profiled = iteration == args.repeats + 1
        gc.collect()
        before = gpu_processes()
        p.grad.zero_(); n.grad.zero_(); wp.synchronize_device(device)
        events = {}

        def instrument(function, name):
            def wrapped(*a, **kw):
                start = wp.Event(device, enable_timing=True)
                end = wp.Event(device, enable_timing=True)
                wp.record_event(start)
                answer = function(*a, **kw)
                wp.record_event(end)
                events.setdefault(name, []).append((start, end))
                return answer
            return wrapped

        tape = wp.Tape()
        if profiled:
            record = tape.record_func

            def record_func(function, arrays):
                if function.__module__ == 'warp_psr.reconstruction_adjoint':
                    function = instrument(function, 'reconstruction_backward')
                record(function, arrays)

            tape.record_func = record_func
        start = time.perf_counter()
        result = reconstruct_differentiable(p, n, tape=tape, **options)
        wp.synchronize_device(device)
        reconstruction = time.perf_counter() - start
        if bounds is None:
            transform = result.field.transforms.numpy()[1]
            inverse = np.linalg.inv(transform[:3, :3])
            cube = np.array([(x, y, z) for x in (.001, .999) for y in (.001, .999) for z in (.001, .999)])
            model = (cube - transform[:3, 3]) @ inverse.T
            bounds = (model.min(axis=0).tolist(), model.max(axis=0).tolist())
        sparse = meshing.sparse_marching_cubes_from_cells
        dense_descriptor = meshing.IsoSurfaceMarchingCubes.__dict__['extract']
        dense = meshing.IsoSurfaceMarchingCubes.extract
        if profiled:
            result.field.evaluate = instrument(result.field.evaluate, 'field_evaluation_forward')
            meshing.sparse_marching_cubes_from_cells = instrument(sparse, 'marching_cubes_forward')
            meshing.IsoSurfaceMarchingCubes.extract = classmethod(
                lambda cls, *a, **kw: instrument(dense, 'marching_cubes_forward')(*a, **kw))
        try:
            start = time.perf_counter()
            with tape:
                mesh = meshing.extract_mesh(result, resolution=args.resolution, bounds=bounds,
                                            backend=args.backend, differentiable=True)
            wp.synchronize_device(device)
            extraction = time.perf_counter() - start
        finally:
            meshing.sparse_marching_cubes_from_cells = sparse
            meshing.IsoSurfaceMarchingCubes.extract = dense_descriptor
        if not len(mesh.vertices):
            raise AssertionError('benchmark requires a nonempty mesh')
        seed = wp.full(len(mesh.vertices), wp.vec3(1., .3, -.2) / len(mesh.vertices),
                       dtype=wp.vec3, device=device)
        wp.synchronize_device(device)
        backward_function = instrument(tape.backward, 'full_backward') if profiled else tape.backward
        start = time.perf_counter()
        backward_function(grads={mesh.vertices: seed})
        wp.synchronize_device(device)
        backward = time.perf_counter() - start
        gp, gn = p.grad.numpy().astype(float), n.grad.numpy().astype(float)
        if not np.isfinite(gp).all() or not np.isfinite(gn).all():
            raise AssertionError('nonfinite mesh-loss gradient')
        normal_scale = float(np.sum(gn * normals))
        np.testing.assert_allclose(normal_scale, 0., atol=1e-6)
        row = dict(iteration=iteration, reconstruction_seconds=reconstruction,
                   field_to_mesh_seconds=extraction, full_backward_seconds=backward,
                   points_to_mesh_seconds=reconstruction + extraction,
                   forward_and_backward_seconds=reconstruction + extraction + backward,
                   peak_pool_bytes=wp.get_mempool_used_mem_high(device), mesh=mesh.metadata,
                   gradient_norms=dict(positions=float(np.linalg.norm(gp)), normals=float(np.linalg.norm(gn))),
                   uniform_normal_scale_derivative=normal_scale,
                   gpu_processes_before=before, gpu_processes_after=gpu_processes())
        if profiled:
            profile = {name: sum(wp.get_event_elapsed_time(a, b) / 1000 for a, b in pairs)
                       for name, pairs in events.items()}
            profile['mesh_and_field_backward'] = profile['full_backward'] - profile['reconstruction_backward']
        else:
            records.append(row)
            print(json.dumps(row), flush=True)
        del result, mesh, tape, seed, backward_function
        # Profiling wrappers retain their bound tape only during this invocation.
        if profiled:
            del record, record_func

    report = dict(input=provenance, parameters=parameters, count=len(p), bounds=bounds,
                  backend=args.backend, resolution=args.resolution, runs=records,
                  warm_medians={key: statistics.median(r[key] for r in records[1:]) for key in
                      ('reconstruction_seconds', 'field_to_mesh_seconds', 'full_backward_seconds',
                       'points_to_mesh_seconds', 'forward_and_backward_seconds')},
                  profile_device_elapsed_seconds=profile,
                  source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'])),
                  production_source_sha256=production_hashes(), warp=wp.__version__, environment=cpu_environment(),
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
                  loss='Mean vertex dot (1, 0.3, -0.2); connectivity and bounds held fixed.',
                  scope='Isolated wall time with GPU completion. Upload, bounds/seed preparation and validation excluded. Full backward includes MC, field and reconstruction to input points/normals. Peak pool includes retained tapes and excludes native CUDA graph storage.',
                  profile_scope='Separate invocation: CUDA stream elapsed time includes host submission gaps. Mesh/field backward is total minus the reconstruction callback; includes cumulative coefficient and transform VJPs. Forward field time excludes query generation/centering; MC excludes selection and coordinate mapping.',
                  first_call='First process invocation may reuse disk kernel cache.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(warm_medians=report['warm_medians'], profile=profile)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--backend', choices=['dense', 'sparse'], default='sparse')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    benchmark(args)
