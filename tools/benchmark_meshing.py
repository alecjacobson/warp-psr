"""Measure reconstruction, GPU meshing, and full points-to-mesh separately."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import warp as wp
from warp_psr import reconstruct
import warp_psr.meshing as meshing
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.cpu_environment import cpu_environment
from tools.fixture_provenance import production_hashes
from tools.make_fixture import ROOT


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                      '--format=csv,noheader'], text=True)
    entries = [line.strip() for line in output.splitlines() if line.strip()]
    foreign = [line for line in entries if line.split(',')[0].strip() != str(os.getpid())]
    if foreign:
        raise RuntimeError(f'GPU measurement requires isolation; other processes: {foreign}')
    return entries


def benchmark(args):
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError('CUDA is required for this benchmark')
    p, n, parameters, provenance = benchmark_input(args)
    if parameters['precision'] != 'float':
        raise ValueError('Warp meshing currently requires a float32 fixture')
    device = wp.get_device('cuda:0')
    initial_environment = cpu_environment()
    p, n = wp.array(p, dtype=wp.vec3, device=device), wp.array(n, dtype=wp.vec3, device=device)
    options = dict(resolution=args.resolution, backend=args.backend, slab_size=args.slab_size)
    records = []
    bounds = None
    with wp.ScopedDevice(device):
        for iteration in range(args.repeats + 1):
            gc.collect()
            processes_before = gpu_processes()
            wp.synchronize_device(device)
            begin = time.perf_counter()
            result = reconstruct(p, n, **reconstruction_parameters(parameters))
            wp.synchronize_device(device)
            reconstruction_seconds = time.perf_counter() - begin
            if bounds is None:
                inverse = np.linalg.inv(np.array(result.field.linear).reshape(3, 3))
                cube = np.array([(x, y, z) for x in (0.001, 0.999) for y in (0.001, 0.999) for z in (0.001, 0.999)])
                model = (cube - result.field.translation) @ inverse.T
                bounds = (model.min(axis=0).tolist(), model.max(axis=0).tolist())
            begin = time.perf_counter()
            mesh = meshing.extract_mesh(result, bounds=bounds, **options)
            wp.synchronize_device(device)
            meshing_seconds = time.perf_counter() - begin
            record = dict(iteration=iteration, reconstruction_seconds=reconstruction_seconds,
                          meshing_seconds=meshing_seconds,
                          points_to_mesh_seconds=reconstruction_seconds + meshing_seconds,
                          peak_device_pool_bytes=wp.get_mempool_used_mem_high(device),
                          mesh=mesh.metadata, gpu_processes_before=processes_before,
                          gpu_processes_after=gpu_processes())
            records.append(record)
            print(json.dumps(record), flush=True)
            if iteration != args.repeats:
                del mesh, result

        # Validation/downloads happen after every timed run.
        values, _, valid = result.evaluate(mesh.vertices)
        if not valid.numpy().all():
            raise AssertionError('Extracted vertices left the valid reconstruction domain')
        residual = np.abs(values.numpy().astype(np.float64) - result.iso_value)
        residual_summary = dict(max=float(residual.max(initial=0)),
                                rms=float(np.sqrt(np.mean(residual**2))) if len(residual) else 0.0)
        if args.mesh_output:
            args.mesh_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.mesh_output, vertices=mesh.vertices.numpy(), indices=mesh.indices.numpy())

        # A separate instrumented call preserves the uninstrumented warm timings.
        stages = {'field_evaluation': [], 'marching_cubes': []}

        def instrument(function, name):
            def wrapped(*positional, **keywords):
                start, end = wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)
                wp.record_event(start)
                answer = function(*positional, **keywords)
                wp.record_event(end)
                stages[name].append((start, end))
                return answer
            return wrapped

        evaluate = result.field.evaluate
        sparse = meshing.sparse_marching_cubes_from_cells
        dense_descriptor = meshing.IsoSurfaceMarchingCubes.__dict__['extract']
        dense = meshing.IsoSurfaceMarchingCubes.extract
        try:
            result.field.evaluate = instrument(evaluate, 'field_evaluation')
            meshing.sparse_marching_cubes_from_cells = instrument(sparse, 'marching_cubes')
            meshing.IsoSurfaceMarchingCubes.extract = classmethod(
                lambda cls, *a, **kw: instrument(dense, 'marching_cubes')(*a, **kw))
            profiled = meshing.extract_mesh(result, bounds=bounds, **options)
            wp.synchronize_device(device)
            del profiled
        finally:
            result.field.evaluate = evaluate
            meshing.sparse_marching_cubes_from_cells = sparse
            meshing.IsoSurfaceMarchingCubes.extract = dense_descriptor
        profile = {name: sum(wp.get_event_elapsed_time(a, b) / 1000 for a, b in events)
                   for name, events in stages.items()}

    report = dict(input=provenance, parameters=parameters, count=len(p), bounds=bounds,
                  source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                  production_source_sha256=production_hashes(), warp=wp.__version__,
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
                  execution_environment=initial_environment, execution_environment_after=cpu_environment(),
                  runs=records, warm_medians={key: statistics.median(r[key] for r in records[1:])
                      for key in ['reconstruction_seconds', 'meshing_seconds', 'points_to_mesh_seconds']},
                  profile_device_elapsed_seconds=profile, mesh_vertex_field_residual=residual_summary,
                  timing_scope='Wall time includes Python orchestration and GPU completion. Input upload, bounds preparation, validation, and mesh downloads excluded.',
                  profile_scope='Separate instrumented call: CUDA stream elapsed time includes submission gaps. Field evaluation excludes query generation/centering; MC excludes selection and final index-to-model mapping.',
                  first_run_note='First invocation in this process; may reuse existing disk kernel cache.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'warm_medians': report['warm_medians'], 'profile': profile, 'output': str(args.output)}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--backend', choices=['dense', 'sparse'], default='sparse')
    parser.add_argument('--slab-size', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh-output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    benchmark(args)
