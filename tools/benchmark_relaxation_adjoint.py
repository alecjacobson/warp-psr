"""Isolated costs of a real reconstruction level's finite relaxation map.

This measures a solver building block, not differentiable reconstruction.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import warp as wp
from warp_psr import reconstruct
from warp_psr.relaxation import row_schedule, relax_sliced
from warp_psr.relaxation_adjoint import differentiable_relaxation
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.benchmark_meshing import gpu_processes
from tools.fixture_provenance import production_hashes
from tools.make_fixture import ROOT


def benchmark(args):
    wp.init()
    device = wp.get_device('cuda:0')
    gpu_processes()
    p, n, parameters, provenance = benchmark_input(args)
    dtype = wp.float32 if parameters['precision'] == 'float' else wp.float64
    vec = wp.vec3 if dtype == wp.float32 else wp.vec3d
    with wp.ScopedDevice(device):
        result = reconstruct(wp.array(p, dtype=vec), wp.array(n, dtype=vec),
                             **reconstruction_parameters(parameters))
        level = result.diagnostics['levels'][-1]
        depth = level['depth']
        if depth <= result.metadata['parameters']['base_depth']:
            raise ValueError('choose a fixture with a fine cascadic level')
        matrix, rhs = level['matrix'], level['rhs']
        keys = result.diagnostics['tree'].basis[depth].keys
        iters = parameters['iters']
        schedule = row_schedule(keys, depth, sliced=True)
        tracked = args.mode == 'differentiable'
        start = time.perf_counter()
        matrix.values.requires_grad = tracked
        rhs.requires_grad = tracked
        initial = wp.zeros_like(rhs, requires_grad=tracked)
        seed = wp.ones_like(rhs, requires_grad=False)
        wp.synchronize_device(device)
        gradient_setup_seconds = time.perf_counter() - start
        base_pool_bytes = wp.get_mempool_used_mem_current(device)
        records = []
        for iteration in range(args.repeats + 1):
            gc.collect()
            before = gpu_processes()
            wp.synchronize_device(device)
            start = time.perf_counter()
            if tracked:
                with wp.Tape() as tape:
                    output = differentiable_relaxation(matrix, rhs, initial, schedule,
                        tape=tape, iters=iters, sliced_depth=depth)
            else:
                output = wp.zeros_like(rhs, requires_grad=False)
                relax_sliced(matrix, rhs, output, schedule, depth, iters)
            wp.synchronize_device(device)
            forward = time.perf_counter() - start
            retained_pool = wp.get_mempool_used_mem_current(device)
            backward = None
            if tracked:
                start = time.perf_counter()
                tape.backward(grads={output: seed})
                wp.synchronize_device(device)
                backward = time.perf_counter() - start
            record = dict(iteration=iteration, forward_seconds=forward,
                          backward_seconds=backward, retained_pool_bytes=retained_pool,
                          peak_pool_bytes=wp.get_mempool_used_mem_high(device),
                          gpu_processes_before=before, gpu_processes_after=gpu_processes())
            records.append(record)
            # Validation and clearing accumulated gradients are outside timing.
            np.testing.assert_array_equal(output.numpy(), level['coefficients'].numpy())
            if tracked:
                assert all(np.isfinite(a.grad.numpy()).all() for a in [matrix.values, rhs, initial])
                tape.zero()
                del tape
            del output
        report = dict(mode=args.mode, fixture=provenance, depth=depth, rows=len(rhs),
                      matrix_entries=matrix.nnz, iters=iters, dtype=dtype.__name__,
                      history_bytes=2 * iters * len(rhs) * (4 if dtype == wp.float32 else 8),
                      gradient_setup_seconds=gradient_setup_seconds, base_pool_bytes=base_pool_bytes,
                      runs=records, forward_bitwise_equal_to_production=True,
                      warm_forward_seconds=statistics.median(r['forward_seconds'] for r in records[1:]),
                      warm_backward_seconds=statistics.median(r['backward_seconds'] for r in records[1:]) if tracked else None,
                      production_source_sha256=production_hashes(),
                      source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      warp=wp.__version__,
                      gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
                      scope='One finest cascadic level only; reconstruction, row schedule, tracking setup, seed creation, validation and gradient reset excluded. Forward includes output/history creation; backward includes private replay buffers. Peak pool includes fixture reconstruction/setup. No full reconstruction backward is claimed.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ['runs', 'fixture', 'production_source_sha256']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--mode', choices=['baseline', 'differentiable'], required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    benchmark(args)
