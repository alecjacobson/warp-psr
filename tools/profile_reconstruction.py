"""Profile one warmed reconstruction; profiling durations are not benchmarks.

For the complete CUDA timeline, including CUB and graph nodes, run under Nsight
Systems with --capture-range=cudaProfilerApi --cuda-graph-trace=node and select
--mode nsight. CPU mode writes cProfile data and a cumulative-time table.
"""
import argparse
import cProfile
import ctypes
import gc
import json
from pathlib import Path
import pstats
import subprocess
import time
import warp as wp
from warp_psr import reconstruct
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.fixture_provenance import production_hashes
from tools.make_fixture import ROOT


def profile(args):
    wp.init()
    positions, normals, parameters, provenance = benchmark_input(args)
    vector = wp.vec3d if parameters['precision'] == 'double' else wp.vec3
    options = reconstruction_parameters(parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wp.ScopedDevice('cuda'):
        positions = wp.array(positions, dtype=vector)
        normals = wp.array(normals, dtype=vector)
        for _ in range(args.warmups):
            result = reconstruct(positions, normals, **options)
            del result
            gc.collect()
            wp.synchronize()
        cpu = cProfile.Profile() if args.mode == 'cpu' else None
        cuda = ctypes.CDLL('libcuda.so.1') if args.mode == 'nsight' else None
        if cuda and cuda.cuProfilerStart() != 0:
            raise RuntimeError('CUDA profiler start failed')
        if cpu:
            cpu.enable()
        start = time.perf_counter()
        try:
            result = reconstruct(positions, normals, **options)
            wp.synchronize()
            seconds = time.perf_counter() - start
        finally:
            if cpu:
                cpu.disable()
            if cuda and cuda.cuProfilerStop() != 0:
                raise RuntimeError('CUDA profiler stop failed')
    if cpu:
        cpu.dump_stats(str(args.output.with_suffix('.prof')))
        with args.output.with_suffix('.txt').open('w') as output:
            pstats.Stats(cpu, stream=output).sort_stats('cumulative').print_stats(80)
    report = {
        **provenance, 'parameters': parameters, 'mode': args.mode,
        'warmups': args.warmups, 'profiled_seconds': seconds,
        'scope': 'One warmed reconstruction including profiler overhead; use tools.benchmark for timings',
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'production_source_sha256': production_hashes(), 'metadata': result.metadata,
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Profiled reconstruction: {seconds:.4f} s (includes profiler overhead)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--mode', choices=['cpu', 'nsight'], default='cpu')
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('.build/reconstruction-profile'))
    args = parser.parse_args()
    if args.warmups < 1:
        parser.error('warmups must be positive')
    profile(args)
