"""Test NG-style inverse-diagonal reuse without changing production arithmetic/order."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import warp as wp
import warp_psr.relaxation as relaxation
import warp_psr.solver as solver
from warp_psr import reconstruct
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.cpu_environment import cpu_environment
from tools.benchmark_gpu_baseline import gpu_processes


def candidate(directory):
    # Generate an isolated experimental module from the current production code.
    # The two graph update kernels keep their exact residual sum and row order.
    source = Path(relaxation.__file__).read_text()
    source = source.replace('"x": wp.array(dtype=dtype),',
                            '"x": wp.array(dtype=dtype), "diagonal": wp.array2d(dtype=dtype),')
    old = '''            diagonal = dtype(0.0)
            for k in range(state.offsets[i], state.offsets[i + 1]):
                col = state.columns[k]
                r -= state.x[col] * state.values[k]
                if col == i:
                    diagonal = state.values[k]
            if diagonal != dtype(0.0):
                state.x[i] += r * (dtype(1.0) / diagonal)'''
    new = '''            for k in range(state.offsets[i], state.offsets[i + 1]):
                col = state.columns[k]
                r -= state.x[col] * state.values[k]
            if state.diagonal[i, 0] != dtype(0.0):
                state.x[i] += r * state.diagonal[i, 1]'''
    assert source.count(old) == 2
    source = source.replace(old, new)
    source = source.replace('    state.b, state.x = b, x', '''    diagonal = wp.empty((A.nrow, 2), dtype=b.dtype, device=x.device)
    wp.launch(diagonal_kernel(b.dtype), A.nrow,
              inputs=[A.offsets, A.columns, A.values, diagonal], device=x.device)
    state.b, state.x, state.diagonal = b, x, diagonal''')
    source += '''
@lru_cache(None)
def diagonal_kernel(dtype):
    @wp.kernel
    def compute(offsets: wp.array(dtype=int), columns: wp.array(dtype=int),
                values: wp.array(dtype=dtype), diagonal: wp.array2d(dtype=dtype)):
        i = wp.tid()
        d = dtype(0.0)
        for k in range(offsets[i], offsets[i + 1]):
            if columns[k] == i:
                d = values[k]
        diagonal[i, 0] = d
        diagonal[i, 1] = dtype(0.0)
        if d != dtype(0.0):
            diagonal[i, 1] = dtype(1.0) / d
    return compute
'''
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "relaxation_diagonal.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("warp_psr._diagonal_experiment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source.encode()).hexdigest()


def main(args):
    wp.init()
    wp.set_device("cuda:0")
    module, sha = candidate(args.output.parent)
    p, n, parameters, provenance = benchmark_input(args)
    parameters.update(depth=args.depth, solve_depth=args.depth, kernel_depth=args.depth - 2)
    vector = wp.vec3d if parameters['precision'] == 'double' else wp.vec3
    p, n = (wp.array(a, dtype=vector) for a in (p, n))
    methods = {"production": relaxation.relax_sliced, "cached_diagonal": module.relax_sliced}
    options = reconstruction_parameters(parameters)
    reference = None
    records = []
    environment = cpu_environment()
    try:
        # Warm each path, then ABBA blocks to reduce ordering/clock bias.
        order = list(methods) * 2 + ["production", "cached_diagonal", "cached_diagonal", "production"] * args.blocks
        for i, name in enumerate(order):
            solver.relax_sliced = methods[name]
            gc.collect()
            gpu_processes()
            wp.synchronize()
            start = time.perf_counter()
            result = reconstruct(p, n, **options)
            wp.synchronize()
            elapsed = time.perf_counter() - start
            arrays = (result.field.keys.numpy(), result.field.coefficients.numpy(), result.iso_value)
            if reference is None:
                reference = arrays
            else:
                for a, b in zip(reference, arrays):
                    np.testing.assert_array_equal(a, b)
            gpu_processes()
            record = dict(iteration=i, method=name, warm=i >= 4, seconds=elapsed,
                          bitwise_equal_keys_coefficients_isovalue=True)
            records.append(record)
            print(json.dumps(record), flush=True)
            del result
    finally:
        solver.relax_sliced = relaxation.relax_sliced
    report = dict(**provenance, parameters=parameters, generated_candidate_sha256=sha,
                  production_relaxation_sha256=hashlib.sha256(Path(relaxation.__file__).read_bytes()).hexdigest(),
                  environment=environment, environment_after=cpu_environment(), records=records,
                  medians={name: statistics.median(r["seconds"] for r in records if r["warm"] and r["method"] == name)
                           for name in methods})
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("blocks must be positive")
    main(args)
