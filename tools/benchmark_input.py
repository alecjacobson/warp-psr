"""Shared, offline benchmark input selection with reproducible fixture settings."""
import hashlib
import io
import json
import numpy as np
from tools.make_fixture import points_for
from tools.datasets import checksum


def reconstruction_parameters(p):
    return {key: p[source] for key, source in {
        'depth': 'depth', 'full_depth': 'full_depth', 'base_depth': 'base_depth',
        'solve_depth': 'solve_depth', 'kernel_depth': 'kernel_depth', 'scale': 'scale',
        'samples_per_node': 'samples_per_node', 'point_weight': 'weight',
        'exact_interpolation': 'exact', 'iters': 'iters', 'base_v_cycles': 'base_v_cycles',
        'cg_accuracy': 'cg_accuracy',
    }.items()}


def benchmark_input(args):
    if args.fixture:
        manifest = json.loads((args.fixture/'manifest.json').read_text())
        path = args.fixture/'input.pts'
        if checksum(path) != manifest['input_sha256']:
            raise ValueError('Fixture input checksum no longer matches its manifest')
        parameters = manifest['parameters']
        dtype = np.float64 if manifest['precision'] == 'double' else np.float32
        data = np.loadtxt(path, ndmin=2).astype(dtype)
        provenance = {key: manifest[key] for key in ('input_sha256', 'input_source', 'input_variant', 'seed') if key in manifest}
        provenance.update(fixture=manifest['name'], shape=parameters['shape'])
        return data[:, :3], data[:, 3:], parameters, provenance
    p, n = points_for('torus', args.count, 2026)
    data = np.column_stack([p, n]).astype(np.float32)
    stream = io.StringIO()
    np.savetxt(stream, data, fmt='%.17g')
    parameters = {'depth': args.depth, 'full_depth': args.full_depth, 'base_depth': args.full_depth,
                  'solve_depth': args.depth, 'kernel_depth': max(args.depth-2, 0), 'scale': 1.1,
                  'samples_per_node': 1.5, 'weight': 2., 'exact': False, 'iters': 8,
                  'cg_accuracy': 1e-3, 'base_v_cycles': 1, 'precision': 'float'}
    provenance = {'shape': 'torus', 'seed': 2026,
                  'input_sha256': hashlib.sha256(stream.getvalue().encode()).hexdigest()}
    return data[:, :3], data[:, 3:], parameters, provenance
