"""Sample both independently solved fields on one dense visualization-only grid."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import warp as wp
from tools.benchmark_input import benchmark_input, reconstruction_parameters
from tools.datasets import checksum
from tools.make_fixture import ROOT, oracle_parameters, read_output
from tools.fixture_provenance import require_acceptance
from warp_psr import reconstruct


def export(directory, resolution):
    if resolution < 2:
        raise ValueError('resolution must be at least two')
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    report = require_acceptance(directory)
    p, n, parameters, provenance = benchmark_input(argparse.Namespace(fixture=directory))
    dtype = p.dtype
    vector = wp.vec3d if dtype == np.float64 else wp.vec3
    width = float(np.max(np.ptp(p, axis=0))) * parameters['scale']
    center = (p.min(axis=0).astype(np.float64)+p.max(axis=0))/2
    low, high = center-width*.499, center+width*.499
    spec = directory/f'grid-{resolution}.txt'
    spec.write_text(' '.join([str(resolution), *(format(v, '.17g') for v in [*low, *high])])+'\n')
    official_path = directory/f'official-grid-{resolution}.bin'
    with tempfile.TemporaryDirectory(dir=ROOT/'.build') as tmp:
        tmp = Path(tmp)
        (tmp/'empty.txt').write_text('')
        environment = {**os.environ, 'WARP_PSR_REFERENCE_PARALLELISM': 'none',
                       'WARP_PSR_GRID_SPEC': str(spec.resolve()), 'WARP_PSR_GRID_OUTPUT': str(official_path.resolve())}
        subprocess.run([str(ROOT/'.build/oracle'), str(directory/'input.pts'), str(tmp/'empty.txt'),
                        str(tmp/'result.txt'), *oracle_parameters(parameters)], env=environment, check=True)
        ref = read_output(tmp/'result.txt')
    # Exporting the grid must not change the official solve or isovalue.
    np.testing.assert_array_equal(ref['iso'][0, 0], report['reference_iso_value'])
    official = np.fromfile(official_path, dtype=dtype).reshape((resolution,)*3)
    actual = np.empty_like(official)
    wp.init()
    with wp.ScopedDevice('cuda'):
        result = reconstruct(wp.array(p, dtype=vector), wp.array(n, dtype=vector), **reconstruction_parameters(parameters))
        axes = [low[a]+(high[a]-low[a])*(np.arange(resolution, dtype=np.float64)/(resolution-1)) for a in range(3)]
        # Bounded query batches; the dense grid belongs only to visualization.
        for start in range(0, resolution, 4):
            end = min(start+4, resolution)
            q = np.stack(np.meshgrid(axes[0][start:end], axes[1], axes[2], indexing='ij'), axis=-1).reshape(-1, 3).astype(dtype)
            values, _, valid = result.evaluate(wp.array(q, dtype=vector))
            if not valid.numpy().all():
                raise ValueError('Visualization grid exceeds the reconstruction domain')
            actual[start:end] = values.numpy().reshape(end-start, resolution, resolution)
        iso = result.iso_value
    tolerance = {'atol': 1e-8, 'rtol': 1e-6} if dtype == np.float64 else {'atol': 1e-5, 'rtol': 1e-4}
    np.testing.assert_allclose(actual, official, **tolerance)
    np.testing.assert_allclose(actual-iso, official-ref['iso'][0, 0], **tolerance)
    actual_path = directory/f'warp-grid-{resolution}.npy'
    np.save(actual_path, actual)
    difference = actual.astype(np.float64)-official
    metadata = {**provenance, 'resolution': resolution, 'grid_queries': resolution**3, 'dtype': str(dtype),
                'model_min': low.tolist(), 'model_max': high.tolist(), 'order': 'ij, z fastest',
                'official_file': official_path.name, 'warp_file': actual_path.name,
                'official_sha256': checksum(official_path), 'warp_sha256': checksum(actual_path),
                'official_iso_value': float(ref['iso'][0, 0]), 'warp_iso_value': iso,
                'max_abs_value_error': float(np.abs(difference).max()),
                'rms_value_error': float(np.sqrt(np.mean(difference**2))),
                'tolerances': tolerance, 'field_compatibility': True,
                'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                'build': json.loads((ROOT/'.build/build.json').read_text())}
    (directory/f'grid-{resolution}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args()
    export(args.directory, args.resolution)
