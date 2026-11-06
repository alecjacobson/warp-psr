"""Check tracked forward parity and input-gradient invariants on accepted fixtures.

Accepts a fixture directory or a single-fixture tar.gz, without extracting archives.
These invariants complement the finite-difference unit tests; they are not an
independent gradient oracle for arbitrary large inputs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import warp as wp

from warp_psr import reconstruct, reconstruct_differentiable
from tools.benchmark_input import reconstruction_parameters
from tools.fixture_provenance import ROOT, REFERENCE_FILES, production_hashes, require_acceptance


def accepted_data(path):
    """Validate exact input/reference bytes and acceptance before any GPU work."""
    path = Path(path)
    needed = ('manifest.json', 'reconstruction_report.json', 'input.pts', 'queries.txt')
    if path.is_dir():
        require_acceptance(path)
        raw = {name: (path/name).read_bytes() for name in needed}
    else:
        raw, digests, roots = {}, {}, set()
        with tarfile.open(path, 'r|gz') as archive:
            for member in archive:
                if not member.isfile():
                    raise ValueError('Fixture archives must contain regular files only')
                parts = Path(member.name).parts
                if len(parts) != 2 or '..' in parts or Path(member.name).is_absolute():
                    raise ValueError('Expected one fixture directory with flat files')
                roots.add(parts[0])
                name = parts[1]
                if name in digests:
                    raise ValueError('Duplicate archive member')
                digest, saved = hashlib.sha256(), []
                with archive.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(chunk)
                        if name in needed:
                            saved.append(chunk)
                digests[name] = digest.hexdigest()
                if name in needed:
                    raw[name] = b''.join(saved)
        if len(roots) != 1 or set(raw) != set(needed):
            raise ValueError('Incomplete fixture archive')
        manifest = json.loads(raw['manifest.json'])
        references = manifest.get('reference_file_sha256', {})
        if not all(name in references for name in REFERENCE_FILES):
            raise ValueError('Missing reference provenance')
        expected = {**references, 'input.pts': manifest['input_sha256'],
                    'queries.txt': manifest['queries_sha256']}
        for name, digest in expected.items():
            if digests.get(name) != digest:
                raise ValueError(f'Fixture archive checksum mismatch: {name}')

    manifest = json.loads(raw['manifest.json'])
    accepted = json.loads(raw['reconstruction_report.json'])
    identity = {key: manifest[key] for key in ('input_sha256', 'queries_sha256', 'precision',
                                              'parameters', 'reference_file_sha256')}
    if (not accepted.get('reconstruction_compatibility')
            or not accepted.get('tree', {}).get('exact_active_and_support_keys')
            or accepted.get('fixture_provenance') != identity
            or accepted.get('production_source_sha256') != production_hashes()):
        raise ValueError('Stale reconstruction acceptance')
    return (manifest, np.loadtxt(io.BytesIO(raw['input.pts']), ndmin=2),
            np.loadtxt(io.BytesIO(raw['queries.txt']), ndmin=2))


def check(path, device='cuda:0'):
    manifest, data, queries = accepted_data(path)
    double = manifest['precision'] == 'double'
    vector, scalar, dtype = ((wp.vec3d, wp.float64, np.float64) if double
                             else (wp.vec3, wp.float32, np.float32))
    data, queries = data.astype(dtype), queries.astype(dtype)
    wp.init()
    device = wp.get_device(device)
    p = wp.array(data[:, :3], dtype=vector, device=device, requires_grad=True)
    n = wp.array(data[:, 3:], dtype=vector, device=device, requires_grad=True)
    q = wp.array(queries, dtype=vector, device=device)
    options = reconstruction_parameters(manifest['parameters'])
    reference = reconstruct(p, n, **options)
    expected_value, expected_gradient, expected_valid = reference.evaluate(q)
    tape = wp.Tape()
    tracked = reconstruct_differentiable(p, n, tape=tape, **options)
    with tape:
        value, gradient, valid = tracked.evaluate(q)
    for actual, expected in [(value, expected_value), (gradient, expected_gradient), (valid, expected_valid),
                             (tracked.field.coefficients, reference.field.coefficients),
                             (tracked.field.cumulative, reference.field.cumulative)]:
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    if not valid.numpy().all() or float(tracked.iso_value.numpy()[0]) != reference.iso_value:
        raise AssertionError('Invalid queries or changed tracked isovalue')

    start, end = manifest['query_families']['random_volume']
    if end <= start:
        raise ValueError('The fixture must include random-volume queries')
    indices = np.linspace(start, end-1, min(32, end-start), dtype=int)
    loss_queries = wp.array(queries[indices], dtype=vector, device=device)
    with tape:
        loss_values, spatial, _ = tracked.evaluate(loss_queries, centered=True)
    tape.backward(grads={loss_values: wp.full(len(indices), 1./len(indices), dtype=scalar, device=device)})
    gp, gn = p.grad.numpy().astype(float), n.grad.numpy().astype(float)
    if not np.isfinite(gp).all() or not np.isfinite(gn).all():
        raise AssertionError('Nonfinite input gradient')
    # Translation of every input point is equivalent to the opposite translation
    # of the query. Scaling all supplied normals does not change their directions.
    # Multiply spatial derivatives by the input extent to compare dimensionless
    # quantities across meshes with very different coordinate scales.
    extent = float(np.ptp(data[:, :3].astype(float), axis=0).max())
    actual = gp.sum(axis=0)*extent
    expected = -spatial.numpy().astype(float).mean(axis=0)*extent
    normal_scale = float(np.sum(gn*data[:, 3:].astype(float)))
    rtol, atol = (1e-7, 1e-9) if double else (1e-3, 5e-5)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    normal_atol = 1e-12 if double else 1e-6
    np.testing.assert_allclose(normal_scale, 0., atol=normal_atol)
    return {
        'fixture': manifest['name'], 'passed': True, 'precision': manifest['precision'],
        'points': len(data), 'queries': len(queries),
        'input_sha256': manifest['input_sha256'], 'queries_sha256': manifest['queries_sha256'],
        'exact_tracked_forward_values_gradients_coefficients_isovalue': True,
        'translation_dimensionless_actual': actual.tolist(), 'translation_dimensionless_expected': expected.tolist(),
        'translation_rtol': rtol, 'translation_atol': atol,
        'uniform_normal_scale_derivative': normal_scale, 'normal_scale_atol': normal_atol,
        'gradient_norms': {'positions': float(np.linalg.norm(gp)), 'normals': float(np.linalg.norm(gn))},
        'loss_query_indices': indices.tolist(), 'production_source_sha256': production_hashes(),
        'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'Fresh default/tracked forward parity on every fixture query, followed by finite input gradients '
                 'and translation/normal-scale VJP invariants for a centered mean field loss. Complements finite '
                 'differences; fixed assignments and first-order contract.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.fixture, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
