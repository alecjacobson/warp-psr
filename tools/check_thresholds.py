"""Exercise quantized ownership and oracle-derived splat-depth transitions."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import warp as wp

from tools.check_reconstruction import check as check_reconstruction
from tools.check_tree import check_tree
from tools.datasets import checksum
from tools.fixture_provenance import verify_fixture
from tools.make_fixture import ROOT, oracle_parameters, points_for, read_output
from warp_psr.samples import prepare_samples
from warp_psr.density import build_density
from warp_psr.normals import build_normal_field
from warp_psr.tree import build_tree


def ownership_input(dtype):
    """Scale 2 and symmetric bounds make model-to-cube arithmetic exact."""
    positions = [np.full(3, -.5, dtype=dtype), np.full(3, .5, dtype=dtype)]
    normals = [[1., 0., 0.], [0., 1., 0.]]
    probes = []
    for depth in range(1, 5):
        for numerator in range(1, 1 << depth):
            split = dtype(numerator / (1 << depth))
            if not .25 < split < .75:
                continue
            for axis in range(3):
                for side, coordinate in [('below', np.nextafter(split, dtype(-np.inf))),
                                         ('exact', split),
                                         ('above', np.nextafter(split, dtype(np.inf)))]:
                    cube = np.array([.40625, .546875, .59375], dtype=dtype)
                    cube[axis] = coordinate
                    model = (cube[[2, 0, 1]] - dtype(.5)) * dtype(2)
                    probes.append({'input_index': len(positions), 'axis': axis,
                                   'split_depth': depth, 'split': float(split), 'side': side,
                                   'cube': cube.tolist()})
                    positions.append(model)
                    normals.append([0., 0., 2.])
    # One aggregate cancels exactly, another retains multiplicity, and a zero
    # normal is omitted. All positions still participate in normalization.
    for position, normal in [([.21, -.19, .23], [1, 0, 0]),
                             ([.21, -.19, .23], [-3, 0, 0]),
                             ([-.23, .19, -.21], [0, 2, 0]),
                             ([-.23, .19, -.21], [0, 5, 0]),
                             ([.11, .13, -.17], [0, 0, 0])]:
        positions.append(np.array(position, dtype=dtype))
        normals.append(normal)
    return np.column_stack([positions, normals]).astype(dtype), probes


def parameters(precision, target=1.5):
    return dict(depth=5, full_depth=2, base_depth=2, solve_depth=5,
                kernel_depth=3, scale=1.1, samples_per_node=float(target),
                weight=2., exact=False, iters=8, precision=precision,
                cg_accuracy=1e-3, base_v_cycles=1)


def discover_targets(directory, precision):
    """Bisect the official top-depth decision to adjacent representable targets."""
    dtype = np.float32 if precision == 'float' else np.float64
    directory.mkdir(parents=True, exist_ok=True)
    p, n = points_for('sphere', 512, 2026)
    np.savetxt(directory/'input.pts', np.column_stack([p, n]).astype(dtype), fmt='%.17g')
    (directory/'queries.txt').write_text('')
    history = []

    def run(target):
        env = {k: v for k, v in os.environ.items() if not k.startswith('WARP_PSR_')}
        env.update(WARP_PSR_REFERENCE_PARALLELISM='none', WARP_PSR_ORACLE_DIR=str(directory))
        subprocess.run([str(ROOT/'.build/oracle_instrumented'), str(directory/'input.pts'),
                        str(directory/'queries.txt'), str(directory/'output.txt'),
                        *oracle_parameters(parameters(precision, target))],
                       env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return read_output(directory/'splats.txt')['splat']

    baseline = run(dtype(1.5))
    index = int(np.argmax(baseline[:, 3]))
    boundary = 3
    guess = dtype(1.5 * 4 ** (baseline[index, 3] - boundary))

    def decision(target):
        row = run(target)[index]
        history.append({'target': float(target), 'raw_depth': float(row[3]), 'top_depth': int(row[5])})
        return row[5] > boundary

    low, high = dtype(guess/2), dtype(guess*2)
    if not decision(low) or decision(high):
        raise ValueError('Oracle-derived density bracket does not straddle the requested depth')
    while np.nextafter(low, dtype(np.inf)) < high:
        mid = dtype(low + dtype((high-low)/dtype(2)))
        if not low < mid < high:
            raise ValueError('Density bisection failed to make progress')
        if decision(mid):
            low = mid
        else:
            high = mid
    assert decision(low) and not decision(high)
    report = {'precision': precision, 'input_sha256': checksum(directory/'input.pts'),
              'aggregate_index': index, 'cube_position': baseline[index, :3].tolist(),
              'integer_depth': boundary, 'adjacent_targets': [float(low), float(high)],
              'search': history, 'method': 'Bisection of the pinned serial oracle top-depth decision; adjacent scalar values'}
    (directory/'discovery.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def field_keys(rows):
    d, x, y, z = rows[:, :4].astype(np.int64).T
    return (d << 48) | (x << 32) | (y << 16) | z


def check_preprocessing(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    verify_fixture(directory, manifest)
    p = manifest['parameters']
    double = manifest['precision'] == 'double'
    dtype, vector = (np.float64, wp.vec3d) if double else (np.float32, wp.vec3)
    atol, rtol = (1e-10, 1e-9) if double else (1e-6, 1e-5)
    data = np.loadtxt(directory/'input.pts').astype(dtype)
    pre = read_output(directory/'preprocess.txt')
    stages = read_output(directory/'stages.txt')
    samples = prepare_samples(wp.array(data[:, :3], dtype=vector, device='cuda'),
                              wp.array(data[:, 3:], dtype=vector, device='cuda'),
                              depth=p['depth'], scale=p['scale'])
    order = np.argsort(samples.first_indices.numpy())
    np.testing.assert_array_equal(samples.keys.numpy()[order], field_keys(pre['sample'][:, 1:5]))
    np.testing.assert_array_equal(samples.weights.numpy()[order], pre['sample'][:, 8])
    np.testing.assert_allclose(samples.position_sums.numpy()[order], pre['sample'][:, 5:8], atol=atol, rtol=rtol)
    np.testing.assert_allclose(samples.normal_sums.numpy()[order], pre['sample'][:, 9:12], atol=atol, rtol=rtol)
    density = build_density(samples, p['depth'], p['kernel_depth'], p['samples_per_node'])
    for d, level in enumerate(density.levels):
        rows = pre['density'][pre['density'][:, 0] == d]
        rows = rows[np.argsort(field_keys(rows))]
        np.testing.assert_array_equal(level.keys.numpy(), field_keys(rows))
        np.testing.assert_allclose(level.values.numpy()[:, 0], rows[:, 4], atol=atol, rtol=rtol)
    live = np.linalg.norm(samples.normal_sums.numpy()[order], axis=1) > 0
    splats = read_output(directory/'splats.txt')['splat']
    dw = density.depths_and_weights.numpy()[order][live]
    np.testing.assert_allclose(dw, splats[:, 3:5], atol=atol, rtol=rtol)
    top = np.ceil(np.clip(dw[:, 0], p['base_depth'], p['depth'])).astype(int)
    np.testing.assert_array_equal(top, splats[:, 5], err_msg='Discrete splat depth differs')
    normals = build_normal_field(samples, density, p['base_depth'], p['depth'])
    rows = stages['normal'][np.argsort(field_keys(stages['normal']))]
    normal_keys, normal_values = normals.records.keys.numpy(), normals.records.values.numpy()[:, :3]
    expected_keys = field_keys(rows)
    shared, actual_indices, expected_indices = np.intersect1d(normal_keys, expected_keys, return_indices=True)
    # The reference export follows tree finalization, which removes some zero
    # stencil entries. Prove that different storage contains only exact zeros;
    # retain exact comparisons of the final support/active tree below.
    extra = ~np.isin(normal_keys, shared)
    missing = ~np.isin(expected_keys, shared)
    np.testing.assert_array_equal(normal_values[extra], 0)
    np.testing.assert_array_equal(rows[missing, 4:], 0)
    np.testing.assert_allclose(normal_values[actual_indices], rows[expected_indices, 4:], atol=atol, rtol=rtol)
    tree = build_tree(normals, p['depth'], p['full_depth'], p['base_depth'])
    tree_report = check_tree(tree, read_output(directory/'oracle_instrumented.txt')['node'])
    return {'accepted_samples': int(samples.weights.numpy().sum()), 'aggregates': len(order),
            'cancelled_normal_aggregates': int((~live).sum()), 'splats': len(splats),
            'extra_zero_normal_records': int(extra.sum()), 'missing_zero_normal_records': int(missing.sum()),
            'exact_aggregate_keys_and_weights': True, 'exact_top_depths': True,
            'max_raw_depth_error': float(np.abs(dw[:, 0]-splats[:, 3]).max()), 'tree': tree_report}


def main(args):
    wp.init()
    reports, discoveries = [], []
    for precision in ('float', 'double'):
        dtype = np.float32 if precision == 'float' else np.float64
        source = ROOT/'.build/threshold-inputs'/precision
        source.mkdir(parents=True, exist_ok=True)
        data, probes = ownership_input(dtype)
        np.savetxt(source/'ownership.pts', data, fmt='%.17g')
        (source/'ownership.json').write_text(json.dumps(probes, indent=2)+'\n')
        discovery = discover_targets(source/'density', precision)
        discoveries.append(discovery)
        cases = [(f'ownership-{method}', ['--input', str(source/'ownership.pts'),
                  '--scale', '2', '--weight', str(weight)]) for method, weight in [('psr', 0), ('spsr', 2)]]
        cases += [(f'density-{side}', ['--samples-per-node', str(target)])
                  for side, target in zip(('below', 'above'), discovery['adjacent_targets'])]
        for case, options in cases:
            name = f'threshold-{case}-{precision}'
            directory = ROOT/'.build/fixtures'/name
            directory.mkdir(parents=True, exist_ok=True)
            entry = {'fixture': name, 'passed': False, 'stage': 'fixture_generation'}
            with (directory/'threshold.log').open('w') as log, contextlib.redirect_stdout(log):
                try:
                    subprocess.run([sys.executable, 'tools/make_fixture.py', '--name', name,
                                    '--precision', precision, '--depth', '5', '--full-depth', '2',
                                    '--grid', '8', *options], cwd=ROOT, check=True,
                                   stdout=log, stderr=subprocess.STDOUT)
                    entry['stage'] = 'preprocessing'
                    entry['preprocessing'] = [check_preprocessing(directory) for _ in range(args.repeats)]
                    entry['stage'] = 'reconstruction'
                    entry['repeated_reconstruction_acceptance'] = []
                    for _ in range(args.repeats):
                        reconstruction = check_reconstruction(directory)
                        entry['repeated_reconstruction_acceptance'].append({key: reconstruction[key] for key in
                            ('max_abs_value_error', 'max_abs_centered_value_error',
                             'max_scaled_gradient_budget_ratio', 'iso_value', 'coefficients')})
                    entry['reconstruction'] = reconstruction
                    entry['manifest'] = json.loads((directory/'manifest.json').read_text())
                    entry['passed'] = True
                except (AssertionError, ValueError, subprocess.CalledProcessError) as error:
                    entry['failure'] = str(error)
                    traceback.print_exc(file=log)
            reports.append(entry)
            report = {'scope': 'Quantized ownership and adjacent oracle-derived density thresholds in both precisions',
                      'repeats': args.repeats, 'all_passed': all(item['passed'] for item in reports),
                      'discoveries': discoveries, 'profiles': reports}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(name, 'PASS' if entry['passed'] else f"FAIL ({entry.get('stage', 'fixture')})", flush=True)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT/'.build/threshold-validation.json')
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('Use at least two preprocessing repeats')
    raise SystemExit(main(args))
