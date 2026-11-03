"""Quantify quota-aware OpenMP reference variability against serial acceptance fixtures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import warp as wp
from reference.adapter import inverse_kernel
from tools.datasets import checksum
from reference.cells import query_cell_widths
from tools.benchmark_input import benchmark_input
from tools.cpu_environment import cpu_environment, available_threads
from tools.check_repeatability import errors, digest_keys, reference_keys
from tools.make_fixture import ROOT, oracle_parameters, read_output


def support(nodes):
    rows = nodes[(nodes[:, 0] >= 0) & (nodes[:, 5] == 0), :4].astype(np.int64)
    return rows[np.lexsort(rows.T[::-1])]


def main(args):
    wp.init()
    environment = cpu_environment()
    threads = args.threads or available_threads(environment)
    entries = []
    for name in args.fixtures:
        directory = ROOT/'.build/fixtures'/name
        _, _, parameters, provenance = benchmark_input(argparse.Namespace(fixture=directory))
        baseline = read_output(directory/'oracle_instrumented.txt')
        normal = read_output(directory/'stages.txt')['normal']
        manifest = json.loads((directory/'manifest.json').read_text())
        if checksum(directory/'queries.txt') != manifest['queries_sha256']:
            raise ValueError('Fixture query checksum mismatch')
        dtype = wp.float64 if parameters['precision'] == 'double' else wp.float32
        mat = wp.types.matrix(shape=(4, 4), dtype=dtype)
        transform = baseline['transform'].reshape(4, 4).T
        source = wp.array(transform[None], dtype=mat, device='cpu')
        inverse = wp.empty_like(source)
        wp.launch(inverse_kernel(dtype), 1, inputs=[source, inverse], device='cpu')
        model_to_cube = inverse.numpy()[0]
        queries = np.loadtxt(directory/'queries.txt').astype(np.float64 if dtype == wp.float64 else np.float32)
        cube = queries@model_to_cube[:3, :3].T+model_to_cube[:3, 3]
        widths = query_cell_widths(baseline['node'], normal, cube, parameters['depth'])
        inverse_linear = np.linalg.inv(model_to_cube[:3, :3])
        atol, rtol = (1e-8, 1e-6) if parameters['precision'] == 'double' else (1e-5, 1e-4)
        runs = []
        first = None
        output = ROOT/'.build/openmp-validation'/name
        output.mkdir(parents=True, exist_ok=True)
        for i in range(args.repeats):
            path = output/f'run-{i}.txt'
            env = {k: v for k, v in os.environ.items() if not k.startswith('WARP_PSR_')}
            env.update(WARP_PSR_REFERENCE_PARALLELISM='openmp', OMP_THREAD_LIMIT=str(threads),
                       OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE')
            subprocess.run([str(ROOT/'.build/oracle_instrumented_openmp'), str(directory/'input.pts'),
                            str(directory/'queries.txt'), str(path), *oracle_parameters(parameters)], env=env, check=True)
            data = read_output(path)
            if first is None:
                first = data
            active_equal = digest_keys(reference_keys(data['node'])) == digest_keys(reference_keys(baseline['node']))
            support_equal = np.array_equal(support(data['node']), support(baseline['node']))
            comparison = errors(data['q'], data['iso'][0, 0], baseline['q'], baseline['iso'][0, 0], inverse_linear, widths, atol, rtol)
            repeatability = errors(data['q'], data['iso'][0, 0], first['q'], first['iso'][0, 0], inverse_linear, widths, atol, rtol)
            runs.append({'against_serial': comparison, 'against_first_openmp': repeatability,
                         'exact_active_keys': active_equal, 'exact_support_keys': support_equal,
                         'transform_matches': np.array_equal(data['transform'], baseline['transform']),
                         'effective_threads': int(data['openmp_effective_threads'][0, 0])})
        passed = all(r['against_serial']['within_budget'] and r['against_first_openmp']['within_budget']
                     and r['exact_active_keys'] and r['exact_support_keys'] and r['transform_matches']
                     and r['effective_threads'] == threads for r in runs)
        entries.append({**provenance, 'fixture': name, 'parameters': parameters, 'runs': runs, 'passed': passed})
        print(name, 'PASS' if passed else 'FAIL', flush=True)
    report = {'scope': 'OpenMP compatibility is separate from the unchanged serial/GPU acceptance budgets; repeatability requires multiple runs',
              'repeats':args.repeats,'repeatability_measured':args.repeats>1,
              'execution_environment': environment, 'requested_threads': threads,
              'build': json.loads((ROOT/'.build/build.json').read_text()),
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
              'all_passed': all(r['passed'] for r in entries), 'fixtures': entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', nargs='+', default=['eagle-10k-d7-spsr-float', 'torso-10k-d7-psr-double'])
    parser.add_argument('--threads', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT/'.build/openmp-validation.json')
    args = parser.parse_args()
    if args.repeats < 1 or (args.threads is not None and args.threads < 1):
        parser.error('Require at least one run and positive thread count')
    raise SystemExit(main(args))
