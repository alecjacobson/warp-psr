"""Reproduce public-cloud accuracy and matched timing experiments in fresh processes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tools.make_fixture import ROOT


def cases(suite):
    entries = []
    if suite in ('smoke', 'all'):
        for dataset in ('horse', 'bunny', 'eagle', 'torso'):
            for precision in ('float', 'double'):
                for method, weight in (('psr', 0), ('spsr', 2)):
                    entries.append((f'{dataset}-10k-d7-{method}-{precision}', dataset,
                                    ['--max-points', '10000', '--depth', '7', '--full-depth', '3',
                                     '--precision', precision, '--weight', str(weight)]))
    if suite in ('scaling', 'all'):
        for dataset in ('horse', 'bunny', 'eagle', 'torso'):
            for method, weight in (('psr', 0), ('spsr', 2)):
                entries.append((f'{dataset}-full-d8-{method}', dataset,
                                ['--depth', '8', '--full-depth', '4', '--weight', str(weight)]))
        for count in (10000, 100000, 1000000):
            entries.append((f'torso-{count}-d8', 'torso', ['--max-points', str(count), '--depth', '8', '--full-depth', '4']))
        for target in (3, 10):
            entries.append((f'eagle-full-d8-density{target}', 'eagle',
                            ['--depth', '8', '--full-depth', '4', '--samples-per-node', str(target)]))
        entries.append(('eagle-full-d9-spsr', 'eagle', ['--depth', '9', '--full-depth', '4']))
        entries.append(('torso-1m-d9-spsr', 'torso', ['--max-points', '1000000', '--depth', '9', '--full-depth', '4']))
    return entries


def main(args):
    entries = cases(args.suite)
    if args.case:
        names = set(args.case)
        entries = [entry for entry in entries if entry[0] in names]
        if names != {entry[0] for entry in entries}:
            raise ValueError('Unknown case name for the selected suite')
    reports = []
    for name, dataset, options in entries:
        directory = ROOT/'.build/fixtures'/name
        directory.mkdir(parents=True, exist_ok=True)
        if args.benchmark_only:
            manifest=json.loads((directory/'manifest.json').read_text())
            validation=json.loads((directory/'reconstruction_report.json').read_text())
            actual_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ROOT.glob('src/**/*.py'))}
            if (not validation['reconstruction_compatibility'] or not validation['tree']['exact_active_and_support_keys']
                    or validation['production_source_sha256'] != actual_hashes
                    or validation['input_sha256'] != manifest['input_sha256']):
                raise ValueError(f'{name}: regenerate accuracy results for the current production source/input')
            expected={'dataset':dataset,'grid':24,'seed':2026,'precision':'float','weight':2.,'samples_per_node':1.5}
            for option,value in zip(options[::2],options[1::2]):
                key=option[2:].replace('-','_')
                expected[key]=type(manifest['parameters'][key])(value)
            if any(manifest['parameters'][key] != value for key,value in expected.items()):
                raise ValueError(f'{name}: fixture parameters differ from the selected case')
        commands = [[sys.executable, 'tools/make_fixture.py', '--name', name, '--dataset', dataset, '--grid', '24', *options],
                    [sys.executable, '-m', 'tools.check_reconstruction', str(directory)]]
        if args.benchmark_only:
            commands=[]
        if args.benchmark:
            for module, suffix, extra in [('tools.benchmark', 'gpu', []),
                                          ('tools.benchmark_reference', 'serial', ['--parallelism', 'none']),
                                          ('tools.benchmark_reference', 'openmp', ['--parallelism', 'openmp'])]:
                commands.append([sys.executable, '-m', module, '--fixture', str(directory),
                                 '--output', str(directory/f'benchmark-{suffix}.json'), *extra])
        log_path=directory/('dataset-benchmark.log' if args.benchmark_only else 'dataset-validation.log')
        with log_path.open('w') as log:
            for command in commands:
                print(' '.join(command), file=log, flush=True)
                status = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
                if status:
                    break
        entry = {'fixture': name, 'passed': status == 0}
        if status:
            entry.update(failed_command=command, exit_code=status,
                         log_tail=log_path.read_text()[-5000:])
        else:
            entry['manifest'] = json.loads((directory/'manifest.json').read_text())
            entry['reconstruction'] = json.loads((directory/'reconstruction_report.json').read_text())
            if args.benchmark:
                entry['benchmarks'] = {suffix: json.loads((directory/f'benchmark-{suffix}.json').read_text())
                                       for suffix in ('gpu', 'serial', 'openmp')}
                hashes = {item['input_sha256'] for item in entry['benchmarks'].values()}
                assert hashes == {entry['manifest']['input_sha256']}
        reports.append(entry)
        report = {'scope': 'Public datasets from the pinned PoissonRecon README; fields, exact active/support keys, optional matched timing',
                  'suite': args.suite, 'all_passed': all(r['passed'] for r in reports), 'profiles': reports}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(name, 'PASS' if not status else f'FAIL (see {log_path})', flush=True)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['smoke', 'scaling', 'all'], default='smoke')
    parser.add_argument('--case', action='append', help='Run only named cases (repeatable)')
    parser.add_argument('--benchmark', action='store_true', help='Fresh-process GPU, serial C++, quota-aware OpenMP C++ timing for each passing case')
    parser.add_argument('--benchmark-only',action='store_true',help='Time existing passing fixtures after verifying production source hashes and case settings')
    parser.add_argument('--output', type=Path, default=ROOT/'.build/dataset-validation.json')
    args=parser.parse_args()
    if args.benchmark_only:args.benchmark=True
    raise SystemExit(main(args))
