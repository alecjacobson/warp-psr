"""Explore root preconditioners/direct solves without changing production defaults.

The reference-compatible solver uses CG only for its eight-row root; fine levels
use ordered relaxation. Each reconstruction rebuilds its preconditioner or direct
factorization, so timings include setup and do not assume topology/factor reuse.
Run in an idle environment. Failed field gates remain failed in the report.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import warp as wp
import warp.sparse as sparse
from warp.optim.linear import preconditioner
import warp_psr.solver as solver
from tools.benchmark import benchmark
from tools.check_reconstruction import check

METHODS = ['reference', 'diag', 'block_jacobi_direct',
           'block_jacobi_sequential', 'block_jacobi_tile', 'fsai', 'cudss']


@contextmanager
def root_solver(method):
    """Override only this standalone process's root CG call, restoring on error."""
    original_operator, original_cg = solver.aslinearoperator, solver.cg
    matrices, calls = {}, []

    def capture_operator(matrix):
        operator = original_operator(matrix)
        # The callback must not retain every reconstruction's GPU allocation.
        matrices.clear()
        matrices[id(operator)] = matrix
        return operator

    def solve(operator, b, x, **kwargs):
        matrix = matrices.get(id(operator))
        if matrix is None or matrix.shape != (8, 8):
            raise ValueError('This experiment requires the ordinary eight-row root')
        record = {'rows': matrix.shape[0], 'method': method}
        if method == 'cudss':
            import warp_cudss
            # Preserve the full assembled matrix. The adapter owns no production
            # dependency; only the optional experiment imports cuDSS.
            direct = warp_cudss.solve(matrix, b, x, mtype='spd')
            direct.release()
            record['iterations'] = None
            result = None
        else:
            M = None
            if method.startswith('block_jacobi'):
                # Scalar CSR block-Jacobi would silently reduce to Jacobi.
                # Explicitly group the eight rows into two 4x4 blocks instead.
                blocks = sparse.bsr_copy(matrix, block_shape=(4, 4))
                M = preconditioner(blocks, method)
                record['block_shape'] = [4, 4]
            elif method != 'reference':
                M = preconditioner(matrix, method)
            result = original_cg(operator, b, x, M=M, **kwargs)
            record.update(iterations=result[0], residual=result[1], tolerance=result[2])
        calls.append(record)
        return result

    solver.aslinearoperator, solver.cg = capture_operator, solve
    try:
        yield calls
    finally:
        solver.aslinearoperator, solver.cg = original_operator, original_cg


def compare(args):
    manifest = json.loads((args.fixture / 'manifest.json').read_text())
    p = manifest['parameters']
    if p['base_depth'] == 0 or p['weight'] <= 0 or p['exact']:
        raise ValueError('Use screened fixtures with base_depth > 0 for this root experiment')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Root solver exploration; production defaults unchanged',
              'method': args.method, 'fixture': manifest['name'],
              'input_sha256': manifest['input_sha256'], 'parameters': p,
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'])),
              'warp': wp.__version__}
    with root_solver(args.method) as calls:
        try:
            with (out / 'validation.log').open('w') as log, redirect_stdout(log):
                accuracy = check(args.fixture)
            report['accuracy'] = accuracy
            report['accuracy_passed'] = True
        except (AssertionError, ValueError, RuntimeError, ImportError, OSError) as error:
            report.update(accuracy_passed=False, validation_error=str(error)[:4000])
        report['method_exercised'] = bool(calls)
        if not calls:
            report['method_note'] = ('The reference initial-residual threshold skipped root CG. '
                                     'No preconditioner or direct solver was invoked.')
        report['validation_root_calls'] = list(calls)
        calls.clear()
        if not args.validate_only:
            try:
                with (out / 'benchmark.log').open('w') as log, redirect_stdout(log):
                    report['benchmark'] = benchmark(SimpleNamespace(
                        fixture=args.fixture, repeats=args.repeats, output=out/'benchmark.json'))
                report['benchmark_root_calls'] = list(calls)
            except (ValueError, RuntimeError, ImportError, OSError) as error:
                report['benchmark_error'] = str(error)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ['method', 'method_exercised', 'accuracy_passed', 'validation_error', 'benchmark_error']}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    wp.init()
    compare(args)
