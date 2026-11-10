"""SAP sensitivity experiment: cache input-independent frequency/Laplacian tensors.

The published implementation rebuilds these on the CPU in every forward call.
This variant moves exactly those operations into module setup. It is not an
unmodified upstream timing and is reported separately.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools import benchmark_gpu_baseline as baseline


original_loader = baseline.sap_class


def cached_class(root):
    original = original_loader(root)
    path = root / 'src/dpsr.py'
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'DPSR')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    statements, omega_assignments, omega_augments, lap_assignments = [], 0, 0, 0
    for node in forward.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == 'omega':
                omega_assignments += 1
                if omega_assignments == 1:
                    node.value = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                               attr='cached_omega', ctx=ast.Load())
                    statements.append(node)
                continue
            if name == 'Lap':
                lap_assignments += 1
                node.value = ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                           attr='cached_lap', ctx=ast.Load())
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == 'omega':
            omega_augments += 1
            continue
        statements.append(node)
    if (omega_assignments, omega_augments, lap_assignments) != (2, 1, 1):
        raise ValueError('Upstream FFT forward changed')
    forward.body = statements
    namespace = dict(original.forward.__globals__)
    tree = ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[]))
    exec(compile(tree, str(path), 'exec'), namespace)

    class CachedDPSR(original):
        def __init__(self, res, **kwargs):
            super().__init__(res, **kwargs)
            # Match the published CPU float32 construction/multiplication exactly.
            omega = original.forward.__globals__['fftfreqs'](res, dtype=torch.float32).unsqueeze(-1)
            omega *= 2 * np.pi
            omega = omega.cuda()
            self.register_buffer('cached_omega', omega)
            self.register_buffer('cached_lap', -torch.sum(omega**2, -2))

    CachedDPSR.forward = namespace['forward']
    return CachedDPSR


def main(args):
    baseline.source_info(args.upstream, baseline.SAP_COMMIT)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    torch.manual_seed(2026)
    # Deterministic scatter isolates the adapter from atomic splat order noise.
    torch.use_deterministic_algorithms(True)
    p = .05 + .9 * torch.rand((1, 8192, 3), device='cuda')
    n = torch.nn.functional.normalize(torch.randn_like(p), dim=-1)
    original = original_loader(args.upstream)((64, 64, 64), sig=args.sigma).cuda()
    cached = cached_class(args.upstream)((64, 64, 64), sig=args.sigma).cuda()
    expected, actual = original(p, n), cached(p, n)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    validation = dict(resolution=64, points=8192, deterministic_scatter=True,
                      bitwise_equal=True, maximum_absolute_error=float((actual - expected).abs().max().item()))
    del original, cached, expected, actual, p, n
    torch.use_deterministic_algorithms(False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline.sap_class = cached_class
    baseline.main(args)
    report = json.loads(args.output.read_text())
    report.update(variant='cached frequency and Laplacian tensors; operations moved to setup',
                  adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), validation=validation)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--depth', type=int, choices=[8, 9], required=True)
    parser.add_argument('--sigma', type=float, default=10.)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mesh', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.method = 'sap'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    main(args)
