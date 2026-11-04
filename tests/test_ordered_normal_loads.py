"""Exact polynomial stencils and serial reductions retain the FEM discretization."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
import pytest
import warp as wp
from warp_psr import reconstruct
from warp_psr.ordered_normal_loads import normal_integral, ordered_normal_loads
from warp_psr.normal_loads import assemble_reference_normal_loads
from warp_psr.integration import integration_grid
from warp_psr.solver import prolongation


@wp.kernel
def evaluate_integrals(args: wp.array(dtype=wp.vec4i), out: wp.array(dtype=wp.vec2d)):
    i = wp.tid()
    a = args[i]
    out[i] = normal_integral(a[0], a[2], a[1], a[3])


@pytest.mark.reference
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_normal_integrals_against_pinned_polynomial_oracle(depth):
    wp.init()
    exe = Path(".build/basis")
    if not exe.exists():
        pytest.skip("Build reference")
    rows = np.array(
        [
            list(map(float, line.split()[1:]))
            for line in subprocess.check_output(
                [str(exe), str(depth), "--normal"], text=True
            ).splitlines()
        ]
    )
    for device in ["cpu"] + (["cuda"] if wp.is_cuda_available() else []):
        args = wp.array(rows[:, :4].astype("i"), dtype=wp.vec4i, device=device)
        out = wp.empty(len(args), dtype=wp.vec2d, device=device)
        wp.launch(evaluate_integrals, len(args), inputs=[args, out], device=device)
        np.testing.assert_allclose(out.numpy(), rows[:, 4:], atol=2e-15, rtol=1e-14)
        np.testing.assert_array_equal(out.numpy().astype("f"), rows[:, 4:].astype("f"))


@pytest.mark.cuda
@pytest.mark.reference
def test_ordered_loads_preserve_fem_with_random_normal_coefficients():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    directory = Path(".build/fixtures/profile-full-zero-double")
    if not (directory / "input.pts").exists():
        pytest.skip("Generate base-zero profiles")
    data = np.loadtxt(directory / "input.pts")
    with wp.ScopedDevice("cuda"):
        result = reconstruct(
            wp.array(data[:, :3], dtype=wp.vec3d),
            wp.array(data[:, 3:], dtype=wp.vec3d),
            depth=4,
            full_depth=0,
            base_depth=0,
        )
        tree = result.diagnostics["tree"]
        rng = np.random.default_rng(4391)
        levels = []
        all_keys = []
        all_values = []
        for d, basis in enumerate(tree.basis):
            keys = basis.keys.numpy()
            c = np.stack(
                [(keys >> 32) & 65535, (keys >> 16) & 65535, keys & 65535], axis=1
            )
            keys = keys[(c < (1 << d)).all(axis=1)]
            values = np.column_stack(
                [rng.normal(size=(len(keys), 3)), np.zeros(len(keys))]
            )
            levels.append(
                SimpleNamespace(
                    keys=wp.array(keys, dtype=wp.int64),
                    values=wp.array(values, dtype=wp.vec4d),
                )
            )
            all_keys.append(keys)
            all_values.append(values)
        normals = SimpleNamespace(
            levels=levels,
            average=wp.zeros(1, dtype=wp.vec3d),
            records=SimpleNamespace(
                keys=wp.array(np.concatenate(all_keys), dtype=wp.int64),
                values=wp.array(np.concatenate(all_values), dtype=wp.vec4d),
            ),
        )
        grid = integration_grid(tree, normals, 4, dtype=wp.float64)
        transfers = [
            prolongation(tree.basis[d - 1].keys, tree.basis[d].keys, d, wp.float64)
            for d in range(1, 5)
        ]
        expected = assemble_reference_normal_loads(
            tree, grid, normals, 4, 0, transfers, ordered_root=False, stencil_loads=False
        )
        actual = ordered_normal_loads(tree, normals, 4)
        for a, e in zip(actual, expected):
            scale = np.max(np.abs(e.numpy())) or 1.0
            np.testing.assert_allclose(
                a.numpy() / scale, e.numpy() / scale, atol=1e-10, rtol=1e-9
            )
