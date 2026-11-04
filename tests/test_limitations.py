"""Minimal upstream limitation reproducers; xfail is not a compatibility pass."""
import pytest
import warp as wp
import warp.fem as fem
from warp._src.codegen import WarpCodegenError
from warp_psr.forms import uniform_space, assemble_stiffness


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Issue #8: Warp float64 FEM gradient type mismatch")
def test_float64_fem_gradient_reproducer():
    with wp.ScopedDevice("cpu"):
        assemble_stiffness(uniform_space(1, wp.float64), "cpu")


@pytest.mark.xfail(strict=True, raises=WarpCodegenError,
                   reason="Issue #66: float64 Grid3D particle lookup receives float32 max_dist")
def test_float64_grid_particle_lookup_reproducer():
    with wp.ScopedDevice("cpu"):
        grid = fem.Grid3D(wp.vec3i(2), scalar_type=wp.float64)
        points = wp.array([[.25, .25, .25]], dtype=wp.vec3d)
        fem.PicQuadrature(fem.Cells(grid), points, measures=wp.ones(1, dtype=wp.float64))
