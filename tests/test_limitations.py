"""Minimal upstream limitation reproducers; xfail is not a compatibility pass."""
import pytest
import warp as wp
from warp_psr.forms import uniform_space, assemble_stiffness


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason="Issue #8: Warp float64 FEM gradient type mismatch")
def test_float64_fem_gradient_reproducer():
    with wp.ScopedDevice("cpu"):
        assemble_stiffness(uniform_space(1, wp.float64), "cpu")
