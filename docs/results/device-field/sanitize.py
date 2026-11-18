"""Check fused field evaluation and adjoints under Compute Sanitizer."""
import pytest
import warp as wp

wp.config.lineinfo = True
raise SystemExit(pytest.main([
    "tests/test_device_field.py::test_nested_device_function",
    "tests/test_device_field.py::test_fused_field_finite_differences[cuda:0-False-float32]",
    "tests/test_device_field.py::test_fused_field_finite_differences[cuda:0-True-float64]",
    "tests/test_device_field.py::test_fused_reconstruction_adjoint",
    "-q", "--disable-warnings", "--maxfail=1",
]))
