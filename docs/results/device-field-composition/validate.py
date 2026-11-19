"""Replay ordinary-helper validation with the selected Warp installation."""
import argparse
from pathlib import Path
import sys
import tempfile

import pytest
import warp._src.codegen as codegen

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("full", "memcheck", "wheel-policy"), default="full")
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tests"))

if args.mode == "wheel-policy":
    # Experimental code-generation injection; do not modify the installed wheel.
    original = codegen.codegen_func

    def patched(adj, c_func_name, device="cpu", options=None, forward_only=False,
                reverse_only=False, inline_hint=None):
        if device == "cuda" and inline_hint is None and any(
                isinstance(arg.type, codegen.Struct) for arg in adj.args):
            inline_hint = "forceinline"
        return original(adj, c_func_name, device=device, options=options,
                        forward_only=forward_only, reverse_only=reverse_only, inline_hint=inline_hint)

    codegen.codegen_func = patched
elif not hasattr(codegen, "get_func_inline_hint"):
    raise RuntimeError("Run against the recorded Warp fix commit for this validation")

source = (root / "tests/test_device_field.py").read_text()
replacements = {
    "    @wp.kernel\n    def query_kernel": """    @wp.func
    def composed_query(field: field_type, point: vector):
        return evaluate_field(field, point)

    @wp.kernel
    def query_kernel""",
    "value, gradient, ok = evaluate_field(field, queries[i])":
        "value, gradient, ok = composed_query(field, queries[i])",
    "@wp.func(inline=True)\ndef nested_query": "@wp.func\ndef nested_query",
}
for before, after in replacements.items():
    if source.count(before) != 1:
        raise ValueError("The original test changed; review the helper substitution before rerunning")
    source = source.replace(before, after)

with tempfile.TemporaryDirectory(prefix="psr-composition-") as directory:
    target = Path(directory) / ("test_composition_" + args.mode.replace("-", "_") + ".py")
    target.write_text(source)
    if args.mode == "full":
        cases = [str(root / "tests"), str(target)]
    else:
        names = ["test_nested_device_function",
                 "test_fused_field_finite_differences[cuda:0-False-float32]",
                 "test_fused_field_finite_differences[cuda:0-True-float64]"]
        if args.mode == "memcheck":
            names.append("test_fused_reconstruction_adjoint")
        cases = [str(target) + "::" + name for name in names]
    raise SystemExit(pytest.main(cases + ["-q", "--disable-warnings", "--maxfail=1"]))
