# Read-only solver traces

The instrumented offline reference can export every regular-grid relaxation pass,
every sliced relaxation visit, the root CG recurrence and stopping decision, and
final level coefficients. `tools.check_solver_trace` independently replays those
updates using serial NumPy scalar arithmetic in the input precision. It requires
bitwise equality with the recorded updates and the final node coefficients.
These exports never enter production reconstruction.

Generate a small example after building the pinned reference:

```sh
python tools/build_reference.py
python tools/make_fixture.py --name solver-trace-tight-float --shape sphere \
  --count 128 --depth 3 --full-depth 2 --iters 3 --grid 8 --precision float \
  --cg-accuracy 1e-8 --operators --solver-trace
python -m tools.check_solver_trace .build/fixtures/solver-trace-tight-float
python -m tools.check_reconstruction .build/fixtures/solver-trace-tight-float
```

Use a new fixture name for each parameter set. Tracing is opt-in and intended for
small diagnostic cases: each relaxation record includes its matrix, actual RHS,
diagonal reciprocal, colors, and neighboring coefficients. The fixture manifest
hashes the trace alongside its ordinary reference exports. The unchanged and
instrumented drivers must agree bitwise on all query values, gradients, isovalue,
and transform before fixture generation succeeds.

Hooks are inserted only into copied reference headers. They read the matrices,
constraints and coefficients already used by the solver. They do not enable
`showResidual` or `computeNorms`; those options change the sliced matrix window.
The uninstrumented binary continues to use the original pinned headers. The build
cache includes both patched solver headers and the tracing helper.

## Trace format

`solve_trace.txt` begins with `format 1`. Each `event` has a consecutive sequence
number, a type and its context. Following lines belong to that event until the
next event. Floating-point numbers use 17 significant digits.

| Event | Context and data |
| --- | --- |
| `solve` | Base/minimum/maximum depths, cycle settings, CG accuracy, initial coefficients and constraints. |
| `gs` | Depth, restriction/prolongation/slice phase, pass number or slice coordinate, global row offset, color direction and row count. Includes ordered sparse rows, reciprocal diagonal, RHS, colors, coefficient halo, and before/after values. |
| `root` | Root depth, base cycle, global row offset, constant-mode flag and the actual Galerkin matrix. |
| `cg_initial` | Dimension, iteration limit, squared absolute threshold and initial squared residual, with RHS, coefficients and residual. |
| `cg_iteration` | Iteration, old/new squared residuals, direction product, alpha, beta and squared relative threshold, with updated coefficients, residual and direction. |
| `cg_stop` | Initial-threshold skip, relative tolerance, iteration limit or breakdown, with actual iteration count and residual/threshold. |
| `level` | Completed restriction/prolongation level, outer cycle and coefficients indexed by the canonical reference node IDs. |
| `end` | The solve completed and the stream closed. |

Sparse columns are relative to the event's row offset. Sliced matrices can refer
to negative column indices; the recorded halo supplies those neighboring values.
The checker preserves matrix-entry and color order rather than replacing the
updates with a generic dense solve. Its bitwise replay claim concerns the serial
reference trace; production GPU comparisons retain their fixed numerical budgets.

The root solver first skips when `residual_squared <= accuracy_squared`. If it
runs, its stopping threshold becomes `accuracy_squared * initial_residual_squared`.
The [root-solver experiments](solver-experiments.md) show why a tighter diagnostic
fixture is needed: the default eagle skips root CG entirely. A stop at the
iteration cap is recorded as a limit, not convergence.

CPU reference CI generates fresh float32/float64 traces with loose, default and tight CG
accuracy, checks instrumentation neutrality and independently replays all updates.

The [six-fixture report](results/solver-traces.json) records 39 relaxation updates
per fixture, exact replay, untouched-reference neutrality, CG stop reasons and
fixed-budget GPU acceptance. The loose cases skip CG, default cases stop after
six iterations, and tight cases use eight iterations. The tight float32 case
hits its iteration cap; the float64 case meets its threshold.
