# Differentiability work

Reconstruction does **not** yet differentiate input positions or normals.
[Issue #76](https://github.com/alecjacobson/warp-psr/issues/76) tracks that complete
path. Spatial field gradients returned by `evaluate` are a different quantity.
[Mesh interpolation gradients](meshing.md#differentiation) already work for
tracked callbacks and directly specified field coefficients.

## Finite ordered relaxation

The experimental `warp_psr.relaxation_adjoint.differentiable_relaxation` is a
building block for the reconstruction backward pass. It differentiates the
actual finite ordered Gauss–Seidel updates with respect to scalar CSR matrix
values, the right-hand side, and the initial guess. It supports float32/float64,
CPU/CUDA, regular forward/reverse color sweeps, and degree-one sliced sweeps.

```python
from warp_psr.relaxation_adjoint import differentiable_relaxation

# A, rhs, initial and schedule are already constructed on the same device.
# Track only the inputs whose gradients are needed.
A.values.requires_grad = True
rhs.requires_grad = True
initial.requires_grad = True
with wp.Tape() as tape:
    coefficients = differentiable_relaxation(
        A, rhs, initial, schedule, tape=tape, iters=8, sliced_depth=depth
    )
    # A downstream differentiable operation can consume coefficients here.
tape.backward(grads={coefficients: seed})
# A.values.grad, rhs.grad and initial.grad contain vector-Jacobian products.
```

This is the derivative of the incomplete iteration map, including its dependence
on the initial guess. It does not substitute the derivative of an exactly solved
linear system. Zero-diagonal rows retain their initial component and skip their
update. The derivative follows that branch with the zero-diagonal decision fixed.

CSR topology and the row schedule are fixed. CSR rows must have unique columns;
rows within each color group must be uncoupled except for their diagonals. The
sliced mode additionally requires a degree-one stencil, with no coupling across
more than one z slice. Use the schedules constructed by `row_schedule` for the
reconstruction's degree-one operators. This low-level API does not validate
matrix values or the coupling graph by downloading them to the CPU.

The forward pass saves the old component and rounded residual for each row visit:
`2 * iters * number_of_rows * sizeof(scalar)` bytes. Backward traverses the groups
in reverse, restoring a private copy of the solution before accumulating each
update's derivative. Commuting slices share launches in both directions.
Outputs and caller inputs remain unchanged by backward. Inputs, schedules, and
outputs must remain unchanged until backward completes; each call owns its
history. `tape.zero()` permits another backward with a new seed. Independent
tapes and inputs can use independent streams, and backward can run on a stream
other than forward. Shared gradient buffers must not have concurrent backwards.
Both CUDA directions use bounded, per-stream graph caches. A descriptor binds
fresh arrays before each launch; it does not retain a particular solve's buffers.
First use captures the schedule, and eviction waits for that stream's outstanding
work before releasing its descriptor. CPU execution uses the same row updates.

External CUDA graph capture and higher-order derivatives are unsupported. This
separate module is not imported by forward-only reconstruction and adds no
buffers, launches, or tape operations to that path.

## Functional products and root CG

`warp_psr.solver_adjoint` also provides functional scalar CSR products and
`finite_root_cg` for the small root system (at most 64 rows). The CG implementation
uses the pinned solver's dot-reduction layout, update arithmetic, and convergence
checks, while retaining distinct arrays for each iteration. Its backward uses
Warp's tape with the stopping branches fixed. The optional root mean operator
includes all 64 padded rows when supplied that matrix. It does not implement the
reconstruction solver's outer pre-CG skip test. The scalar CSR product matches
Warp's non-tiled row reduction; it does not promise the arithmetic of the optional
tiled long-row algorithm. Both helpers leave input values unchanged. Tolerances,
scalar product multipliers, and matrix topology are fixed parameters.

## Measurements and validation

The [per-run report](results/finite-solver-adjoints.json) records five isolated
warm repetitions on L40. These measure **one finest cascadic level**, with the
matrix, RHS, and initial state all tracked in the enabled mode. They exclude
reconstruction, row scheduling, tracking setup, validation, and gradient reset.

| System | Rows | Existing forward | Tracked forward | Backward | Saved history |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sphere, depth 4 | 3,168 | 0.956 ms | 1.318 ms | 3.124 ms | 0.193 MiB |
| Eagle, depth 9 | 2,052,088 | 12.505 ms | 16.729 ms | 50.210 ms | 125.25 MiB |

Both tracked forwards are bitwise equal to the production level solution on
every run. Cached graphs reduce eagle tracked-forward time from 78.5 ms to
16.7 ms and backward from 84.2 ms to 50.2 ms. The first eagle graph calls take
63.8 ms forward and 86.8 ms backward; the first sphere forward also compiles
kernels and takes 1.50 seconds. Tracking setup, first calls, all warm samples,
retained allocations, and pool high-water marks are recorded separately.

The eagle retains 1,537,425,667 pool bytes after the tracked forward versus
1,159,840,243 for the existing forward, including their resident systems. The
saved history is one part of that increase; input/output gradient buffers also
consume memory. The 2,516,547,663-byte peak in both processes includes prior
fixture reconstruction and is not an isolated backward peak. Native graph
storage is outside the pool metric. No full reconstruction-backward timing or
memory claim follows from these component measurements.

Tests cover matrix/RHS/initial-state directional derivatives, both precisions
on CPU/CUDA, zero diagonals, skipped and truncated CG, the 64-row root mean,
larger sliced groups, fresh buffers, overlapping gradient views, repeated tapes,
and independent streams. The full suite plus restored reference checks and the
two new alias cases totals 518 passing tests, one CPU skip for the CUDA stream
test, and one expected upstream failure. All forward pipeline files are
byte-identical to meshing main; importing reconstruction does not import either
adjoint module. The existing 31 public gates and reconstruction timing evidence
therefore remain applicable to the unchanged default path.

```sh
python -m tools.benchmark_relaxation_adjoint \
  --fixture .build/fixtures/eagle-full-d9-spsr --mode differentiable \
  --repeats 5 --output .build/eagle-finite-relaxation.json
```

## Work still required

These helpers alone do not differentiate multigrid, the outer conditional solve,
assembly, normalization, grouping, density/normal splats, the computed isovalue,
or the transformation of evaluated queries. The full mode must preserve the
chosen discrete assignments for backward and validate directional derivatives
of its actual public forward output, including the solver's stopping branches.
It also needs reconstruction-to-mesh gradient tests and separate enabled-mode
forward, backward, and memory measurements. These remain open in #76 and #77.
