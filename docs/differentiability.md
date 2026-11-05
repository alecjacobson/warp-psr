# Differentiable reconstruction

`reconstruct_differentiable` is an opt-in, first-order backward path for input
point positions and normals. The default `reconstruct` function retains its
forward implementation and does not import or allocate backward machinery.
Both float32 and float64 reconstruction are supported on CUDA.

```python
import warp as wp
from warp_psr import reconstruct_differentiable

# points, normals and queries are matching CUDA vec3/vec3d arrays.
points.requires_grad = True
normals.requires_grad = True
tape = wp.Tape()
result = reconstruct_differentiable(points, normals, tape=tape, depth=6)
with tape:
    values, spatial_gradients, valid = result.evaluate(queries, centered=True)
    # Record a scalar loss kernel consuming values here.
tape.backward(grads={values: wp.ones_like(values, requires_grad=False)})
# points.grad and normals.grad now contain the loss VJPs.
```

Call reconstruction **outside** a `with tape` block, then record evaluations and
losses inside the same supplied tape. Earlier input-producing kernels may be
recorded in a preceding block of that tape. Warp 1.19 has no public tape
suspension context; this interface avoids recording the forward-only pipeline
or changing private tape state. An attempt to reconstruct inside a recording
tape fails explicitly.

The returned `iso_value` is a one-element tracked **device array**. With
`centered=True`, values subtract this computed isovalue and retain its input
gradients. With `centered=False`, evaluation returns the ordinary implicit
field. Returned spatial gradients describe variation with query position;
`tape.backward` computes variation with reconstruction inputs. Query positions
can also be tracked. Normalization, its model/cube transform and the computed
isovalue remain connected to the loss.

## Derivative contract

The backward pass differentiates the **actual finite forward algorithm**:
ordered Gauss–Seidel, cascadic corrections, Galerkin multigrid, conditional root
CG, and the padded constant mode. It does not replace these iterations with an
implicit adjoint of a fully converged system. Replay is checkpointed by stage:
field/isovalue, solve/assembly, normal splats, density, then sample preparation.
Each stage releases its intermediates before the next stage.

The tree, active/support sets, sample grouping, bins, normal splat levels,
lookup choices and solver stopping decisions remain fixed. Gradients are local
piecewise derivatives, valid away from transitions in these assignments.
Sample bounds select the lowest-index point on an extrema tie; branch-selected
normalization and interpolation derivatives at ties/knots do not describe a
finite difference that changes branches. Rejected normal samples contribute no
normal gradient; their positions still participate in the original bounding
box. Cancelled normal groups follow the zero-contribution branch. Invalid field
queries retain the existing NaN/valid-mask behavior and must be excluded from a
loss. Boundary nudges retain their fixed branch derivative.

Only positions and normals are differentiable reconstruction inputs initially.
Scalar settings, including `point_weight` (screening strength), scale, density
target, depths, iteration counts and CG accuracy, are constants. Passing device
arrays for these options fails explicitly. Higher-order derivatives and external
CUDA graph capture are unsupported. Inputs, queries and returned checkpoint
arrays must stay immutable until backward finishes. `tape.zero()` supports
repeated backward calls; multiple reconstructions can share a tape. Independent
tapes can use independent streams, with normal Warp stream ordering for loss
and input buffers. Concurrent backwards must not share gradient buffers.

The legacy `LevelField.evaluate(requires_grad=True)` remains available for direct
non-cumulative coefficient fields. It rejects cumulative reference fields:
their former automatic reverse pass lost contributions. Use
`reconstruct_differentiable` for the validated reference-basis VJP.

## Mesh losses

```python
from warp_psr.meshing import extract_mesh

# Reuse the reconstruction tape and keep bounds inside its valid cube.
with tape:
    mesh = extract_mesh(result, resolution=128, bounds=bounds,
                        backend="sparse", differentiable=True)
    # Record a loss on mesh.vertices here.
```

Dense and sparse Warp MC both propagate vertex interpolation gradients through
the implicit field and isovalue to input positions and normals. Warp MC requires
float32 fields; double meshing fails explicitly. Grid bounds, crossing-cell
selection, connectivity and topology are fixed. This is not a derivative of
appearing/disappearing triangles. The sparse classifier retains all dense-grid
crossings without a signed-distance assumption. An optional `lipschitz` backend
requires a caller-proven bound, as described in [meshing](meshing.md).
Tracked meshing retains callback intermediates, including classification slabs;
its memory/time costs differ from forward-only sparse extraction. Reconstruction
benchmark timings exclude meshing.

## Validation and reconstruction measurements

End-to-end tests compare point and normal VJPs with finite differences of the
ordinary public reconstruction. Coverage includes both precisions, PSR/SPSR,
active fine levels (including direct screening), computed nonzero isovalues,
exact-interpolation compatibility, an early solve depth, repeated base cycles,
multiple calls, repeated backwards and streams. Dense/sparse mesh tests preserve
connectivity during perturbations. Stage tests also cover cancelled groups,
omitted NaN normals, multiple reduction tiles, adaptive normal loads, assembly
and every finite solver component. Float32 checks use step sweeps to distinguish
rounding noise from changed bins; float64 checks retain tight derivative budgets.

The 31 public datasets reproduce archived, reference-validated field values and
spatial gradients bit for bit in both default and differentiable modes, with
identical isovalues and exact reference tree hashes. The original accuracy
budgets are unchanged. Detailed performance reports will accompany the final
validation of this change; initial full-eagle measurements are approximately
0.234 s enabled forward, 0.532 s backward, and 5.01 GiB process peak pool memory
on L40, excluding input upload, query evaluation and meshing. These are initial
measurements, not portable thresholds.

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
Warp's serial or 64-lane tiled reduction, including the capacity-based selection
heuristic that can select tiles for short coarse Galerkin rows. Both helpers leave input values unchanged. Tolerances,
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
