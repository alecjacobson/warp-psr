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

External CUDA graph capture and higher-order derivatives are unsupported. This
separate module is not imported by forward-only reconstruction and adds no
buffers, launches, or tape operations to that path.

## Work still required

This helper alone does not differentiate multigrid, the conditional root CG,
assembly, normalization, grouping, density/normal splats, the computed isovalue,
or the transformation of evaluated queries. The full mode must preserve the
chosen discrete assignments for backward and validate directional derivatives
of its actual public forward output, including the solver's stopping branches.
It also needs reconstruction-to-mesh gradient tests and separate enabled-mode
forward, backward, and memory measurements. These remain open in #76 and #77.
