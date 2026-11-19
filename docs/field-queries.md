# Query the implicit field in a Warp kernel

`result.evaluate(points)` allocates and returns arrays of raw implicit values,
spatial gradients in input coordinates, and integer validity flags. To combine
evaluation with application work in one kernel, pass `result.device_field()` to
the `evaluate_field` device function:

```python
import warp as wp
from warp_psr import Field, evaluate_field

@wp.kernel
def classify(field: Field, points: wp.array[wp.vec3], iso_value: float,
             inside: wp.array[int]):
    i = wp.tid()
    value, gradient, valid = evaluate_field(field, points[i])
    inside[i] = int(valid != 0 and value < iso_value)

field = result.device_field()
inside = wp.empty(len(points), dtype=int, device=points.device)
wp.launch(classify, len(points), inputs=[field, points, result.iso_value],
          outputs=[inside], device=points.device)
```

Use `Field64`, `wp.vec3d`, and `wp.float64` for double precision. `Field` and
`Field64` are kernel argument annotations; obtain handles from the reconstruction
or its `LevelField`, rather than filling their implementation members yourself.
Handle creation shares the existing arrays and performs no device allocation,
copy, or host synchronization. The single-query call performs no kernel launch
or allocation and can be used in loops or captured application kernels.
The existing batched reconstruction-field kernel is retained independently;
adding or using the device function does not route batched queries through it.
The [validation report](results/device-field/README.md) compares both routes on
bunny, horse, and eagle, including depth 9 and double precision.

With the pinned Warp 1.19 release, decorate any additional device helper that
accepts the handle with **`@wp.func(inline=True)`**. Plain `@wp.func` lets the
compiler choose its inlining strategy, which can cause a CUDA local-memory fault
with this handle. Explicit `inline=False` also avoids the reproduced fault;
`inline=True` remains the supported choice for the pinned release. The public
evaluator already enforces inlining. The underlying compiler issue is tracked in
[Warp #2362](https://github.com/NVIDIA/warp/issues/2362), with integration status in
[issue #101](https://github.com/alecjacobson/warp-psr/issues/101).

The point and launch must match the field's precision and device. Points outside
the normalized reconstruction cube, including nonfinite points, return NaN for
both value and gradient and zero validity. Valid queries return one. Values are
not signed distances: the reconstructed surface is `value == result.iso_value`.
The gradient is the spatial derivative of the field, not a normalized normal.
Evaluation retains the batched API's adaptive-cell, knot, and boundary rules.
For bitwise agreement, compile the caller's module with
`wp.set_module_options({"fuse_fp": False})`, as the built-in evaluator does;
other compiler settings can change floating-point rounding.

Handles keep references to the field arrays. Keep arrays immutable while kernels
or backward passes use them. A handle from ordinary `reconstruct` treats the
reconstruction and its host transform as constants; query positions can still
participate in first-order autodiff.

## Differentiable reconstruction

A handle from `reconstruct_differentiable` shares its tracked coefficients and
device normalization transform. Record application kernels on the same supplied
tape. Creating the handle on the query stream waits for the reconstruction's
readiness event without synchronizing the host.

```python
from warp_psr import reconstruct_differentiable

@wp.kernel
def field_loss(field: Field, points: wp.array[wp.vec3], iso: wp.array[float],
               loss: wp.array[float]):
    i = wp.tid()
    value, gradient, valid = evaluate_field(field, points[i])
    if valid != 0:
        residual = value - iso[0]
        wp.atomic_add(loss, 0, residual * residual)

tape = wp.Tape()
result = reconstruct_differentiable(points, normals, tape=tape, depth=6)
field = result.device_field()
loss = wp.zeros(1, dtype=float, device=points.device, requires_grad=True)
with tape:
    wp.launch(field_loss, len(queries), inputs=[field, queries, result.iso_value],
              outputs=[loss], device=points.device)
tape.backward(loss)
```

Set `requires_grad=True` on input point/normal arrays and any queries whose
derivatives are needed. Both values and spatial gradients support first-order
losses, including the mixed second spatial derivatives needed for gradient
losses. Mask invalid queries before forming a loss. Isovalue centering is
explicit in this example so its gradient remains connected.

The same [fixed-assignment derivative contract](differentiability.md#derivative-contract)
applies: tree topology, support, lookup choices, and solver decisions stay fixed.
This is first-order autodiff, not differentiation of topology changes or support
for higher-order tapes. The differentiable reconstruction/backward pipeline
retains its existing restriction on external CUDA graph capture.
