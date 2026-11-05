"""Reverse the actual ordered Gauss--Seidel iteration map (opt-in substrate).

This module is not imported by reconstruction. It is the solver building block
for issue #76, not an end-to-end reconstruction backward implementation.
"""
from functools import lru_cache
from types import SimpleNamespace
import warp as wp
from .relaxation import wavefront_groups

wp.set_module_options({'fuse_fp': False})


@lru_cache(None)
def precision_kernels(dtype):
    pair = wp.types.vector(length=2, dtype=dtype)

    @wp.kernel
    def copy(src: wp.array(dtype=dtype), dst: wp.array(dtype=dtype)):
        i = wp.tid()
        dst[i] = src[i]

    @wp.kernel
    def add(src: wp.array(dtype=dtype), dst: wp.array(dtype=dtype)):
        i = wp.tid()
        wp.atomic_add(dst, i, src[i])

    @wp.kernel
    def forward(rows: wp.array(dtype=int), group_offsets: wp.array(dtype=int), first_group: int, first_visit: int,
                group_stride: int,
                offsets: wp.array(dtype=int), columns: wp.array(dtype=int), values: wp.array(dtype=dtype),
                b: wp.array(dtype=dtype), x: wp.array(dtype=dtype), history: wp.array2d(dtype=pair)):
        batch, row = wp.tid()
        group = first_group + batch * group_stride
        visit = first_visit + batch
        j = group_offsets[group] + row
        if j < group_offsets[group + 1]:
            i = rows[j]
            r = b[i]
            diagonal = dtype(0)
            for k in range(offsets[i], offsets[i + 1]):
                col = columns[k]
                r -= x[col] * values[k]
                if col == i:
                    diagonal = values[k]
            # Store the old component and the actual rounded residual. Rewinding
            # each independent group reconstructs all its input neighbor values.
            history[visit, i] = pair(x[i], r)
            if diagonal != dtype(0):
                x[i] += r * (dtype(1) / diagonal)

    @wp.kernel
    def reverse(rows: wp.array(dtype=int), group_offsets: wp.array(dtype=int), first_group: int, first_visit: int,
                group_stride: int,
                offsets: wp.array(dtype=int), columns: wp.array(dtype=int), values: wp.array(dtype=dtype),
                x: wp.array(dtype=dtype), history: wp.array2d(dtype=pair), adj_x: wp.array(dtype=dtype),
                adj_b: wp.array(dtype=dtype), adj_values: wp.array(dtype=dtype)):
        batch, row = wp.tid()
        group = first_group + batch * group_stride
        visit = first_visit + batch
        j = group_offsets[group] + row
        if j < group_offsets[group + 1]:
            i = rows[j]
            diagonal = dtype(0)
            diagonal_index = int(-1)
            for k in range(offsets[i], offsets[i + 1]):
                if columns[k] == i:
                    diagonal = values[k]
                    diagonal_index = k
            old = history[visit, i]
            x[i] = old[0]
            if diagonal != dtype(0):
                seed = adj_x[i]
                inv = dtype(1) / diagonal
                adj_r = seed * inv
                if adj_b.shape[0] != 0:
                    wp.atomic_add(adj_b, i, adj_r)
                # Keep both contributions from the actual update expression:
                # x_i + (b_i - sum_j A_ij*x_j) * (1/A_ii).
                # This is the iteration derivative, not an implicit solve.
                if adj_values.shape[0] != 0:
                    wp.atomic_add(adj_values, diagonal_index, -(seed * old[1]) / (diagonal * diagonal))
                for k in range(offsets[i], offsets[i + 1]):
                    col = columns[k]
                    if adj_values.shape[0] != 0:
                        wp.atomic_add(adj_values, k, -adj_r * x[col])
                    wp.atomic_add(adj_x, col, -adj_r * values[k])

    return SimpleNamespace(copy=copy, add=add, forward=forward, reverse=reverse)


def differentiable_relaxation(matrix, b, initial, schedule, *, tape, iters,
                              forward=True, sliced_depth=None):
    """Return a tracked finite-iteration solution without mutating ``initial``.

    Differentiate with respect to tracked CSR ``matrix.values``, ``b``, and
    ``initial``. The caller supplies its ``wp.Tape`` explicitly. The row schedule
    and CSR topology are fixed. As in production relaxation, rows in a schedule
    group must be uncoupled except for their diagonals, and CSR columns must be
    unique within each row. Inputs, schedules, and returned outputs must remain
    unchanged until backward completes. Each invocation owns its saved history;
    independent tapes can run on independent streams. Gradients accumulate into
    the caller's buffers, so concurrent backwards must use distinct inputs.

    ``sliced_depth=None`` uses regular forward/reverse color sweeps; otherwise
    use the reference's sliced fine-level schedule. The latter has the same
    arithmetic/dependencies as the production wavefront schedule and requires
    the degree-one stencil (no coupling across more than one z slice). Save two scalar
    values per row visit, rather than a whole vector per group. Reverse execution
    rewinds a private solution copy, leaving the output and caller arrays intact.

    This deliberately separate API adds no allocations, launches, tape handling,
    or imports to forward-only reconstruction. Full reconstruction gradients,
    multigrid composition, root CG, higher-order derivatives, and external CUDA
    graph capture are not implemented by this function.
    """
    if not isinstance(tape, wp.Tape):
        raise ValueError('an explicit wp.Tape is required')
    if not isinstance(iters, int) or isinstance(iters, bool) or iters < 0:
        raise ValueError('iters must be a nonnegative integer')
    if (b.dtype not in (wp.float32, wp.float64) or initial.dtype != b.dtype or matrix.values.dtype != b.dtype
            or b.ndim != 1 or initial.shape != b.shape or not len(b)
            or matrix.nrow != len(b) or matrix.ncol != len(b)):
        raise ValueError('require a nonempty square scalar CSR matrix and matching float32/float64 vectors')
    device = b.device
    if device.is_capturing:
        raise ValueError('differentiable relaxation does not support external CUDA graph capture')
    if any(a.device != device for a in [initial, matrix.values, matrix.offsets, matrix.columns, schedule.rows, schedule.offsets]):
        raise ValueError('matrix, vectors, and schedule must share one device')
    if sliced_depth is not None:
        if not isinstance(sliced_depth, int) or isinstance(sliced_depth, bool) or not 0 <= sliced_depth <= 15:
            raise ValueError('sliced_depth must be an integer in 0..15')
        if not forward:
            raise ValueError('sliced relaxation uses the reference forward schedule')
        # Production's commuting wavefront batches retain each coupled row's
        # order. The visit index increases with z within one wave.
        program = []
        for step, (z, count, color) in enumerate(wavefront_groups(sliced_depth, iters)):
            wave = step // 4
            visit = (wave - (1 << sliced_depth) + z) // 2
            program.append((4 * z + color, visit, count, 8))
        group_count = 4 * ((1 << sliced_depth) + 1)
    else:
        program = [(g, visit, 1, 0) for visit in range(iters)
                   for g in (range(8) if forward else range(7, -1, -1))]
        group_count = 8
    if schedule.offsets.shape != (group_count + 1,):
        raise ValueError('row schedule does not match the requested relaxation mode')
    kernels = precision_kernels(b.dtype)
    pair = wp.types.vector(length=2, dtype=b.dtype)
    output = wp.empty_like(initial, requires_grad=True)
    history = wp.empty((iters, len(b)), dtype=pair, device=device)
    wp.launch(kernels.copy, len(b), inputs=[initial, output], device=device, record_tape=False)
    schedule_arrays = [schedule.rows, schedule.offsets]
    matrix_arrays = [matrix.offsets, matrix.columns, matrix.values]
    for group, visit, count, stride in program:
        wp.launch(kernels.forward, (count, schedule.maximum),
                  inputs=[*schedule_arrays, group, visit, stride, *matrix_arrays, b, output, history],
                  device=device, record_tape=False)
    event = None
    if device.is_cuda:
        event = wp.Event(device)
        wp.get_stream(device).record_event(event)

    def backward():
        with wp.ScopedDevice(device):
            if event is not None:
                wp.wait_event(event)
            work = wp.clone(output, requires_grad=False)
            adjoint = wp.clone(output.grad, requires_grad=False)
            empty = wp.empty(0, dtype=b.dtype, device=device)
            adj_b = b.grad if b.requires_grad else empty
            adj_values = matrix.values.grad if matrix.values.requires_grad else empty
            for group, visit, count, stride in reversed(program):
                wp.launch(kernels.reverse, (count, schedule.maximum),
                          inputs=[*schedule_arrays, group, visit, stride, *matrix_arrays, work, history,
                                  adjoint, adj_b, adj_values], device=device, record_tape=False)
            if initial.requires_grad:
                wp.launch(kernels.add, len(initial), inputs=[adjoint, initial.grad], device=device, record_tape=False)

    tape.record_func(backward, [a for a in [matrix.values, b, initial, output] if a.requires_grad])
    return output
