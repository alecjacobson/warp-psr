"""Reference cascadic correction solve using Warp FEM and Warp sparse algebra."""

import time
from functools import lru_cache
from types import SimpleNamespace
import warp as wp
import warp.sparse as sparse
from warp.optim.linear import cg
from .field import basis_key, find_key
from .keyed import coordinates
from .forms import (
    assemble_stiffness,
    assemble_screening,
    assemble_point_load,
    point_quadrature,
)
from .normal_loads import assemble_reference_normal_loads
from .constraints import point_constraints
from .spaces import make_level_space
from .relaxation import row_schedule, relax_scheduled, relaxation_command
from .integration import integration_grid

wp.set_module_options({"fuse_fp": False})


@lru_cache(None)
def precision_kernels(dtype):
    vec3 = wp.types.vector(length=3, dtype=dtype)

    @wp.kernel
    def _transfer(
        fine: wp.array(dtype=wp.int64),
        coarse: wp.array(dtype=wp.int64),
        depth: int,
        rows: wp.array(dtype=int),
        cols: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
    ):
        i, k = wp.tid()
        f = coordinates(fine[i])
        c = wp.vec3i(f[0] // 2, f[1] // 2, f[2] // 2) + wp.vec3i(
            k & 1, k >> 1 & 1, k >> 2 & 1
        )
        h = vec3(dtype(1.0)) - wp.abs(vec3(f) * dtype(0.5) - vec3(c))
        v = h[0] * h[1] * h[2]
        idx = int(-1)
        if v > dtype(0.0):
            idx = find_key(coarse, basis_key(depth - 1, c[0], c[1], c[2]))
        rows[i * 8 + k] = i
        cols[i * 8 + k] = idx
        values[i * 8 + k] = v

    @wp.kernel
    def _gs(
        keys: wp.array(dtype=wp.int64),
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        b: wp.array(dtype=dtype),
        x: wp.array(dtype=dtype),
        color: int,
        z: int,
    ):
        i = wp.tid()
        c = coordinates(keys[i])
        if (c[0] & 1) + 2 * (c[1] & 1) + 4 * (c[2] & 1) == color and (
            z < 0 or c[2] == z
        ):
            r = b[i]
            diagonal = dtype(0.0)
            for j in range(offsets[i], offsets[i + 1]):
                col = columns[j]
                r -= x[col] * values[j]
                if col == i:
                    diagonal = values[j]
            if diagonal != dtype(0.0):
                x[i] += r * (dtype(1.0) / diagonal)

    @wp.kernel
    def _combine(
        a: wp.array(dtype=dtype),
        b: wp.array(dtype=dtype),
        scale: dtype,
        out: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        out[i] = a[i] + scale * b[i]

    @wp.kernel
    def _rhs(
        a: wp.array(dtype=wp.float64),
        b: wp.array(dtype=wp.float64),
        out: wp.array(dtype=dtype),
    ):
        i = wp.tid()
        out[i] = dtype(a[i]) + dtype(0.5) * dtype(b[i])

    return SimpleNamespace(_transfer=_transfer, _gs=_gs, _combine=_combine, _rhs=_rhs)


def prolongation(coarse_keys, fine_keys, depth, dtype=wp.float32):
    device = fine_keys.device
    count = len(fine_keys) * 8
    rows, cols = (
        wp.empty(count, dtype=int, device=device),
        wp.empty(count, dtype=int, device=device),
    )
    values = wp.empty(count, dtype=dtype, device=device)
    if count:
        wp.launch(
            precision_kernels(dtype)._transfer,
            (len(fine_keys), 8),
            inputs=[fine_keys, coarse_keys, depth, rows, cols, values],
            device=device,
        )
    return sparse.bsr_from_triplets(
        len(fine_keys), len(coarse_keys), rows, cols, values
    )


def relax(A, b, x, keys, iters, forward=True, z=-1):
    colors = range(8) if forward else range(7, -1, -1)
    for _ in range(iters):
        for color in colors:
            if z >= 0 and (color >> 2) != (z & 1):
                continue
            wp.launch(
                precision_kernels(b.dtype)._gs,
                len(keys),
                inputs=[keys, A.offsets, A.columns, A.values, b, x, color, z],
                device=keys.device,
            )


def regular_multigrid(A, b, keys, transfers, iters, cycles, cg_accuracy):
    base = len(keys) - 1
    matrices = [None] * base + [A]
    restrictions = [sparse.bsr_transposed(P) for P in transfers]
    for d in range(base, 0, -1):
        matrices[d - 1] = sparse.bsr_mm(
            sparse.bsr_mm(restrictions[d - 1], matrices[d]), transfers[d - 1]
        )
    rhs = [wp.zeros(len(k), dtype=b.dtype, device=k.device) for k in keys]
    rhs[base] = b
    x = [wp.zeros_like(v) for v in rhs]
    schedules = [row_schedule(k, d) for d, k in enumerate(keys)]
    commands = [
        relaxation_command(matrices[d], rhs[d], x[d], schedules[d])
        for d in range(base + 1)
    ]
    sync = sum(schedule.sync_seconds for schedule in schedules)
    for _ in range(cycles):
        for d in range(base, 0, -1):
            relax_scheduled(
                matrices[d], rhs[d], x[d], schedules[d], iters, command=commands[d]
            )
            residual = wp.clone(rhs[d])
            sparse.bsr_mv(matrices[d], x[d], residual, alpha=-1.0, beta=1.0)
            sparse.bsr_mv(restrictions[d - 1], residual, rhs[d - 1])
            x[d - 1].zero_()
        # The reference skips CG altogether when its initial squared residual
        # is <= eps^2. After this check its stopping tolerance is relative.
        start = time.perf_counter()
        norm_squared = wp.utils.array_inner(rhs[0], rhs[0])
        sync += time.perf_counter() - start
        if norm_squared > cg_accuracy * cg_accuracy:
            cg(
                matrices[0],
                rhs[0],
                x[0],
                tol=cg_accuracy,
                atol=0,
                maxiter=len(keys[0]),
                check_every=1,
                use_cuda_graph=False,
            )
        for d in range(1, base + 1):
            sparse.bsr_mv(transfers[d - 1], x[d - 1], x[d], beta=1.0)
            x[d - 1].zero_()
            relax_scheduled(
                matrices[d],
                rhs[d],
                x[d],
                schedules[d],
                iters,
                forward=False,
                command=commands[d],
            )
    return x[base], sync


def solve_levels(
    tree,
    geometry,
    normals,
    density,
    *,
    base_depth,
    solve_depth,
    point_weight=2.0,
    exact=False,
    iters=8,
    base_v_cycles=1,
    cg_accuracy=1e-3,
):
    device = normals.records.keys.device
    dtype = density.scale.dtype
    finest = len(tree.basis) - 1
    keys = [level.keys for level in tree.basis]
    transfers = [
        prolongation(keys[d - 1], keys[d], d, dtype) for d in range(1, finest + 1)
    ]
    corrections = [wp.zeros(len(k), dtype=dtype, device=device) for k in keys]
    diagnostics = []
    solver_sync = 0.0
    cumulative = None
    # The reference assembles normal constraints through reconstruction depth,
    # including fine contributions when the requested solve stops earlier.
    normal_loads = assemble_reference_normal_loads(
        tree, geometry, normals, finest, base_depth, transfers
    )
    screened = point_weight > 0 and not exact
    for d in range(base_depth, solve_depth + 1):
        if not len(keys[d]):
            break
        # Normal loads need the finest partition's quadratic breakpoints. The
        # stiffness and screening operators only need this level's basis knots.
        normal_space = make_level_space(geometry, keys[d], d, finest)
        load = normal_loads[d]
        if d == finest:
            operator_geometry = geometry
            operator_finest = finest
            space = normal_space
        else:
            operator_geometry = integration_grid(tree, None, d, dtype=dtype)
            operator_finest = d
            space = make_level_space(operator_geometry, keys[d], d, d)
        del normal_space
        # Double accumulation avoids a coarse row losing accuracy when its
        # integral spans many fine integration cells. Final reference matrices
        # and solve arithmetic are float32.
        A = assemble_stiffness(space, device, accumulation_dtype=wp.float64)
        points = point_constraints(density, normals, tree, d, point_weight, exact)
        if screened:
            quadrature = point_quadrature(space, points.positions, points.weights)
            screen = assemble_screening(
                space,
                points.positions,
                points.weights,
                accumulation_dtype=wp.float64,
                quadrature=quadrature,
            )
            sparse.bsr_axpy(screen, A)
            point_load = assemble_point_load(
                space, points.positions, points.weights, quadrature=quadrature
            )
        else:
            point_load = wp.zeros_like(load)
        b = wp.empty(len(keys[d]), dtype=dtype, device=device)
        wp.launch(
            precision_kernels(dtype)._rhs,
            len(b),
            inputs=[load, point_load, b],
            device=device,
        )
        A = sparse.bsr_copy(A, scalar_type=dtype)
        if d == base_depth:
            corrections[d], sync = regular_multigrid(
                A, b, keys[: d + 1], transfers[:d], iters, base_v_cycles, cg_accuracy
            )
            solver_sync += sync
            cumulative = wp.clone(corrections[d])
        else:
            coarse_space = make_level_space(
                operator_geometry, keys[d - 1], d - 1, operator_finest
            )
            cross = assemble_stiffness(
                space, device, trial_space=coarse_space, accumulation_dtype=wp.float64
            )
            if screened:
                screen = assemble_screening(
                    space,
                    points.positions,
                    points.weights,
                    trial_space=coarse_space,
                    accumulation_dtype=wp.float64,
                    quadrature=quadrature,
                )
                sparse.bsr_axpy(screen, cross)
            cross = sparse.bsr_copy(cross, scalar_type=dtype)
            sparse.bsr_mv(cross, cumulative, b, alpha=-1.0, beta=1.0)
            # Pinned sliceBlockSize=1, degree=1, coarse-to-fine traversal.
            # Each diagonal wave updates increasing z, then moves its front -1.
            schedule = row_schedule(keys[d], d, sliced=True)
            command = relaxation_command(A, b, corrections[d], schedule)
            solver_sync += schedule.sync_seconds
            for front in range(1 << d, -iters, -1):
                for z in range(max(front, 0), min(front + iters, (1 << d) + 1)):
                    relax_scheduled(
                        A, b, corrections[d], schedule, 1, z=z, command=command
                    )
            next_cumulative = wp.clone(corrections[d])
            sparse.bsr_mv(transfers[d - 1], cumulative, next_cumulative, beta=1.0)
            cumulative = next_cumulative
        diagnostics.append(
            {
                "depth": d,
                "matrix": A,
                "rhs": b,
                "normal_load": load,
                "coefficients": corrections[d],
                "operator_integration_cells": operator_geometry.cell_count(),
            }
        )
    return corrections, diagnostics, solver_sync
