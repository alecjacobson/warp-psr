"""Compact GPU row schedules for reference multicolor and sliced relaxation."""

from dataclasses import dataclass
import time
from functools import lru_cache
from types import SimpleNamespace
import warp as wp
from .keyed import coordinates

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _assign(
    keys: wp.array(dtype=wp.int64),
    sliced: int,
    groups: wp.array(dtype=int),
    rows: wp.array(dtype=int),
    counts: wp.array(dtype=int),
):
    i = wp.tid()
    c = coordinates(keys[i])
    group = (c[0] & 1) + 2 * (c[1] & 1) + 4 * (c[2] & 1)
    if sliced:
        group = (c[0] & 1) + 2 * (c[1] & 1) + 4 * c[2]
    groups[i] = group
    rows[i] = i
    wp.atomic_add(counts, group, 1)


@wp.kernel
def _maximum(counts: wp.array(dtype=int), maximum: wp.array(dtype=int)):
    i = wp.tid()
    wp.atomic_max(maximum, 0, counts[i])


@lru_cache(None)
def precision_kernels(dtype):
    @wp.kernel
    def _relax(
        rows: wp.array(dtype=int),
        group_offsets: wp.array(dtype=int),
        group: int,
        offsets: wp.array(dtype=int),
        columns: wp.array(dtype=int),
        values: wp.array(dtype=dtype),
        b: wp.array(dtype=dtype),
        x: wp.array(dtype=dtype),
    ):
        j = group_offsets[group] + wp.tid()
        if j < group_offsets[group + 1]:
            i = rows[j]
            r = b[i]
            diagonal = dtype(0.0)
            for k in range(offsets[i], offsets[i + 1]):
                col = columns[k]
                r -= x[col] * values[k]
                if col == i:
                    diagonal = values[k]
            if diagonal != dtype(0.0):
                x[i] += r * (dtype(1.0) / diagonal)

    return SimpleNamespace(_relax=_relax)


@dataclass
class RowSchedule:
    rows: wp.array
    offsets: wp.array
    maximum: int
    sync_seconds: float


def row_schedule(keys, depth, sliced=False):
    device = keys.device
    n = len(keys)
    group_count = 4 * ((1 << depth) + 1) if sliced else 8
    counts = wp.zeros(group_count + 1, dtype=int, device=device)
    offsets = wp.empty_like(counts)
    groups = wp.empty(2 * n, dtype=int, device=device)
    rows = wp.empty(2 * n, dtype=int, device=device)
    wp.launch(
        _assign, n, inputs=[keys, int(sliced), groups, rows, counts], device=device
    )
    wp.utils.radix_sort_pairs(groups, rows, n)
    wp.utils.array_scan(counts, offsets, inclusive=False)
    maximum = wp.zeros(1, dtype=int, device=device)
    wp.launch(_maximum, group_count, inputs=[counts, maximum], device=device)
    start = time.perf_counter()
    largest = int(maximum.numpy()[0])
    return RowSchedule(rows, offsets, largest, time.perf_counter() - start)


def relaxation_command(A, b, x, schedule):
    """Retain fixed arguments for one solve; only its color/slice group changes."""
    return wp.launch(
        precision_kernels(b.dtype)._relax,
        schedule.maximum,
        inputs=[
            schedule.rows,
            schedule.offsets,
            0,
            A.offsets,
            A.columns,
            A.values,
            b,
            x,
        ],
        device=x.device,
        record_cmd=True,
    )


def relax_scheduled(A, b, x, schedule, iters, forward=True, z=-1, command=None):
    if command is None:
        command = relaxation_command(A, b, x, schedule)
    colors = range(8) if forward else range(7, -1, -1)
    for _ in range(iters):
        for color in colors:
            if z >= 0 and (color >> 2) != (z & 1):
                continue
            group = color if z < 0 else z * 4 + (color & 3)
            command.set_param_at_index(2, group)
            command.launch()
