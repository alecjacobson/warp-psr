"""Compact GPU row schedules for reference multicolor and sliced relaxation."""

from dataclasses import dataclass
import threading
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
        capacity: int,
    ):
        j = group_offsets[group] + wp.tid()
        while j < group_offsets[group + 1]:
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
            j += capacity

    return SimpleNamespace(_relax=_relax)


@lru_cache(None)
def graph_kernels(dtype):
    # The graph captures a program, not a particular reconstruction's buffers.
    # One device-side descriptor binds fresh arrays before each ordered launch.
    state_type = wp.struct(type(f"RelaxationState_{dtype.__name__}", (), {
        "__annotations__": {
            "rows": wp.array(dtype=int), "group_offsets": wp.array(dtype=int),
            "offsets": wp.array(dtype=int), "columns": wp.array(dtype=int),
            "values": wp.array(dtype=dtype), "b": wp.array(dtype=dtype),
            "x": wp.array(dtype=dtype),
        }
    }))

    @wp.kernel
    def bind(state: state_type, descriptor: wp.array(dtype=state_type)):
        descriptor[0] = state

    @wp.kernel
    def update(descriptor: wp.array(dtype=state_type), group: int, capacity: int):
        state = descriptor[0]
        j = state.group_offsets[group] + wp.tid()
        while j < state.group_offsets[group + 1]:
            i = state.rows[j]
            r = state.b[i]
            diagonal = dtype(0.0)
            for k in range(state.offsets[i], state.offsets[i + 1]):
                col = state.columns[k]
                r -= state.x[col] * state.values[k]
                if col == i:
                    diagonal = state.values[k]
            if diagonal != dtype(0.0):
                state.x[i] += r * (dtype(1.0) / diagonal)
            j += capacity

    return state_type, bind, update


def sliced_groups(depth, iters):
    for front in range(1 << depth, -iters, -1):
        for z in range(max(front, 0), min(front + iters, (1 << depth) + 1)):
            for color in range(4):
                yield z * 4 + color


@dataclass
class SlicedGraph:
    descriptor: wp.array
    graph: object
    stream: wp.Stream
    lock: object

    def __del__(self):
        # A bounded cache may evict an entry while another stream is active.
        # Keep its descriptor alive until all launches using it have finished.
        # This wait happens only on eviction/shutdown, never on a cache hit.
        try:
            wp.synchronize_stream(self.stream)
        except (AttributeError, TypeError):
            pass  # Warp may already be torn down at interpreter shutdown.


@lru_cache(maxsize=32)
def sliced_graph(dtype, device, stream, depth, iters, capacity):
    state_type, bind, update = graph_kernels(dtype)
    descriptor = wp.empty(1, dtype=state_type, device=device)
    command = wp.launch(update, capacity, inputs=[descriptor, 0, capacity],
                        device=device, record_cmd=True)
    # Compile the binding kernel before any capture starts.
    wp.launch(bind, 1, inputs=[state_type(), descriptor], device=device,
              record_cmd=True)
    with wp.ScopedCapture(device=device) as capture:
        for group in sliced_groups(depth, iters):
            command.set_param_at_index(1, group)
            command.launch()
    return SlicedGraph(descriptor, capture.graph, wp.get_stream(device), threading.Lock())


def relax_sliced(A, b, x, schedule, depth, iters, use_graph=True):
    """Execute the reference slice/color order with fresh data on every call."""
    if not use_graph or not x.device.is_cuda or x.device.is_capturing:
        command = relaxation_command(A, b, x, schedule)
        for group in sliced_groups(depth, iters):
            command.set_param_at_index(2, group)
            command.launch()
        return
    capacity = schedule.capacity
    cached = sliced_graph(
        b.dtype, str(x.device), wp.get_stream(x.device).cuda_stream, depth, iters, capacity
    )
    state_type, bind, _ = graph_kernels(b.dtype)
    state = state_type()
    state.rows, state.group_offsets = schedule.rows, schedule.offsets
    state.offsets, state.columns, state.values = A.offsets, A.columns, A.values
    state.b, state.x = b, x
    # Separate streams have separate descriptors. Lock the enqueue pair so
    # threads sharing a stream cannot replace a descriptor before its launch.
    with cached.lock:
        wp.launch(bind, 1, inputs=[state, cached.descriptor], device=x.device)
        wp.capture_launch(cached.graph)


@dataclass
class RowSchedule:
    rows: wp.array
    offsets: wp.array
    capacity: int
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
    # Rows in one color have no same-level coupling. A bounded launch visits
    # arbitrarily large groups with a grid stride, preserving every row's
    # arithmetic and the reference color/slice order. Counts stay on device.
    capacity = min(4096, 1 << max(0, n - 1).bit_length())
    return RowSchedule(rows, offsets, capacity, 0.0)


def relaxation_command(A, b, x, schedule):
    """Retain fixed arguments for one solve; only its color/slice group changes."""
    return wp.launch(
        precision_kernels(b.dtype)._relax,
        schedule.capacity,
        inputs=[
            schedule.rows,
            schedule.offsets,
            0,
            A.offsets,
            A.columns,
            A.values,
            b,
            x,
            schedule.capacity,
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
