"""The compact GPU schedule must preserve the verified relaxation arithmetic."""
import numpy as np
import pytest
import warp as wp
from warp_psr.forms import uniform_space,assemble_stiffness
from warp_psr.solver import relax
from warp_psr.relaxation import row_schedule,relax_scheduled,relaxation_command,relax_sliced

@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_compact_schedule_matches_full_row_scan(device):
    wp.init()
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    with wp.ScopedDevice(device):
        space=uniform_space(2)
        A=assemble_stiffness(space,device)
        xyz=np.rint(space.node_positions().numpy()*4).astype(np.int64)
        keys=wp.array((2<<48)|(xyz[:,0]<<32)|(xyz[:,1]<<16)|xyz[:,2],dtype=wp.int64)
        b=wp.array(np.sin(np.arange(len(xyz))).astype('f'))
        reference=wp.array(np.cos(np.arange(len(xyz))).astype('f'))
        actual=wp.clone(reference)
        schedule=row_schedule(keys,2)
        command=relaxation_command(A,b,actual,schedule)
        for forward in [True,False]:
            relax(A,b,reference,keys,3,forward=forward)
            relax_scheduled(A,b,actual,schedule,3,forward=forward,command=command)
            np.testing.assert_array_equal(actual.numpy(),reference.numpy())
        schedule=row_schedule(keys,2,sliced=True)
        command=relaxation_command(A,b,actual,schedule)
        for front in range(4,-3,-1):
            for z in range(max(front,0),min(front+3,5)):
                relax(A,b,reference,keys,1,z=z)
                relax_scheduled(A,b,actual,schedule,1,z=z,command=command)
        np.testing.assert_array_equal(actual.numpy(),reference.numpy())


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('depth', [2, 3])
def test_sliced_graph_rebinds_fresh_arrays_and_preserves_order(dtype, depth):
    import warp.sparse as sparse
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    with wp.ScopedDevice('cuda'):
        space = uniform_space(depth)
        matrix = assemble_stiffness(space, 'cuda')
        xyz = np.rint(space.node_positions().numpy() * (1 << depth)).astype(np.int64)
        keys = wp.array((depth << 48) | (xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2], dtype=wp.int64)
        schedule = row_schedule(keys, depth, sliced=True)
        # Reusing a graph must never reuse the previous RHS, solution, or matrix.
        retained = []
        for iteration in range(3):
            A = sparse.bsr_copy(matrix, scalar_type=dtype)
            sparse.bsr_scale(A, 1.0 + 0.1 * iteration)
            b = wp.array(np.sin(np.arange(len(xyz)) + iteration), dtype=dtype)
            reference = wp.array(np.cos(np.arange(len(xyz)) - iteration), dtype=dtype)
            actual = wp.clone(reference)
            relax_sliced(A, b, reference, schedule, depth, 3, use_graph=False)
            relax_sliced(A, b, actual, schedule, depth, 3)
            np.testing.assert_array_equal(actual.numpy(), reference.numpy())
            retained.append((A, b, actual, reference))


@pytest.mark.cuda
def test_sliced_graph_separates_streams_and_survives_cache_eviction():
    from warp_psr.relaxation import sliced_graph
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    with wp.ScopedDevice('cuda'):
        space = uniform_space(2)
        A = assemble_stiffness(space, 'cuda')
        xyz = np.rint(space.node_positions().numpy() * 4).astype(np.int64)
        keys = wp.array((2 << 48) | (xyz[:, 0] << 32) | (xyz[:, 1] << 16) | xyz[:, 2], dtype=wp.int64)
        schedule = row_schedule(keys, 2, sliced=True)
        wp.synchronize()
        pending = []
        for iteration in range(3):
            stream = wp.Stream('cuda')
            with wp.ScopedStream(stream):
                b = wp.array(np.sin(np.arange(len(xyz)) + iteration), dtype=wp.float32)
                expected = wp.zeros_like(b)
                actual = wp.zeros_like(b)
                relax_sliced(A, b, expected, schedule, 2, 3, use_graph=False)
                relax_sliced(A, b, actual, schedule, 2, 3)
                pending.append((stream, b, expected, actual))
        # Evict with asynchronous launches outstanding on independent streams.
        sliced_graph.cache_clear()
        for stream, b, expected, actual in pending:
            wp.synchronize_stream(stream)
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize('depth', [0, 1, 3, 5])
@pytest.mark.parametrize('iters', [0, 1, 3, 8])
def test_wavefront_preserves_every_coupled_slice_order(depth, iters):
    from warp_psr.relaxation import sliced_groups, wavefront_groups
    sequential = list(sliced_groups(depth, iters))
    parallel = [4 * (first_z + 2 * offset) + color
                for first_z, count, color in wavefront_groups(depth, iters)
                for offset in range(count)]
    assert len(parallel) == len(sequential)
    # Degree-one rows couple only to their own and adjacent z slices. Compare
    # the complete sequence for every such pair, including all color visits.
    for z in range(1 << depth):
        assert [g for g in parallel if g // 4 in (z, z + 1)] == [
            g for g in sequential if g // 4 in (z, z + 1)]


@pytest.mark.cuda
@pytest.mark.parametrize('dtype', [wp.float32, wp.float64])
@pytest.mark.parametrize('iters', [0, 1, 3, 8])
@pytest.mark.parametrize('pattern', ['full', 'sparse', 'empty_slices'])
def test_wavefront_matches_serialized_updates_for_arbitrary_local_matrices(dtype, iters, pattern):
    import itertools
    import warp.sparse as sparse
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    xyz = list(itertools.product(range(9), repeat=3))
    if pattern == 'sparse':
        xyz = [c for i, c in enumerate(xyz) if i % 3 != 0]
    elif pattern == 'empty_slices':
        xyz = [c for c in xyz if c[2] not in (2, 5)]
    indices = {c: i for i, c in enumerate(xyz)}
    rows, cols, values = [], [], []
    for c, i in indices.items():
        total = 2.0
        for delta in itertools.product((-1, 0, 1), repeat=3):
            j = indices.get(tuple(a + b for a, b in zip(c, delta)))
            if j is None or j == i:
                continue
            value = np.cos(123 * i + 7 * j)
            rows.append(i); cols.append(j); values.append(value)
            total += abs(value)
        rows.append(i); cols.append(i); values.append(total)
    with wp.ScopedDevice('cuda'):
        A = sparse.bsr_from_triplets(len(xyz), len(xyz), wp.array(rows, dtype=int),
                                    wp.array(cols, dtype=int), wp.array(values, dtype=dtype))
        keys = wp.array([(3 << 48) | (x << 32) | (y << 16) | z for x, y, z in xyz], dtype=wp.int64)
        schedule = row_schedule(keys, 3, sliced=True)
        b = wp.array(np.sin(np.arange(len(xyz))), dtype=dtype)
        expected = wp.array(np.cos(np.arange(len(xyz))), dtype=dtype)
        actual = wp.clone(expected)
        relax_sliced(A, b, expected, schedule, 3, iters, use_graph=False)
        relax_sliced(A, b, actual, schedule, 3, iters)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
