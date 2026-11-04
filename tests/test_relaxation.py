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
