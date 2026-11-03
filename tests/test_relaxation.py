"""The compact GPU schedule must preserve the verified relaxation arithmetic."""
import numpy as np
import pytest
import warp as wp
from warp_psr.forms import uniform_space,assemble_stiffness
from warp_psr.solver import relax
from warp_psr.relaxation import row_schedule,relax_scheduled

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
        for forward in [True,False]:
            relax(A,b,reference,keys,3,forward=forward)
            relax_scheduled(A,b,actual,schedule,3,forward=forward)
            np.testing.assert_array_equal(actual.numpy(),reference.numpy())
        schedule=row_schedule(keys,2,sliced=True)
        for front in range(4,-3,-1):
            for z in range(max(front,0),min(front+3,5)):
                relax(A,b,reference,keys,1,z=z)
                relax_scheduled(A,b,actual,schedule,1,z=z)
        np.testing.assert_array_equal(actual.numpy(),reference.numpy())
