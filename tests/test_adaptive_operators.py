"""Reference matrices, loads, constraints and transfers on an adaptive GPU tree."""
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import pytest
import warp as wp
import warp.sparse as sparse
from warp_psr import reconstruct
from warp_psr.solver import prolongation
from tools.make_fixture import read_output


def csr(a):
    return csr_matrix((a.values.numpy().ravel(),a.columns.numpy(),a.offsets.numpy()),shape=(a.nrow,a.ncol))

@pytest.mark.reference
@pytest.mark.cuda
@pytest.mark.parametrize('name,weight,exact',[('sphere',2.,False),('sphere-psr',0.,False),('sphere-exact',2.,True)])
def test_level_operators_and_transfers(name,weight,exact):
    wp.init()
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    directory=Path('.build/fixtures')/name
    if not (directory/'operators.txt').exists():pytest.skip('Generate operator exports')
    data=np.loadtxt(directory/'input.pts').astype('f')
    with wp.ScopedDevice('cuda'):
        result=reconstruct(wp.array(data[:,:3],dtype=wp.vec3),wp.array(data[:,3:],dtype=wp.vec3),depth=4,full_depth=2,point_weight=weight,exact_interpolation=exact)
        tree=result.diagnostics['tree']
        nodes=read_output(directory/'oracle_instrumented.txt')['node']
        lookup={int(row[6]):tuple(map(int,row[:4])) for row in nodes}
        indices=[{(d,(int(k)>>32)&65535,(int(k)>>16)&65535,int(k)&65535):i for i,k in enumerate(level.keys.numpy())} for d,level in enumerate(tree.basis)]
        ops=[line.split() for line in (directory/'operators.txt').read_text().splitlines()]
        for level in result.diagnostics['levels']:
            d=level['depth'];actual=csr(level['matrix']);rr=[];cc=[];vv=[]
            for row in ops:
                if row[0]!='A' or int(row[1])!=d:continue
                i=indices[d].get(lookup[int(row[2])]);j=indices[d].get(lookup[int(row[3])])
                if i is not None and j is not None:rr.append(i);cc.append(j);vv.append(float(row[4]))
            expected=csr_matrix((vv,(rr,cc)),shape=actual.shape)
            # Normalized random-vector actions test accumulated operator error.
            x=np.random.default_rng(d).normal(size=(actual.shape[1],4))
            lhs,rhs=actual@x,expected@x
            scale=np.max(np.abs(rhs))
            np.testing.assert_allclose(lhs/scale,rhs/scale,atol=1e-6,rtol=1e-5)
            np.testing.assert_allclose((actual-actual.T).data,0,atol=1e-6)
        for d in range(1,5):
            actual=csr(prolongation(tree.basis[d-1].keys,tree.basis[d].keys,d)).T
            rr=[];cc=[];vv=[]
            for row in ops:
                if row[0]!='R' or int(row[1])!=d:continue
                i=indices[d-1].get(lookup[int(row[2])]);j=indices[d].get(lookup[int(row[3])])
                if i is not None and j is not None:rr.append(i);cc.append(j);vv.append(float(row[4]))
            expected=csr_matrix((vv,(rr,cc)),shape=actual.shape)
            assert (actual-expected).nnz==0
        reference_transform=read_output(directory/'oracle.txt')['transform'].reshape(4,4).T
        np.testing.assert_array_equal(result.metadata['cube_to_model'],reference_transform)
