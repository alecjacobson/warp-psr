"""Tensorize reference exact 1D integrals; compare real Warp FEM assembly."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
import warp as wp
import warp.fem as fem
from warp_psr.forms import uniform_space, assemble_stiffness, assemble_screening, value, gradient

ROOT = Path(__file__).resolve().parents[1]


def dense(a):
    from scipy.sparse import csr_matrix
    return csr_matrix((a.values.numpy().reshape(-1), a.columns.numpy(), a.offsets.numpy()),
                      shape=(a.nrow, a.ncol)).toarray()


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    with wp.ScopedDevice(request.param):
        yield request.param


@pytest.mark.reference
def test_stiffness_against_reference(device):
    exe = ROOT / ".build/basis"
    if not exe.exists():
        pytest.skip("Run tools/build_reference.py")
    depth = 2
    rows = [line.split() for line in subprocess.check_output([str(exe), str(depth)], text=True).splitlines()]
    m, k = np.zeros((5,5)), np.zeros((5,5))
    for tag, i, j, a, b in rows:
        if tag == "a":
            m[int(i),int(j)], k[int(i),int(j)] = float(a), float(b)
    s = uniform_space(depth, wp.float32)
    p = np.rint(s.node_positions().numpy()*4).astype(int)
    expected = np.zeros((len(p), len(p)))
    for axis in range(3):
        term = np.ones_like(expected)
        for a in range(3):
            term *= (k if a == axis else m)[p[:,a,None], p[None,:,a]]
        expected += term
    actual = dense(assemble_stiffness(s, device))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(actual.sum(axis=1), 0, atol=1e-6)


def test_point_weights_and_basis(device):
    s = uniform_space(2)
    positions = np.array([[.017,.231,.673], [.991,.823,.401], [.41,.57,.69]], dtype=np.float32)
    weights = np.array([.31, 2.7, 4.1], dtype=np.float32)
    nodes = s.node_positions().numpy()
    phi = np.prod(np.maximum(1-np.abs(positions[:,None]-nodes[None])*4, 0), axis=2)
    q = wp.array(positions, dtype=wp.vec3, device=device)
    w = wp.array(weights, device=device)
    actual = dense(assemble_screening(s, q, w))
    np.testing.assert_allclose(actual, phi.T @ (weights[:,None]*phi), atol=1e-6, rtol=1e-5)
    # Nontrivial coefficients exercise all basis functions, including boundary hats.
    coefficients = np.sin(np.arange(len(nodes))).astype(np.float32)
    field = s.make_field()
    field.dof_values = wp.array(coefficients, device=device)
    quad = fem.PicQuadrature(fem.Cells(s.geometry), q, measures=w)
    out = wp.empty(len(q), dtype=wp.float32, device=device)
    grad = wp.empty(len(q), dtype=wp.vec3, device=device)
    fem.interpolate(value, dest=out, at=quad, fields={"u": field})
    fem.interpolate(gradient, dest=grad, at=quad, fields={"u": field})
    np.testing.assert_allclose(out.numpy(), phi @ coefficients, atol=1e-6, rtol=1e-5)
    dx = (positions[:,None]-nodes[None])*4
    hat = np.maximum(1-np.abs(dx), 0)
    expected = np.zeros((len(positions), 3))
    for a in range(3):
        derivative = np.where(np.abs(dx[:,:,a])<1, -4*np.sign(dx[:,:,a]), 0)
        for b in range(3):
            if b != a:
                derivative *= hat[:,:,b]
        expected[:,a] = derivative @ coefficients
    np.testing.assert_allclose(grad.numpy(), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.reference
def test_cross_level_integrals(device):
    """Common fine partition plus exact prolongation, including boundary functions.

    Selecting a local subset also checks a coarse/fine interface without assuming
    that a redundant union of all levels is an independent global solve space.
    """
    exe=ROOT/".build/basis"
    if not exe.exists():
        pytest.skip("Run tools/build_reference.py")
    rows=[line.split() for line in subprocess.check_output([str(exe),"2"],text=True).splitlines()]
    cm,ck=np.zeros((3,5)),np.zeros((3,5))
    for tag,i,j,a,b in rows:
        if tag=="c":
            cm[int(i),int(j)],ck[int(i),int(j)]=float(a),float(b)
        if tag=="b":
            i,x,v,g=int(i),float(j),float(a),float(b)
            np.testing.assert_allclose(v,max(1-abs(4*x-i),0),atol=1e-14)
            expected=-4*np.sign(4*x-i) if abs(4*x-i)<1 else 0
            np.testing.assert_allclose(g,expected,atol=1e-14)
    coarse,fine=uniform_space(1),uniform_space(2)
    cp=np.rint(coarse.node_positions().numpy()*2).astype(int)
    fp=np.rint(fine.node_positions().numpy()*4).astype(int)
    prolongation=np.prod(np.maximum(1-np.abs(fp[:,None]/2-cp[None]),0),axis=2)
    actual=prolongation.T@dense(assemble_stiffness(fine,device))
    expected=np.zeros_like(actual)
    for axis in range(3):
        term=np.ones_like(expected)
        for a in range(3):
            term*=(ck if a==axis else cm)[cp[:,a,None],fp[None,:,a]]
        expected+=term
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
    selected=np.all(fp<=2,axis=1)
    # Local refinement functions couple to coarse functions across their support.
    np.testing.assert_allclose(actual[:,selected],expected[:,selected],atol=1e-6,rtol=1e-5)
