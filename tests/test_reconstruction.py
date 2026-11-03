"""Independent production reconstruction checked against unchanged C++ outputs."""
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from tools.check_reconstruction import check


@pytest.mark.cuda
@pytest.mark.reference
@pytest.mark.parametrize('name',['sphere','sphere-psr','sphere-exact','sparse','torus'])
def test_independent_reconstruction(name):
    wp.init()
    if not wp.is_cuda_available():pytest.skip('CUDA unavailable')
    directory=Path('.build/fixtures')/name
    if not (directory/'oracle.txt').exists():pytest.skip('Generate reference fixtures')
    report=check(directory)
    assert report['reconstruction_compatibility']


@pytest.mark.reference
def test_pinned_exact_constraint_bug():
    directory=Path('.build/fixtures/sphere-exact')
    if not (directory/'operators.txt').exists():pytest.skip('Generate operator exports')
    assert not any(line.startswith('point ') for line in (directory/'operators.txt').read_text().splitlines())
    psr=Path('.build/fixtures/sphere-psr/oracle.txt')
    if psr.exists():
        from tools.make_fixture import read_output
        exact=read_output(directory/'oracle.txt');unscreened=read_output(psr)
        np.testing.assert_array_equal(exact['q'],unscreened['q'])
        np.testing.assert_array_equal(exact['iso'],unscreened['iso'])


def test_geometry_span_rejected_before_allocation():
    from warp_psr import reconstruct
    with pytest.raises(ValueError,match='at most 8 integration levels'):
        reconstruct(None,None,depth=12,full_depth=3)
