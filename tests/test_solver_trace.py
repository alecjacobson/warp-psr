from pathlib import Path
import pytest
from tools.check_solver_trace import check


@pytest.mark.reference
@pytest.mark.parametrize('precision',['float','double'])
@pytest.mark.parametrize('accuracy',['loose','default','tight'])
def test_read_only_solver_trace_replays(precision,accuracy):
    path=Path('.build/fixtures')/f'solver-trace-{accuracy}-{precision}'
    if not (path/'solve_trace.txt').exists():
        pytest.skip('Generate opt-in solver trace fixture')
    report=check(path)
    assert report['instrumentation_max_abs_error']==0
    assert set(report['gs_phases'])=={'restrict','prolong','slice'}
    if accuracy=='tight':
        assert any(s['iterations']>0 for s in report['cg_stops'])
    if accuracy=='loose':
        assert all(s['reason']=='initial' for s in report['cg_stops'])
