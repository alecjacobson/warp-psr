"""Report checks keep discrete identity and centered-field budgets independent."""
import numpy as np
from tools.check_repeatability import digest_keys, reference_keys, errors


def test_canonical_keys_ignore_node_order_and_incidental_indices():
    nodes=np.array([[1,0,0,0,2,0,100,0],[1,1,0,0,2,0,1000,0],[1,9,0,0,0,1,-1,0]])
    shuffled=nodes[::-1].copy()
    shuffled[:,6]=[73,45,21]
    assert digest_keys(reference_keys(nodes))==digest_keys(reference_keys(shuffled))
    changed=nodes.copy()
    changed[0,2]=1
    assert digest_keys(reference_keys(nodes))!=digest_keys(reference_keys(changed))


def test_centered_value_budget_can_fail_when_raw_value_passes():
    reference=np.array([[1.,0.,0.,0.]])
    actual=reference.copy()
    actual[0,0]+=2e-5
    result=errors(actual,1.,reference,1.,np.eye(3),np.ones(1),1e-5,1e-4)
    assert result['budget_ratios']['value']<1
    assert result['budget_ratios']['centered_value']>1
    assert not result['within_budget']
