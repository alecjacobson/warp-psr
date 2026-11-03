"""Explicit oracle defaults preserve the previously frozen reference run."""
import json
import os
import subprocess
import numpy as np
import pytest
from tools.make_fixture import ROOT, oracle_parameters, read_output


@pytest.mark.reference
def test_explicit_defaults_match_legacy_oracle_arguments(tmp_path):
    directory=ROOT/'.build/fixtures/sphere'
    if not (directory/'manifest.json').exists() or not (ROOT/'.build/oracle').exists():
        pytest.skip('Build oracle and generate sphere fixture')
    parameters=json.loads((directory/'manifest.json').read_text())['parameters']
    arguments=oracle_parameters(parameters)
    results=[]
    for name,settings in [('legacy',arguments[:6]),('explicit',arguments)]:
        output=tmp_path/(name+'.txt')
        environment={**os.environ,'WARP_PSR_REFERENCE_PARALLELISM':'none'}
        subprocess.run([str(ROOT/'.build/oracle'),str(directory/'input.pts'),str(directory/'queries.txt'),str(output),*settings],env=environment,check=True)
        results.append(read_output(output))
        assert results[-1]['parallel_async'][0,0]==0
        assert results[-1]['threads'][0,0]>=1
    for key in ('q','iso','transform'):
        np.testing.assert_array_equal(results[0][key],results[1][key])
