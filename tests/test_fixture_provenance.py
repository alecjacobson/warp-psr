import json

import pytest

from tools.datasets import checksum
from tools.fixture_provenance import (REFERENCE_FILES, production_hashes,
                                      require_acceptance, verify_fixture)


@pytest.fixture
def accepted_fixture(tmp_path):
    for name in ('input.pts', 'queries.txt', *REFERENCE_FILES):
        (tmp_path/name).write_text('0 0 0 0 0 1\n')
    manifest = {'input_sha256': checksum(tmp_path/'input.pts'),
                'queries_sha256': checksum(tmp_path/'queries.txt'),
                'reference_file_sha256': {name: checksum(tmp_path/name) for name in REFERENCE_FILES},
                'precision': 'float', 'parameters': {'depth': 4, 'iters': 8}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    report = {'reconstruction_compatibility': True,
              'tree': {'exact_active_and_support_keys': True},
              'fixture_provenance': verify_fixture(tmp_path),
              'production_source_sha256': production_hashes()}
    (tmp_path/'reconstruction_report.json').write_text(json.dumps(report))
    return tmp_path


@pytest.mark.parametrize('name', ['input.pts', 'queries.txt', *REFERENCE_FILES])
def test_changed_bytes_reject_acceptance(accepted_fixture, name):
    path = accepted_fixture/name
    path.write_text(path.read_text()+'\n')  # Same numbers, different provenance.
    with pytest.raises(ValueError, match='checksum mismatch'):
        require_acceptance(accepted_fixture)


@pytest.mark.parametrize('change', ['parameters', 'precision', 'source', 'failed', 'tree', 'legacy'])
def test_stale_or_incomplete_acceptance(accepted_fixture, change):
    manifest_path = accepted_fixture/'manifest.json'
    report_path = accepted_fixture/'reconstruction_report.json'
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    if change == 'parameters':
        manifest['parameters']['iters'] = 16
    elif change == 'precision':
        manifest['precision'] = 'double'
    elif change == 'source':
        report['production_source_sha256'] = {}
    elif change == 'failed':
        report['reconstruction_compatibility'] = False
    elif change == 'tree':
        report['tree']['exact_active_and_support_keys'] = False
    else:
        del report['fixture_provenance']
    manifest_path.write_text(json.dumps(manifest))
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Stale reconstruction acceptance'):
        require_acceptance(accepted_fixture)


def test_unchanged_acceptance_is_reusable(accepted_fixture):
    assert require_acceptance(accepted_fixture)['reconstruction_compatibility']
