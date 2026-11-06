import json
import tarfile

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


def _change_acceptance(accepted_fixture, change):
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


@pytest.mark.parametrize('change', ['parameters', 'precision', 'source', 'failed', 'tree', 'legacy'])
def test_stale_or_incomplete_acceptance(accepted_fixture, change):
    _change_acceptance(accepted_fixture, change)
    with pytest.raises(ValueError, match='Stale reconstruction acceptance'):
        require_acceptance(accepted_fixture)


def test_unchanged_acceptance_is_reusable(accepted_fixture):
    assert require_acceptance(accepted_fixture)['reconstruction_compatibility']


def _archive_fixture(directory):
    archive_path = directory/'fixture.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for path in sorted(directory.iterdir()):
            if path != archive_path:
                archive.add(path, arcname='fixture/'+path.name, recursive=False)
    return archive_path


def test_accepted_archive_matches_directory_without_extraction(accepted_fixture, monkeypatch):
    from tools.check_differentiable_fixture import accepted_data
    import numpy as np
    expected = accepted_data(accepted_fixture)
    archive_path = _archive_fixture(accepted_fixture)
    before = sorted(accepted_fixture.iterdir())
    def no_extraction(*args, **kwargs):
        raise AssertionError('Fixture checking must not extract files')
    monkeypatch.setattr(tarfile.TarFile, 'extract', no_extraction)
    monkeypatch.setattr(tarfile.TarFile, 'extractall', no_extraction)
    actual = accepted_data(archive_path)
    assert expected[0] == actual[0]
    np.testing.assert_array_equal(expected[1], actual[1])
    np.testing.assert_array_equal(expected[2], actual[2])
    assert sorted(accepted_fixture.iterdir()) == before


@pytest.mark.parametrize('name', ['input.pts', 'queries.txt', *REFERENCE_FILES])
def test_archive_rejects_changed_bytes(accepted_fixture, name):
    from tools.check_differentiable_fixture import accepted_data
    path = accepted_fixture/name
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        accepted_data(_archive_fixture(accepted_fixture))


@pytest.mark.parametrize('change', ['parameters', 'precision', 'source', 'failed', 'tree', 'legacy'])
def test_archive_rejects_stale_acceptance(accepted_fixture, change):
    from tools.check_differentiable_fixture import accepted_data
    _change_acceptance(accepted_fixture, change)
    with pytest.raises(ValueError, match='Stale reconstruction acceptance'):
        accepted_data(_archive_fixture(accepted_fixture))


@pytest.mark.parametrize('member_name,link', [('fixture/input.pts', False), ('../escape', False),
                                            ('second/extra', False), ('fixture/link', True)])
def test_archive_rejects_ambiguous_members(accepted_fixture, member_name, link):
    from tools.check_differentiable_fixture import accepted_data
    archive_path = accepted_fixture/'fixture.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for path in sorted(accepted_fixture.iterdir()):
            if path != archive_path:
                archive.add(path, arcname='fixture/'+path.name, recursive=False)
        member = tarfile.TarInfo(member_name)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = 'input.pts'
        archive.addfile(member)
    with pytest.raises(ValueError):
        accepted_data(archive_path)
