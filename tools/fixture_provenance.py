"""Bind offline acceptance to the exact fixture and production implementation."""
import json
from pathlib import Path

from tools.datasets import checksum


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILES = ('oracle.txt', 'oracle_instrumented.txt', 'preprocess.txt',
                   'stages.txt', 'splats.txt')


def production_hashes():
    return {str(path.relative_to(ROOT)): checksum(path)
            for path in sorted(ROOT.glob('src/**/*.py'))}


def verify_fixture(directory, manifest=None):
    directory = Path(directory)
    if manifest is None:
        manifest = json.loads((directory/'manifest.json').read_text())
    for name, key in [('input.pts', 'input_sha256'), ('queries.txt', 'queries_sha256')]:
        if checksum(directory/name) != manifest[key]:
            raise ValueError(f'{name}: fixture checksum mismatch; regenerate the fixture')
    references = manifest.get('reference_file_sha256', {})
    if not all(name in references for name in REFERENCE_FILES):
        raise ValueError('Reference export checksums missing; regenerate the fixture')
    for name, digest in references.items():
        if checksum(directory/name) != digest:
            raise ValueError(f'{name}: reference checksum mismatch; regenerate the fixture')
    return {key: manifest[key] for key in ('input_sha256', 'queries_sha256', 'precision',
                                          'parameters', 'reference_file_sha256')}


def require_acceptance(directory):
    """Reject stale acceptance before expensive benchmarks or visualization."""
    directory = Path(directory)
    identity = verify_fixture(directory)
    report = json.loads((directory/'reconstruction_report.json').read_text())
    if (not report.get('reconstruction_compatibility')
            or not report.get('tree', {}).get('exact_active_and_support_keys')
            or report.get('fixture_provenance') != identity
            or report.get('production_source_sha256') != production_hashes()):
        raise ValueError('Stale reconstruction acceptance; rerun tools.check_reconstruction')
    return report
