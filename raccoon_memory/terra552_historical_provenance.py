"""Verify the recorded Terra training sources in a separate process.

The collector's original provenance code expects its historical physical
sources at today's workspace paths.  For this read-only training audit only,
the one changed session source is resolved to an archived, digest-matched
copy.  Every other original check remains in the collector.  The caller must
independently fingerprint the current sources used for a new execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SESSION_SOURCE = 'raccoon_memory/scene_oracle_free_session.py'
HISTORICAL_SESSION = (ROOT / 'artifacts/scene_runtime/20261104_applied_target_1103_development_v1'
                      / SESSION_SOURCE)
HISTORICAL_SESSION_SHA256 = '29c87eb1d3d221bada899875800f1e60e658b4eb5d0249f0246cdcba1babaf0a'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_hash_resolver(original_sha, current, archive, expected):
    """Resolve one exact source to verified historical bytes, never other files."""
    current, archive = Path(current).resolve(), Path(archive).resolve()
    if _digest(archive) != expected:
        raise ValueError('archived historical session fingerprint differs')

    def digest(path):
        if Path(path).resolve() == current:
            value = _digest(archive)
            if value != expected:
                raise ValueError('archived historical session changed during verification')
            return value
        return original_sha(path)

    return digest


def verify_training_groups(binding):
    """Run the unchanged retrospective audit using the original session bytes."""
    if __name__ != '__main__':
        raise RuntimeError('historical training audit must run in its own subprocess')
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import collect_terra_coverage_demonstrations as coverage
    from scripts import collect_terra_coverage_v21 as coverage_ik
    from scripts import evaluate_terra_v21_memory_session as evidence

    registration_path = coverage.OUTPUT / 'preregistration.json'
    registration_bytes = registration_path.read_bytes()
    registration_sha = hashlib.sha256(registration_bytes).hexdigest()
    if registration_sha != coverage_ik.PARENT_RELEASE_SHA:
        raise ValueError('historical training preregistration fingerprint differs')
    registration = json.loads(registration_bytes)
    recorded = registration['physical_source_sha256'][SESSION_SOURCE]
    if recorded != HISTORICAL_SESSION_SHA256:
        raise ValueError('historical session differs from the recorded training source')
    resolver = historical_hash_resolver(coverage.sha, ROOT / SESSION_SOURCE,
                                        HISTORICAL_SESSION, recorded)
    # This redirection is confined to the subprocess running this module.
    # A live collector continues to compare against its actual workspace.
    with patch.object(coverage, 'sha', side_effect=resolver):
        groups = evidence.training_groups(binding)
    if _digest(registration_path) != registration_sha or _digest(HISTORICAL_SESSION) != recorded:
        raise ValueError('historical training evidence changed during verification')
    return {'schema': 'terra552-historical-training-provenance.v1',
            'groups': sorted(groups), 'preregistration_sha256': registration_sha,
            'historical_session_sha256': recorded,
            'historical_session_path': str(HISTORICAL_SESSION),
            'current_session_sha256': _digest(ROOT / SESSION_SOURCE)}


if __name__ == '__main__':
    binding = json.load(sys.stdin)
    print(json.dumps(verify_training_groups(binding), sort_keys=True))
