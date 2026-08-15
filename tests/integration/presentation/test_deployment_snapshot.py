'''Integrity checks for the public deployment artifact snapshot.'''

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_DIRECTORY = PROJECT_ROOT / 'data' / 'deployment'
EXPECTED_SHA256 = {
    'events.parquet': (
        'd6393aae5b781176f9d02ca75a78c0f6c1a3303b502ac24ad20725ee3fb01161'
    ),
    'incidents.parquet': (
        '6861a919fe257bef1f7caa3e71e91ab9c3958bfb911a2c45c33c36ee49949393'
    ),
    'incident_events.parquet': (
        'd40b79d6ddd52030aa44269934d1288a3d2cb9cc3dc996dcb3861a995f5959db'
    ),
    'forecast_model.json': (
        '1bd17eb1e7e4249cd9aee8a0e648442e44ae41842952079336ced05c9679551d'
    ),
    'retrieval_test_results.csv': (
        'e30093a73e29c258227e91c0cf1c489a2635545d1bafd46d881ac4bac0f7fc3f'
    ),
    'forecast_test_results.csv': (
        'f618337aff377f9242821c2351b76ea0dff76746faa3d9dfc994443d00d3fec0'
    ),
    'forecast_test_support_groups.csv': (
        '3aa361b1009eff28147c088ab37f7851c211d7de8767c85f592f76d43f3d2de3'
    ),
}


def _sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix in {'.json', '.csv'}:
        content = content.replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()


def test_deployment_snapshot_matches_reviewed_hashes():
    actual = {
        filename: _sha256(DEPLOYMENT_DIRECTORY / filename)
        for filename in EXPECTED_SHA256
    }

    assert actual == EXPECTED_SHA256


def test_deployment_snapshot_preserves_application_contract():
    events = pd.read_parquet(DEPLOYMENT_DIRECTORY / 'events.parquet')
    incidents = pd.read_parquet(DEPLOYMENT_DIRECTORY / 'incidents.parquet')
    mapping = pd.read_parquet(
        DEPLOYMENT_DIRECTORY / 'incident_events.parquet'
    )
    with (DEPLOYMENT_DIRECTORY / 'forecast_model.json').open(
        encoding='utf-8'
    ) as model_file:
        model = json.load(model_file)

    assert events.shape == (444_834, 9)
    assert len(incidents) == 38_153
    assert len(mapping) == len(events)
    assert model['model_version'] == 'transition_frequency_v1'
    assert len(model['alarm_codes']) == 139
