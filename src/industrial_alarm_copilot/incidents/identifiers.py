'''Stable identifiers for derived alarm episodes.'''

import hashlib
import json

import pandas as pd


INCIDENT_SCHEMA_VERSION = 1
INCIDENT_SOURCE = 'time_gap_heuristic'


def build_incident_id(
    *,
    machine_id: str,
    split: str,
    start_time: pd.Timestamp | str,
    first_source_row: int,
    gap_minutes: float,
    schema_version: int = INCIDENT_SCHEMA_VERSION,
    incident_source: str = INCIDENT_SOURCE,
) -> str:
    '''Build a reproducible SHA-256-based identifier for one episode.'''
    identity = {
        'schema_version': int(schema_version),
        'incident_source': str(incident_source),
        'gap_minutes': format(float(gap_minutes), '.12g'),
        'machine_id': str(machine_id),
        'split': str(split),
        'start_time': pd.Timestamp(start_time).isoformat(
            timespec='microseconds'
        ),
        'first_source_row': int(first_source_row),
    }
    serialized_identity = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    )
    digest = hashlib.sha256(serialized_identity.encode('utf-8')).hexdigest()
    return f'inc_{digest[:16]}'
