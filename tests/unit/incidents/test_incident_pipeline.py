'''In-memory incident analysis pipeline unit tests.'''

import pandas as pd

from industrial_alarm_copilot.incidents.pipeline import (
    build_incident_analysis,
)


def test_build_incident_analysis_connects_summaries_mappings_and_baselines():
    events = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(
                [
                    '2019-01-01 10:00:00',
                    '2019-01-01 10:05:00',
                    '2019-01-01 10:40:00',
                    '2019-01-01 10:41:00',
                    '2019-01-01 09:00:00',
                    '2019-01-01 09:10:00',
                ]
            ),
            'alarm_code': ['11', '11', '26', '98', '31', '31'],
            'machine_id': ['4', '4', '4', '4', '19', '19'],
            'source_row': [0, 1, 2, 3, 4, 5],
            'split': [
                'train',
                'train',
                'train',
                'validation',
                'train',
                'validation',
            ],
            'gap_seconds': [
                float('nan'),
                300.0,
                2100.0,
                60.0,
                float('nan'),
                600.0,
            ],
            'is_exact_duplicate': [False] * 6,
        }
    )

    analysis = build_incident_analysis(
        events,
        gap_minutes=30,
        baseline_quantile=0.95,
        minimum_incident_count=2,
    )

    assert len(analysis.incidents) == 5
    assert analysis.incidents['event_count'].sum() == len(events)
    assert len(analysis.incident_events) == len(events)
    assert analysis.incident_events['source_row'].is_unique
    assert set(analysis.incident_events['incident_id']) == set(
        analysis.incidents['incident_id']
    )
    assert analysis.global_baseline.incident_count == 3
    assert analysis.machine_baselines['incident_count'].tolist() == [1, 2]
    assert analysis.incidents.groupby('machine_id')[
        'baseline_scope'
    ].first().to_dict() == {
        '19': 'global_fallback',
        '4': 'machine',
    }
