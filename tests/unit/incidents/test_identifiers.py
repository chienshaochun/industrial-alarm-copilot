'''Stable incident identifier unit tests.'''

import pytest

from industrial_alarm_copilot.incidents.identifiers import build_incident_id


BASE_IDENTITY = {
    'machine_id': '4',
    'split': 'train',
    'start_time': '2019-02-21 19:57:57.532',
    'first_source_row': 42,
    'gap_minutes': 30,
}


def test_build_incident_id_has_reproducible_known_value():
    assert build_incident_id(**BASE_IDENTITY) == 'inc_a726ed82c68db78d'
    assert (
        build_incident_id(**(BASE_IDENTITY | {'gap_minutes': 30.0}))
        == 'inc_a726ed82c68db78d'
    )


@pytest.mark.parametrize(
    ('field', 'different_value'),
    [
        ('machine_id', '6'),
        ('split', 'validation'),
        ('start_time', '2019-02-21 19:57:57.533'),
        ('first_source_row', 43),
        ('gap_minutes', 60),
    ],
)
def test_build_incident_id_changes_with_episode_identity(
    field, different_value
):
    changed_identity = BASE_IDENTITY | {field: different_value}

    assert build_incident_id(**changed_identity) != build_incident_id(
        **BASE_IDENTITY
    )
