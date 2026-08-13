'''Retrieval experiment setting tests.'''

import pytest

from industrial_alarm_copilot.retrieval.settings import (
    parse_retrieval_settings,
)


def _settings():
    return {
        'top_k': 5,
        'alarm_weight': 1.0,
        'shape_weight': 1.0,
        'candidate_policy': 'expanding_history',
        'future_horizon_hours_candidates': [1, 6, 24],
        'relevance_threshold_candidates': [0.1, 0.3, 0.5],
        'selected_feature_version': 'alarm_plus_shape_v1',
        'future_horizon_hours': 6,
        'relevance_threshold': 0.3,
    }


def test_parse_retrieval_settings_preserves_validation_grid():
    parsed = parse_retrieval_settings(_settings())

    assert parsed.top_k == 5
    assert parsed.candidate_policy == 'expanding_history'
    assert parsed.future_horizon_hours_candidates == (1.0, 6.0, 24.0)
    assert parsed.relevance_threshold_candidates == (0.1, 0.3, 0.5)
    assert parsed.selected_feature_version == 'alarm_plus_shape_v1'
    assert parsed.future_horizon_hours == 6.0
    assert parsed.relevance_threshold == 0.3


def test_parse_retrieval_settings_rejects_invalid_threshold():
    settings = _settings()
    settings['relevance_threshold_candidates'] = [0]

    with pytest.raises(ValueError, match='thresholds'):
        parse_retrieval_settings(settings)


def test_parse_retrieval_settings_rejects_unvalidated_selection():
    settings = _settings()
    settings['future_horizon_hours'] = 12

    with pytest.raises(ValueError, match='horizon'):
        parse_retrieval_settings(settings)
