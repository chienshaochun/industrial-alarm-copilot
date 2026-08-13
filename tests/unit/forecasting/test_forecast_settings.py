'''Forecast experiment setting tests.'''

import pytest

from industrial_alarm_copilot.forecasting.settings import (
    parse_forecast_settings,
)


def _settings():
    return {
        'top_k': 5,
        'forecast_horizon_hours_candidates': [1, 6, 24],
        'selected_forecast_horizon_hours': 6,
        'rare_max_train_support': 49,
        'common_min_train_support': 500,
        'minimum_machine_train_samples': 200,
        'minimum_transition_train_samples': 20,
        'linear_c': 1.0,
        'linear_max_iter': 300,
        'linear_class_weight_candidates': ['none', 'balanced'],
        'sequence_max_length': 64,
        'sequence_embedding_dim': 24,
        'sequence_hidden_dim': 48,
        'sequence_machine_embedding_dim': 8,
        'sequence_batch_size': 256,
        'sequence_epochs': 8,
        'sequence_learning_rate': 0.001,
        'sequence_weight_candidates': ['none', 'balanced_capped'],
        'sequence_positive_weight_cap': 20.0,
        'random_seed': 0,
        'selection_macro_f1_tolerance': 0.01,
        'selected_model_version': 'transition_frequency_v1',
    }


def test_parse_forecast_settings_preserves_selected_contract():
    parsed = parse_forecast_settings(_settings())

    assert parsed.top_k == 5
    assert parsed.forecast_horizon_hours_candidates == (1.0, 6.0, 24.0)
    assert parsed.selected_forecast_horizon_hours == 6.0
    assert parsed.rare_max_train_support == 49
    assert parsed.common_min_train_support == 500
    assert parsed.minimum_machine_train_samples == 200
    assert parsed.minimum_transition_train_samples == 20
    assert parsed.linear_c == 1.0
    assert parsed.linear_max_iter == 300
    assert parsed.linear_class_weight_candidates == ('none', 'balanced')
    assert parsed.sequence_max_length == 64
    assert parsed.sequence_hidden_dim == 48
    assert parsed.sequence_weight_candidates == ('none', 'balanced_capped')
    assert parsed.random_seed == 0
    assert parsed.selection_macro_f1_tolerance == 0.01
    assert parsed.selected_model_version == 'transition_frequency_v1'


def test_parse_forecast_settings_rejects_unstudied_horizon():
    settings = _settings()
    settings['selected_forecast_horizon_hours'] = 12

    with pytest.raises(ValueError, match='horizon'):
        parse_forecast_settings(settings)


def test_parse_forecast_settings_requires_medium_support_range():
    settings = _settings()
    settings['common_min_train_support'] = 50

    with pytest.raises(ValueError, match='medium'):
        parse_forecast_settings(settings)
