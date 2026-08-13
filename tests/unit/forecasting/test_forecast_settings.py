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
    }


def test_parse_forecast_settings_preserves_selected_contract():
    parsed = parse_forecast_settings(_settings())

    assert parsed.top_k == 5
    assert parsed.forecast_horizon_hours_candidates == (1.0, 6.0, 24.0)
    assert parsed.selected_forecast_horizon_hours == 6.0
    assert parsed.rare_max_train_support == 49
    assert parsed.common_min_train_support == 500


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
