'''Forecast model selection policy tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.forecasting.selection import select_forecast_model


def test_selection_balances_macro_then_maximizes_micro_f1():
    metrics = pd.DataFrame(
        {
            'split': ['validation'] * 3,
            'model_version': [
                'machine_frequency_v1',
                'transition_frequency_v1',
                'gru_balanced_capped_v1',
            ],
            'micro_f1_at_k': [0.56, 0.55, 0.45],
            'macro_f1_at_k': [0.08, 0.093, 0.099],
        }
    )

    selected = select_forecast_model(metrics, macro_f1_tolerance=0.01)

    assert selected['model_version'] == 'transition_frequency_v1'
    assert selected['eligible_model_count'] == 2
    assert selected['best_macro_f1'] == pytest.approx(0.099)


def test_selection_rejects_test_metrics():
    metrics = pd.DataFrame(
        {
            'split': ['test'],
            'model_version': ['transition_frequency_v1'],
            'micro_f1_at_k': [0.5],
            'macro_f1_at_k': [0.1],
        }
    )

    with pytest.raises(ValueError, match='validation'):
        select_forecast_model(metrics, macro_f1_tolerance=0.01)
