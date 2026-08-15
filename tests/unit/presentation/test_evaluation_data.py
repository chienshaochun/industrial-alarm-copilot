'''Locked evaluation view-model tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.presentation.data import build_evaluation_data


def _inputs():
    retrieval = pd.DataFrame(
        [{'evaluation_split': 'test', 'mean_hit_at_k': 0.7}]
    )
    forecast = pd.DataFrame(
        [{'split': 'test', 'model_version': 'transition_frequency_v1'}]
    )
    support = pd.DataFrame(
        {'support_group': ['rare', 'medium', 'common']}
    )
    return retrieval, forecast, support


def test_build_evaluation_data_accepts_only_locked_test_artifacts():
    evaluation = build_evaluation_data(*_inputs())

    assert evaluation.retrieval['mean_hit_at_k'] == 0.7
    assert evaluation.forecasting['model_version'] == (
        'transition_frequency_v1'
    )
    assert evaluation.support_groups['support_group'].tolist() == [
        'rare',
        'medium',
        'common',
    ]


def test_build_evaluation_data_rejects_validation_as_final_result():
    retrieval, forecast, support = _inputs()
    retrieval.loc[0, 'evaluation_split'] = 'validation'

    with pytest.raises(ValueError, match='test split'):
        build_evaluation_data(retrieval, forecast, support)
