'''Forecast label coverage and support profile tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.forecasting.profile import (
    assign_forecast_support_groups,
    build_forecast_alarm_support_profile,
    build_forecast_split_profile,
    summarize_forecast_support_groups,
)
from industrial_alarm_copilot.forecasting.vocabulary import (
    encode_forecast_labels,
    fit_forecast_label_vocabulary,
)


def _build_labels():
    return pd.DataFrame(
        {
            'incident_id': [
                'train_complete',
                'train_empty',
                'train_incomplete',
                'validation_complete',
            ],
            'split': ['train', 'train', 'train', 'validation'],
            'outcome_is_complete': [True, True, False, True],
            'future_alarm_codes': [
                ('98', '11'),
                (),
                ('137',),
                ('26', '98'),
            ],
            'future_alarm_counts': [
                (('98', 2), ('11', 1)),
                (),
                (('137', 1),),
                (('26', 3), ('98', 1)),
            ],
            'has_future_alarms': [True, False, True, True],
        }
    )


def test_build_forecast_split_profile_separates_coverage_and_unknowns():
    labels = _build_labels()
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)

    profile = build_forecast_split_profile(labels, encoded).set_index(
        'split'
    )

    train = profile.loc['train']
    assert train['episode_count'] == 3
    assert train['complete_outcome_count'] == 2
    assert train['outcome_coverage'] == pytest.approx(2 / 3)
    assert train['complete_with_future_alarm_count'] == 1
    assert train['complete_empty_outcome_count'] == 1
    assert train['complete_empty_outcome_share'] == pytest.approx(0.5)
    assert train['mean_known_label_count'] == pytest.approx(1.0)
    assert train['unknown_label_query_count'] == 0

    validation = profile.loc['validation']
    assert validation['complete_outcome_count'] == 1
    assert validation['mean_known_label_count'] == pytest.approx(1.0)
    assert validation['unknown_label_query_count'] == 1
    assert validation['unknown_label_query_share'] == pytest.approx(1.0)
    assert validation['unknown_alarm_code_occurrence_count'] == 1
    assert validation['unknown_alarm_event_count'] == 3


def test_build_forecast_alarm_support_profile_uses_sample_support():
    labels = _build_labels()
    vocabulary = fit_forecast_label_vocabulary(labels)

    profile = build_forecast_alarm_support_profile(vocabulary)

    assert profile['alarm_code'].tolist() == ['11', '98']
    assert profile['support_rank'].tolist() == [1, 2]
    assert profile['train_sample_support'].tolist() == [1, 1]
    assert profile['train_sample_share'].tolist() == pytest.approx([0.5, 0.5])


def test_forecast_support_groups_use_fixed_train_boundaries():
    profile = pd.DataFrame(
        {
            'alarm_code': ['rare_low', 'rare_edge', 'medium', 'common'],
            'train_sample_support': [1, 49, 50, 500],
        }
    )

    grouped = assign_forecast_support_groups(
        profile,
        rare_max_train_support=49,
        common_min_train_support=500,
    )
    summary = summarize_forecast_support_groups(grouped).set_index(
        'support_group'
    )

    assert grouped['support_group'].tolist() == [
        'rare',
        'rare',
        'medium',
        'common',
    ]
    assert summary.loc['rare', 'label_count'] == 2
    assert summary.loc['medium', 'label_count'] == 1
    assert summary.loc['common', 'label_count'] == 1
    assert summary.loc['rare', 'positive_sample_count'] == 50
    assert summary['label_share'].sum() == pytest.approx(1.0)
    assert summary['positive_sample_share'].sum() == pytest.approx(1.0)
