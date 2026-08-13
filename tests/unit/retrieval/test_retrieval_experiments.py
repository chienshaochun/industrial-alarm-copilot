'''Validation-only retrieval experiment tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.experiments import (
    run_selected_test_evaluation,
    run_validation_experiment_grid,
)
from industrial_alarm_copilot.retrieval.settings import (
    RetrievalExperimentSettings,
)


def test_run_validation_experiment_grid_expands_only_validation_scores():
    incidents = pd.DataFrame(
        {
            'incident_id': ['train_a', 'train_b', 'validation', 'test'],
            'machine_id': ['4', '6', '4', '4'],
            'start_time': pd.to_datetime(
                [
                    '2020-01-01 07:00:00',
                    '2020-01-01 08:00:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 14:00:00',
                ]
            ),
            'end_time': pd.to_datetime(
                [
                    '2020-01-01 07:30:00',
                    '2020-01-01 08:30:00',
                    '2020-01-01 10:30:00',
                    '2020-01-01 14:30:00',
                ]
            ),
            'split': ['train', 'train', 'validation', 'test'],
            'event_count': [1, 2, 2, 1],
            'duration_seconds': [0, 30, 20, 0],
            'distinct_alarm_count': [1, 2, 2, 1],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'alarm_document': ['26', '98 11', '98 11', '98'],
        }
    )
    events = pd.DataFrame(
        {
            'machine_id': ['4', '4', '6', '6', '4', '4', '4', '4'],
            'timestamp': pd.to_datetime(
                [
                    '2020-01-01 08:00:00',
                    '2020-01-01 09:45:00',
                    '2020-01-01 09:00:00',
                    '2020-01-01 13:00:00',
                    '2020-01-01 11:00:00',
                    '2020-01-01 13:00:00',
                    '2020-01-01 15:00:00',
                    '2020-01-01 17:00:00',
                ],
                format='mixed',
            ),
            'alarm_code': ['98', '11', '26', '26', '98', '98', '11', '98'],
        }
    )
    settings = RetrievalExperimentSettings(
        top_k=5,
        alarm_weight=1.0,
        shape_weight=1.0,
        candidate_policy='expanding_history',
        future_horizon_hours_candidates=(1.0, 2.0),
        relevance_threshold_candidates=(0.1, 0.5),
    )

    results = run_validation_experiment_grid(
        incidents,
        documents,
        events,
        settings,
    )

    assert len(results) == 8
    assert results['selection_split'].unique().tolist() == ['validation']
    assert set(results['feature_version']) == {
        'alarm_tfidf_v1',
        'alarm_plus_shape_v1',
    }
    assert set(results['future_horizon_hours']) == {1.0, 2.0}
    assert set(results['relevance_threshold']) == {0.1, 0.5}
    assert results['query_count'].eq(1).all()
    assert results['top_k'].eq(5).all()
    assert results['query_limit'].isna().all()


def _build_selected_test_inputs():
    incidents = pd.DataFrame(
        {
            'incident_id': ['train', 'validation', 'test'],
            'machine_id': ['4', '4', '4'],
            'start_time': pd.to_datetime(
                [
                    '2020-01-01 07:00:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 14:00:00',
                ]
            ),
            'end_time': pd.to_datetime(
                [
                    '2020-01-01 07:30:00',
                    '2020-01-01 10:30:00',
                    '2020-01-01 14:30:00',
                ]
            ),
            'split': ['train', 'validation', 'test'],
            'event_count': [1, 2, 2],
            'duration_seconds': [0, 30, 20],
            'distinct_alarm_count': [1, 2, 2],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'alarm_document': ['98', '98 11', '98 11'],
        }
    )
    events = pd.DataFrame(
        {
            'machine_id': ['4', '4', '4', '4'],
            'timestamp': pd.to_datetime(
                [
                    '2020-01-01 08:00:00',
                    '2020-01-01 11:00:00',
                    '2020-01-01 15:00:00',
                    '2020-01-01 17:00:00',
                ]
            ),
            'alarm_code': ['98', '11', '11', '26'],
        }
    )
    return incidents, documents, events


def test_run_selected_test_evaluation_uses_only_locked_test_setting():
    incidents, documents, events = _build_selected_test_inputs()
    settings = RetrievalExperimentSettings(
        top_k=5,
        alarm_weight=1.0,
        shape_weight=1.0,
        candidate_policy='expanding_history',
        future_horizon_hours_candidates=(1.0, 2.0),
        relevance_threshold_candidates=(0.1, 0.5),
        selected_feature_version='alarm_plus_shape_v1',
        future_horizon_hours=2.0,
        relevance_threshold=0.5,
    )

    results = run_selected_test_evaluation(
        incidents,
        documents,
        events,
        settings,
    )

    assert len(results) == 1
    assert results.loc[0, 'evaluation_split'] == 'test'
    assert results.loc[0, 'feature_version'] == 'alarm_plus_shape_v1'
    assert results.loc[0, 'future_horizon_hours'] == 2.0
    assert results.loc[0, 'relevance_threshold'] == 0.5
    assert results.loc[0, 'query_count'] == 1


def test_run_selected_test_evaluation_requires_locked_setting():
    incidents, documents, events = _build_selected_test_inputs()
    settings = RetrievalExperimentSettings(
        top_k=5,
        alarm_weight=1.0,
        shape_weight=1.0,
        candidate_policy='expanding_history',
        future_horizon_hours_candidates=(1.0, 2.0),
        relevance_threshold_candidates=(0.1, 0.5),
    )

    with pytest.raises(ValueError, match='selected'):
        run_selected_test_evaluation(
            incidents,
            documents,
            events,
            settings,
        )
