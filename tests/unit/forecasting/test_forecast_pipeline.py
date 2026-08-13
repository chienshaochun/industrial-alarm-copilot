'''Locked forecasting pipeline tests.'''

import pandas as pd

from industrial_alarm_copilot.forecasting.pipeline import (
    ForecastPreparedInputs,
    run_selected_forecast_evaluation,
)
from industrial_alarm_copilot.forecasting.profile import (
    assign_forecast_support_groups,
    build_forecast_alarm_support_profile,
)
from industrial_alarm_copilot.forecasting.settings import ForecastExperimentSettings
from industrial_alarm_copilot.forecasting.vocabulary import (
    encode_forecast_labels,
    fit_forecast_label_vocabulary,
)


def _settings():
    return ForecastExperimentSettings(
        top_k=2,
        forecast_horizon_hours_candidates=(6.0,),
        selected_forecast_horizon_hours=6.0,
        rare_max_train_support=1,
        common_min_train_support=3,
        minimum_machine_train_samples=2,
        minimum_transition_train_samples=2,
        linear_c=1.0,
        linear_max_iter=10,
        linear_class_weight_candidates=('none',),
        sequence_max_length=4,
        sequence_embedding_dim=2,
        sequence_hidden_dim=2,
        sequence_machine_embedding_dim=2,
        sequence_batch_size=2,
        sequence_epochs=1,
        sequence_learning_rate=0.001,
        sequence_weight_candidates=('none',),
        sequence_positive_weight_cap=2.0,
        random_seed=0,
        selection_macro_f1_tolerance=0.01,
        selected_model_version='transition_frequency_v1',
    )


def test_locked_pipeline_evaluates_test_with_hierarchical_scopes():
    labels = pd.DataFrame(
        {
            'incident_id': ['a1', 'a2', 'b1', 'c1', 'c2', 'x1', 'q1', 'q2', 'q3'],
            'machine_id': ['4', '4', '4', '9', '9', '19', '4', '4', '19'],
            'split': ['train'] * 6 + ['test'] * 3,
            'outcome_is_complete': [True] * 9,
            'future_alarm_codes': [
                ('11',), ('11',), ('98',), ('26',), ('26',), ('98',),
                ('11',), ('98',), ('26',),
            ],
            'future_alarm_counts': [
                ((code, 1),)
                for code in ['11', '11', '98', '26', '26', '98', '11', '98', '26']
            ],
        }
    )
    context = pd.DataFrame(
        {
            'incident_id': labels['incident_id'],
            'last_alarm_code': ['A', 'A', 'B', 'A', 'A', 'X', 'A', 'B', 'X'],
        }
    )
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)
    support = assign_forecast_support_groups(
        build_forecast_alarm_support_profile(vocabulary), 1, 3
    )
    inputs = ForecastPreparedInputs(
        events=pd.DataFrame(),
        incidents=pd.DataFrame(),
        labels=labels,
        encoded=encoded,
        incident_context=context,
        grouped_support=support,
    )

    run = run_selected_forecast_evaluation(inputs, _settings(), 'test')

    assert run.evaluation.split_metrics.iloc[0]['split'] == 'test'
    assert run.evaluation.split_metrics.iloc[0]['model_version'] == (
        'transition_frequency_v1'
    )
    assert set(run.support_group_metrics['support_group']) == {
        'rare', 'medium', 'common'
    }
    assert run.scope_profile['query_count'].sum() == 3
