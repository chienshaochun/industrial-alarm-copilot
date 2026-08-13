'''Top-K forecasting metric tests.'''

import numpy as np
import pandas as pd
import pytest

from industrial_alarm_copilot.forecasting.evaluation import (
    ForecastScoreMatrix,
    build_forecast_support_group_metrics,
    evaluate_forecast_scores,
)
from industrial_alarm_copilot.forecasting.vocabulary import (
    encode_forecast_labels,
    fit_forecast_label_vocabulary,
)


def test_evaluate_forecast_scores_computes_top_k_and_coverage():
    labels = pd.DataFrame(
        {
            'incident_id': [
                'train',
                'validation_hit',
                'validation_empty',
                'validation_incomplete',
            ],
            'split': ['train', 'validation', 'validation', 'validation'],
            'outcome_is_complete': [True, True, True, False],
            'future_alarm_codes': [('11', '98'), ('11', '98'), (), ('11',)],
            'future_alarm_counts': [
                (('11', 1), ('98', 1)),
                (('11', 1), ('98', 1)),
                (),
                (('11', 1),),
            ],
        }
    )
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)
    scores = ForecastScoreMatrix(
        incident_ids=encoded.incident_ids,
        alarm_codes=encoded.alarm_codes,
        scores=np.asarray([[0.9, 0.8]] * len(labels)),
        model_version='test_model',
    )

    evaluation = evaluate_forecast_scores(
        labels,
        encoded,
        scores,
        evaluation_split='validation',
        top_k=2,
    )

    assert evaluation.query_metrics['incident_id'].tolist() == [
        'validation_hit',
        'validation_empty',
    ]
    assert evaluation.query_metrics['true_positive_count'].tolist() == [2, 0]
    assert evaluation.query_metrics['precision_at_k'].tolist() == pytest.approx(
        [1.0, 0.0]
    )
    assert evaluation.query_metrics['recall_at_k'].iloc[0] == pytest.approx(1.0)
    assert pd.isna(evaluation.query_metrics['recall_at_k'].iloc[1])

    metrics = evaluation.split_metrics.iloc[0]
    assert metrics['episode_count'] == 3
    assert metrics['complete_outcome_count'] == 2
    assert metrics['outcome_coverage'] == pytest.approx(2 / 3)
    assert metrics['mean_hit_at_k'] == pytest.approx(0.5)
    assert metrics['mean_precision_at_k'] == pytest.approx(0.5)
    assert metrics['mean_recall_at_k'] == pytest.approx(1.0)
    assert metrics['micro_precision_at_k'] == pytest.approx(0.5)
    assert metrics['micro_recall_at_k'] == pytest.approx(1.0)
    assert metrics['micro_f1_at_k'] == pytest.approx(2 / 3)
    assert metrics['macro_f1_at_k'] == pytest.approx(2 / 3)

    support_profile = pd.DataFrame(
        {
            'alarm_code': ['11', '98'],
            'train_sample_support': [20, 500],
            'support_group': ['rare', 'common'],
        }
    )
    group_metrics = build_forecast_support_group_metrics(
        evaluation.per_label_metrics,
        support_profile,
    ).set_index('support_group')

    assert group_metrics.loc['rare', 'evaluated_label_count'] == 1
    assert group_metrics.loc['rare', 'micro_recall_at_k'] == pytest.approx(1.0)
    assert group_metrics.loc['common', 'evaluated_label_count'] == 1
    assert group_metrics.loc['common', 'micro_recall_at_k'] == pytest.approx(1.0)
    assert group_metrics.loc['medium', 'label_count'] == 0
    assert pd.isna(group_metrics.loc['medium', 'macro_f1_at_k'])
