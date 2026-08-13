'''Deterministic forecasting baseline tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.forecasting.baselines import (
    fit_global_frequency_baseline,
    rank_global_frequency_baseline,
    score_global_frequency_baseline,
)
from industrial_alarm_copilot.forecasting.evaluation import (
    evaluate_forecast_scores,
)
from industrial_alarm_copilot.forecasting.vocabulary import (
    encode_forecast_labels,
    fit_forecast_label_vocabulary,
)


def _build_labels():
    return pd.DataFrame(
        {
            'incident_id': [
                'train_a',
                'train_b',
                'train_incomplete',
                'validation',
            ],
            'split': ['train', 'train', 'train', 'validation'],
            'outcome_is_complete': [True, True, False, True],
            'future_alarm_codes': [
                ('98', '11'),
                ('98', '26'),
                ('11',),
                ('11', '26'),
            ],
            'future_alarm_counts': [
                (('98', 1), ('11', 1)),
                (('98', 1), ('26', 1)),
                (('11', 1),),
                (('11', 1), ('26', 1)),
            ],
        }
    )


def test_global_frequency_baseline_uses_complete_train_only():
    labels = _build_labels()
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)

    baseline = fit_global_frequency_baseline(labels, encoded)

    assert baseline.alarm_codes == ('11', '26', '98')
    assert baseline.scores.tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert baseline.train_sample_count == 2
    assert baseline.model_version == 'global_frequency_v1'


def test_global_frequency_ranking_is_deterministic_for_ties():
    labels = _build_labels()
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)
    baseline = fit_global_frequency_baseline(labels, encoded)

    ranking = rank_global_frequency_baseline(baseline, top_k=2)

    assert ranking['rank'].tolist() == [1, 2]
    assert ranking['alarm_code'].tolist() == ['98', '11']
    assert ranking['model_score'].tolist() == pytest.approx([1.0, 0.5])
    assert ranking['model_version'].unique().tolist() == [
        'global_frequency_v1'
    ]


def test_global_frequency_validation_metrics_preserve_empty_coverage():
    labels = _build_labels()
    empty_validation = pd.DataFrame(
        {
            'incident_id': ['validation_empty', 'validation_incomplete'],
            'split': ['validation', 'validation'],
            'outcome_is_complete': [True, False],
            'future_alarm_codes': [(), ('98',)],
            'future_alarm_counts': [(), (('98', 1),)],
        }
    )
    labels = pd.concat([labels.iloc[:3], empty_validation], ignore_index=True)
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)
    baseline = fit_global_frequency_baseline(labels, encoded)
    scores = score_global_frequency_baseline(
        baseline,
        incident_ids=encoded.incident_ids,
    )

    evaluation = evaluate_forecast_scores(
        labels,
        encoded,
        scores,
        evaluation_split='validation',
        top_k=2,
    )

    metrics = evaluation.split_metrics.iloc[0]
    assert metrics['episode_count'] == 2
    assert metrics['complete_outcome_count'] == 1
    assert metrics['outcome_coverage'] == pytest.approx(0.5)
    assert metrics['complete_empty_known_label_count'] == 1
    assert metrics['mean_hit_at_k'] == pytest.approx(0.0)
    assert metrics['mean_precision_at_k'] == pytest.approx(0.0)
    assert pd.isna(metrics['mean_recall_at_k'])
    assert pd.isna(metrics['macro_f1_at_k'])
    assert metrics['evaluated_label_count'] == 0
