'''Deterministic forecasting baseline tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.forecasting.baselines import (
    fit_global_frequency_baseline,
    fit_machine_frequency_baseline,
    fit_transition_frequency_baseline,
    rank_global_frequency_baseline,
    score_global_frequency_baseline,
    score_machine_frequency_baseline,
    score_transition_frequency_baseline,
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


def test_machine_frequency_baseline_conditions_and_falls_back():
    labels = pd.DataFrame(
        {
            'incident_id': [
                'm4_train_a',
                'm4_train_b',
                'm9_train_a',
                'm9_train_b',
                'm19_train',
                'm4_validation',
                'm9_validation',
                'm19_validation',
                'unseen_validation',
            ],
            'machine_id': ['4', '4', '9', '9', '19', '4', '9', '19', '20'],
            'split': [
                'train',
                'train',
                'train',
                'train',
                'train',
                'validation',
                'validation',
                'validation',
                'validation',
            ],
            'outcome_is_complete': [True] * 9,
            'future_alarm_codes': [
                ('11',),
                ('11',),
                ('26', '98'),
                ('26', '98'),
                ('98',),
                ('98',),
                ('11',),
                ('26',),
                ('26',),
            ],
            'future_alarm_counts': [
                tuple((code, 1) for code in codes)
                for codes in [
                    ('11',),
                    ('11',),
                    ('26', '98'),
                    ('26', '98'),
                    ('98',),
                    ('98',),
                    ('11',),
                    ('26',),
                    ('26',),
                ]
            ],
        }
    )
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)

    baseline = fit_machine_frequency_baseline(
        labels,
        encoded,
        minimum_machine_train_samples=2,
    )
    predictions = score_machine_frequency_baseline(
        baseline,
        incident_ids=encoded.incident_ids,
        machine_ids=tuple(labels['machine_id']),
    )

    assert dict(baseline.machine_train_sample_counts) == {
        '19': 1,
        '4': 2,
        '9': 2,
    }
    assert set(dict(baseline.machine_scores)) == {'4', '9'}
    assert predictions.baseline_scopes[-4:] == (
        'machine',
        'machine',
        'global_fallback',
        'global_fallback',
    )
    assert predictions.machine_train_sample_counts[-4:].tolist() == [
        2,
        2,
        1,
        0,
    ]
    validation_scores = predictions.score_matrix.scores[-4:]
    assert validation_scores[0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert validation_scores[1].tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert validation_scores[2].tolist() == pytest.approx(
        baseline.global_scores
    )
    assert validation_scores[3].tolist() == pytest.approx(
        baseline.global_scores
    )


def test_transition_frequency_baseline_uses_hierarchical_fallback():
    labels = pd.DataFrame(
        {
            'incident_id': [
                'm4_a1',
                'm4_a2',
                'm4_b1',
                'm9_a1',
                'm9_a2',
                'm19_x1',
                'query_transition',
                'query_machine',
                'query_global',
            ],
            'machine_id': ['4', '4', '4', '9', '9', '19', '4', '4', '19'],
            'split': ['train'] * 6 + ['validation'] * 3,
            'outcome_is_complete': [True] * 9,
            'future_alarm_codes': [
                ('11',),
                ('11',),
                ('98',),
                ('26',),
                ('26',),
                ('98',),
                ('98',),
                ('26',),
                ('11',),
            ],
            'future_alarm_counts': [
                ((code, 1),)
                for code in ['11', '11', '98', '26', '26', '98', '98', '26', '11']
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

    baseline = fit_transition_frequency_baseline(
        labels,
        encoded,
        context,
        minimum_machine_train_samples=2,
        minimum_transition_train_samples=2,
    )
    predictions = score_transition_frequency_baseline(
        baseline,
        incident_ids=encoded.incident_ids,
        machine_ids=tuple(labels['machine_id']),
        last_alarm_codes=tuple(context['last_alarm_code']),
    )

    assert predictions.baseline_scopes[-3:] == (
        'transition',
        'machine_fallback',
        'global_fallback',
    )
    assert predictions.transition_train_sample_counts[-3:].tolist() == [
        2,
        1,
        1,
    ]
    validation_scores = predictions.score_matrix.scores[-3:]
    assert validation_scores[0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    machine4_scores = dict(
        baseline.machine_baseline.machine_scores
    )['4']
    assert validation_scores[1].tolist() == pytest.approx(machine4_scores)
    assert validation_scores[2].tolist() == pytest.approx(
        baseline.machine_baseline.global_scores
    )
