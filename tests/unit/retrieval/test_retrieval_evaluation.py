'''Offline retrieval evaluation candidate tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.evaluation import (
    build_retrieval_evaluation_index,
    evaluate_retrieval_query,
    evaluate_retrieval_splits,
    prepare_retrieval_query_evidence,
    score_retrieval_query_evidence,
    select_outcome_evaluable_candidates,
)
from industrial_alarm_copilot.retrieval.features import (
    fit_alarm_tfidf,
    transform_alarm_documents,
)


def test_select_outcome_evaluable_candidates_applies_second_time_gate():
    incidents = pd.DataFrame(
        {
            'incident_id': [
                'old_complete',
                'outcome_equal_boundary',
                'outcome_incomplete',
                'query',
                'future_episode',
            ],
            'start_time': pd.to_datetime(
                [
                    '2020-01-01 07:00:00',
                    '2020-01-01 08:00:00',
                    '2020-01-01 08:30:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 11:00:00',
                ]
            ),
            'end_time': pd.to_datetime(
                [
                    '2020-01-01 07:30:00',
                    '2020-01-01 08:30:00',
                    '2020-01-01 09:00:00',
                    '2020-01-01 10:30:00',
                    '2020-01-01 11:30:00',
                ]
            ),
            'split': ['train', 'train', 'train', 'test', 'train'],
        }
    )
    outcomes = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'outcome_end_time': pd.to_datetime(
                [
                    '2020-01-01 09:00:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 09:30:00',
                    '2020-01-01 12:00:00',
                    '2020-01-01 13:00:00',
                ]
            ),
            'outcome_is_complete': [True, True, False, True, True],
        }
    )

    candidates = select_outcome_evaluable_candidates(
        incidents,
        outcomes,
        query_incident_id='query',
    )

    assert candidates['incident_id'].tolist() == ['old_complete']

    evaluation_index = build_retrieval_evaluation_index(
        incidents,
        outcomes,
    )
    candidate_rows = evaluation_index.candidate_row_numbers(
        'query',
        'expanding_history',
    )
    assert incidents.iloc[candidate_rows]['incident_id'].tolist() == [
        'old_complete'
    ]


def _build_query_evaluation_inputs():
    incidents = pd.DataFrame(
        {
            'incident_id': [
                'candidate_relevant',
                'candidate_irrelevant',
                'candidate_outcome_too_recent',
                'query',
                'query_incomplete',
            ],
            'machine_id': ['4', '6', '7', '4', '9'],
            'start_time': pd.to_datetime(
                [
                    '2020-01-01 07:00:00',
                    '2020-01-01 08:00:00',
                    '2020-01-01 09:00:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 12:00:00',
                ]
            ),
            'end_time': pd.to_datetime(
                [
                    '2020-01-01 07:30:00',
                    '2020-01-01 08:30:00',
                    '2020-01-01 09:30:00',
                    '2020-01-01 10:30:00',
                    '2020-01-01 12:30:00',
                ]
            ),
            'split': ['train', 'train', 'train', 'test', 'test'],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'alarm_document': [
                '26',
                '98 11',
                '98 11',
                '98 11',
                '98',
            ],
        }
    )
    fitted_tfidf = fit_alarm_tfidf(documents, incidents)
    features = transform_alarm_documents(fitted_tfidf, documents)
    outcomes = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'outcome_end_time': pd.to_datetime(
                [
                    '2020-01-01 08:30:00',
                    '2020-01-01 09:30:00',
                    '2020-01-01 10:00:00',
                    '2020-01-01 11:30:00',
                    '2020-01-01 13:30:00',
                ]
            ),
            'outcome_is_complete': [True, True, True, True, False],
            'has_future_alarms': [True, True, True, True, False],
            'future_alarm_codes': [
                ('11', '98'),
                ('26',),
                ('11', '98'),
                ('11', '98'),
                (),
            ],
        }
    )
    return incidents, documents, features, outcomes


def test_evaluate_retrieval_query_uses_full_pool_and_ranked_top_five():
    incidents, documents, features, outcomes = (
        _build_query_evaluation_inputs()
    )

    evaluation = evaluate_retrieval_query(
        incidents,
        documents,
        features,
        outcomes,
        query_incident_id='query',
        relevance_threshold=0.5,
        top_k=5,
    )

    assert evaluation.status == 'evaluated'
    assert evaluation.evaluable_candidate_count == 2
    assert evaluation.total_relevant_candidate_count == 1
    assert evaluation.ranked_results[
        'candidate_incident_id'
    ].tolist() == ['candidate_irrelevant', 'candidate_relevant']
    assert evaluation.ranked_results['is_relevant'].tolist() == [
        False,
        True,
    ]
    assert evaluation.metrics is not None
    assert evaluation.metrics.hit_at_k is True
    assert evaluation.metrics.precision_at_k == pytest.approx(1 / 5)
    assert evaluation.metrics.recall_at_k == pytest.approx(1.0)
    assert evaluation.metrics.maximum_recall_at_k == pytest.approx(1.0)
    assert evaluation.metrics.recall_efficiency_at_k == pytest.approx(1.0)
    assert evaluation.metrics.reciprocal_rank == pytest.approx(1 / 2)


def test_evaluate_retrieval_query_preserves_incomplete_query_coverage():
    incidents, documents, features, outcomes = (
        _build_query_evaluation_inputs()
    )
    outcomes.loc[
        outcomes['incident_id'].eq('query'),
        'outcome_is_complete',
    ] = False

    evaluation = evaluate_retrieval_query(
        incidents,
        documents,
        features,
        outcomes,
        query_incident_id='query',
        relevance_threshold=0.5,
        top_k=5,
    )

    assert evaluation.status == 'query_outcome_incomplete'
    assert evaluation.metrics is None
    assert evaluation.ranked_results.empty


def test_evaluate_retrieval_splits_reports_scores_and_coverage():
    incidents, documents, features, outcomes = (
        _build_query_evaluation_inputs()
    )

    batch = evaluate_retrieval_splits(
        incidents,
        documents,
        features,
        outcomes,
        relevance_threshold=0.5,
        query_splits=('test',),
        top_k=5,
    )

    assert batch.query_summaries['status'].tolist() == [
        'evaluated',
        'query_outcome_incomplete',
    ]
    assert batch.ranked_results['query_incident_id'].unique().tolist() == [
        'query'
    ]
    split_metrics = batch.split_metrics.iloc[0]
    assert split_metrics['query_count'] == 2
    assert split_metrics['evaluated_query_count'] == 1
    assert split_metrics['evaluation_coverage'] == pytest.approx(0.5)
    assert split_metrics['query_outcome_incomplete_count'] == 1
    assert split_metrics['query_with_relevant_candidates_count'] == 1
    assert split_metrics['mean_hit_at_k'] == pytest.approx(1.0)
    assert split_metrics['mean_precision_at_k'] == pytest.approx(1 / 5)
    assert split_metrics['mean_relevant_candidate_share'] == (
        pytest.approx(1 / 2)
    )
    assert split_metrics['mean_expected_random_precision_at_k'] == (
        pytest.approx(1 / 5)
    )
    assert split_metrics['mean_precision_lift_at_k'] == pytest.approx(1.0)


def test_query_evidence_can_be_rescored_without_retrieval():
    incidents, documents, features, outcomes = (
        _build_query_evaluation_inputs()
    )
    outcomes.loc[
        outcomes['incident_id'].eq('candidate_relevant'),
        'future_alarm_codes',
    ] = pd.Series(
        [('98', '26')],
        index=outcomes.index[
            outcomes['incident_id'].eq('candidate_relevant')
        ],
        dtype='object',
    )

    evidence = prepare_retrieval_query_evidence(
        incidents,
        documents,
        features,
        outcomes,
        query_incident_id='query',
        top_k=5,
    )
    loose = score_retrieval_query_evidence(evidence, 0.3)
    strict = score_retrieval_query_evidence(evidence, 0.5)

    assert loose.ranked_results['candidate_incident_id'].tolist() == (
        strict.ranked_results['candidate_incident_id'].tolist()
    )
    assert loose.total_relevant_candidate_count == 1
    assert strict.total_relevant_candidate_count == 0
    assert loose.metrics is not None
    assert strict.metrics is not None
    assert loose.metrics.hit_at_k is True
    assert strict.metrics.hit_at_k is False
