'''Time-safe offline retrieval evaluation.'''

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from industrial_alarm_copilot.retrieval.candidates import (
    CandidatePolicy,
    select_historical_candidates,
)
from industrial_alarm_copilot.retrieval.features import AlarmFeatureMatrix
from industrial_alarm_copilot.retrieval.metrics import (
    BinaryRankingMetrics,
    compute_binary_ranking_metrics,
)
from industrial_alarm_copilot.retrieval.relevance import (
    label_retrieval_relevance,
)
from industrial_alarm_copilot.retrieval.search import (
    RETRIEVAL_RESULT_COLUMNS,
    retrieve_similar_episodes,
)


QueryEvaluationStatus = Literal[
    'evaluated',
    'query_outcome_incomplete',
    'query_without_future_alarms',
    'no_evaluable_candidates',
]


@dataclass(frozen=True)
class QueryRetrievalEvaluation:
    '''Evaluation result and coverage state for one query episode.'''

    query_incident_id: str
    status: QueryEvaluationStatus
    k: int
    candidate_policy: CandidatePolicy
    feature_version: str
    evaluable_candidate_count: int
    total_relevant_candidate_count: int
    metrics: BinaryRankingMetrics | None
    ranked_results: pd.DataFrame


def select_outcome_evaluable_candidates(
    incidents: pd.DataFrame,
    outcomes: pd.DataFrame,
    query_incident_id: str,
    policy: CandidatePolicy = 'expanding_history',
) -> pd.DataFrame:
    '''Return historical candidates whose complete outcomes predate query.'''
    if not outcomes['incident_id'].is_unique:
        raise ValueError('outcomes incident_id must be unique')

    candidates = select_historical_candidates(
        incidents,
        query_incident_id,
        policy=policy,
    )
    query_rows = incidents.loc[
        incidents['incident_id'].eq(query_incident_id)
    ]
    query_start_time = query_rows.iloc[0]['start_time']

    outcome_columns = outcomes[
        ['incident_id', 'outcome_end_time', 'outcome_is_complete']
    ].copy()
    candidates_with_outcomes = candidates.merge(
        outcome_columns,
        on='incident_id',
        how='left',
        sort=False,
        validate='one_to_one',
    )
    if candidates_with_outcomes['outcome_is_complete'].isna().any():
        raise ValueError('every candidate must have an outcome')

    evaluation_mask = (
        candidates_with_outcomes['outcome_is_complete'].astype(bool)
        & candidates_with_outcomes['outcome_end_time'].lt(query_start_time)
    )
    return candidates_with_outcomes.loc[
        evaluation_mask,
        candidates.columns,
    ].copy()


def _empty_ranked_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RETRIEVAL_RESULT_COLUMNS)


def evaluate_retrieval_query(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    features: AlarmFeatureMatrix,
    outcomes: pd.DataFrame,
    query_incident_id: str,
    relevance_threshold: float,
    top_k: int = 5,
    policy: CandidatePolicy = 'expanding_history',
) -> QueryRetrievalEvaluation:
    '''Evaluate one query against its complete historical outcome pool.'''
    if not outcomes['incident_id'].is_unique:
        raise ValueError('outcomes incident_id must be unique')

    query_outcomes = outcomes.loc[
        outcomes['incident_id'].eq(query_incident_id)
    ]
    if query_outcomes.empty:
        raise ValueError('query must have an outcome')
    query_outcome = query_outcomes.iloc[0]

    base_result = {
        'query_incident_id': str(query_incident_id),
        'k': top_k,
        'candidate_policy': policy,
        'feature_version': features.feature_version,
    }
    if not bool(query_outcome['outcome_is_complete']):
        return QueryRetrievalEvaluation(
            **base_result,
            status='query_outcome_incomplete',
            evaluable_candidate_count=0,
            total_relevant_candidate_count=0,
            metrics=None,
            ranked_results=_empty_ranked_results(),
        )
    if not bool(query_outcome['has_future_alarms']):
        return QueryRetrievalEvaluation(
            **base_result,
            status='query_without_future_alarms',
            evaluable_candidate_count=0,
            total_relevant_candidate_count=0,
            metrics=None,
            ranked_results=_empty_ranked_results(),
        )

    candidates = select_outcome_evaluable_candidates(
        incidents,
        outcomes,
        query_incident_id,
        policy=policy,
    )
    if candidates.empty:
        return QueryRetrievalEvaluation(
            **base_result,
            status='no_evaluable_candidates',
            evaluable_candidate_count=0,
            total_relevant_candidate_count=0,
            metrics=None,
            ranked_results=_empty_ranked_results(),
        )

    candidate_pairs = pd.DataFrame(
        {
            'query_incident_id': str(query_incident_id),
            'candidate_incident_id': candidates[
                'incident_id'
            ].astype(str).to_numpy(),
        }
    )
    labeled_candidate_pool = label_retrieval_relevance(
        candidate_pairs,
        outcomes,
        incidents,
        relevance_threshold,
    )
    total_relevant_candidate_count = int(
        labeled_candidate_pool['is_relevant'].sum()
    )

    query_row = incidents.loc[
        incidents['incident_id'].eq(query_incident_id)
    ]
    evaluation_incidents = pd.concat(
        [query_row, candidates],
        ignore_index=True,
    )
    ranked_results = retrieve_similar_episodes(
        evaluation_incidents,
        documents,
        features,
        query_incident_id,
        top_k=top_k,
        policy=policy,
    )
    ranked_results = label_retrieval_relevance(
        ranked_results,
        outcomes,
        incidents,
        relevance_threshold,
    )
    metrics = compute_binary_ranking_metrics(
        ranked_results['is_relevant'],
        k=top_k,
        total_relevant_candidate_count=total_relevant_candidate_count,
    )

    return QueryRetrievalEvaluation(
        **base_result,
        status='evaluated',
        evaluable_candidate_count=len(candidates),
        total_relevant_candidate_count=total_relevant_candidate_count,
        metrics=metrics,
        ranked_results=ranked_results,
    )
