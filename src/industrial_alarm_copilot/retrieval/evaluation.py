'''Time-safe offline retrieval evaluation.'''

from dataclasses import dataclass
from typing import Literal

import numpy as np
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
from industrial_alarm_copilot.retrieval.outcomes import (
    OutcomeAlarmMatrix,
    build_outcome_alarm_matrix,
    compute_outcome_jaccard_scores,
)
from industrial_alarm_copilot.retrieval.relevance import RELEVANCE_COLUMNS
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


@dataclass(frozen=True)
class QueryRetrievalEvidence:
    '''Threshold-independent candidates, ranking, and outcome overlap.'''

    query_incident_id: str
    status: QueryEvaluationStatus
    k: int
    candidate_policy: CandidatePolicy
    feature_version: str
    evaluable_candidate_count: int
    candidate_outcome_jaccard: np.ndarray
    ranked_results: pd.DataFrame


@dataclass(frozen=True)
class RetrievalBatchEvaluation:
    '''Per-query evidence and split-level evaluation summaries.'''

    query_summaries: pd.DataFrame
    ranked_results: pd.DataFrame
    split_metrics: pd.DataFrame


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


def prepare_retrieval_query_evidence(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    features: AlarmFeatureMatrix,
    outcomes: pd.DataFrame,
    query_incident_id: str,
    top_k: int = 5,
    policy: CandidatePolicy = 'expanding_history',
    outcome_alarm_matrix: OutcomeAlarmMatrix | None = None,
) -> QueryRetrievalEvidence:
    '''Prepare one query once so multiple thresholds can reuse evidence.'''
    if top_k <= 0:
        raise ValueError('top_k must be greater than zero')
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
        return QueryRetrievalEvidence(
            **base_result,
            status='query_outcome_incomplete',
            evaluable_candidate_count=0,
            candidate_outcome_jaccard=np.asarray([], dtype=float),
            ranked_results=_empty_ranked_results(),
        )
    if not bool(query_outcome['has_future_alarms']):
        return QueryRetrievalEvidence(
            **base_result,
            status='query_without_future_alarms',
            evaluable_candidate_count=0,
            candidate_outcome_jaccard=np.asarray([], dtype=float),
            ranked_results=_empty_ranked_results(),
        )

    candidates = select_outcome_evaluable_candidates(
        incidents,
        outcomes,
        query_incident_id,
        policy=policy,
    )
    if candidates.empty:
        return QueryRetrievalEvidence(
            **base_result,
            status='no_evaluable_candidates',
            evaluable_candidate_count=0,
            candidate_outcome_jaccard=np.asarray([], dtype=float),
            ranked_results=_empty_ranked_results(),
        )

    if outcome_alarm_matrix is None:
        outcome_alarm_matrix = build_outcome_alarm_matrix(outcomes)
    candidate_incident_ids = tuple(
        candidates['incident_id'].astype(str)
    )
    candidate_outcome_jaccard = compute_outcome_jaccard_scores(
        outcome_alarm_matrix,
        query_incident_id,
        candidate_incident_ids,
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
    score_by_candidate_id = dict(
        zip(candidate_incident_ids, candidate_outcome_jaccard, strict=True)
    )
    ranked_results['outcome_jaccard'] = [
        score_by_candidate_id[str(candidate_id)]
        for candidate_id in ranked_results['candidate_incident_id']
    ]

    return QueryRetrievalEvidence(
        **base_result,
        status='evaluated',
        evaluable_candidate_count=len(candidates),
        candidate_outcome_jaccard=candidate_outcome_jaccard,
        ranked_results=ranked_results,
    )


def score_retrieval_query_evidence(
    evidence: QueryRetrievalEvidence,
    relevance_threshold: float,
) -> QueryRetrievalEvaluation:
    '''Apply one relevance threshold without repeating retrieval.'''
    if not 0 < relevance_threshold <= 1:
        raise ValueError('relevance_threshold must be in (0, 1]')

    base_result = {
        'query_incident_id': evidence.query_incident_id,
        'status': evidence.status,
        'k': evidence.k,
        'candidate_policy': evidence.candidate_policy,
        'feature_version': evidence.feature_version,
        'evaluable_candidate_count': evidence.evaluable_candidate_count,
    }
    if evidence.status != 'evaluated':
        return QueryRetrievalEvaluation(
            **base_result,
            total_relevant_candidate_count=0,
            metrics=None,
            ranked_results=evidence.ranked_results.copy(),
        )

    total_relevant_candidate_count = int(
        (evidence.candidate_outcome_jaccard >= relevance_threshold).sum()
    )
    ranked_results = evidence.ranked_results.copy()
    ranked_results['query_outcome_is_complete'] = True
    ranked_results['query_has_future_alarms'] = True
    ranked_results['candidate_outcome_is_complete'] = True
    ranked_results['candidate_outcome_available_before_query'] = True
    ranked_results['evaluation_eligible'] = True
    ranked_results['relevance_threshold'] = float(relevance_threshold)
    ranked_results['is_relevant'] = pd.array(
        ranked_results['outcome_jaccard'].ge(relevance_threshold),
        dtype='boolean',
    )
    ranked_results = ranked_results[
        RETRIEVAL_RESULT_COLUMNS + RELEVANCE_COLUMNS
    ]
    metrics = compute_binary_ranking_metrics(
        ranked_results['is_relevant'],
        k=evidence.k,
        total_relevant_candidate_count=total_relevant_candidate_count,
    )

    return QueryRetrievalEvaluation(
        **base_result,
        total_relevant_candidate_count=total_relevant_candidate_count,
        metrics=metrics,
        ranked_results=ranked_results,
    )


def evaluate_retrieval_query(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    features: AlarmFeatureMatrix,
    outcomes: pd.DataFrame,
    query_incident_id: str,
    relevance_threshold: float,
    top_k: int = 5,
    policy: CandidatePolicy = 'expanding_history',
    outcome_alarm_matrix: OutcomeAlarmMatrix | None = None,
) -> QueryRetrievalEvaluation:
    '''Evaluate one query against its complete historical outcome pool.'''
    evidence = prepare_retrieval_query_evidence(
        incidents,
        documents,
        features,
        outcomes,
        query_incident_id,
        top_k=top_k,
        policy=policy,
        outcome_alarm_matrix=outcome_alarm_matrix,
    )
    return score_retrieval_query_evidence(evidence, relevance_threshold)


def build_query_evaluation_record(
    evaluation: QueryRetrievalEvaluation,
    query_split: str,
) -> dict:
    metrics = evaluation.metrics
    relevant_candidate_share = (
        float('nan')
        if evaluation.evaluable_candidate_count == 0
        else evaluation.total_relevant_candidate_count
        / evaluation.evaluable_candidate_count
    )
    expected_random_precision = (
        float('nan')
        if metrics is None or evaluation.evaluable_candidate_count == 0
        else (
            min(evaluation.k, evaluation.evaluable_candidate_count)
            / evaluation.k
            * relevant_candidate_share
        )
    )
    precision_lift = (
        float('nan')
        if metrics is None
        or expected_random_precision == 0
        else metrics.precision_at_k / expected_random_precision
    )
    return {
        'query_incident_id': evaluation.query_incident_id,
        'split': str(query_split),
        'status': evaluation.status,
        'k': evaluation.k,
        'candidate_policy': evaluation.candidate_policy,
        'feature_version': evaluation.feature_version,
        'evaluable_candidate_count': (
            evaluation.evaluable_candidate_count
        ),
        'total_relevant_candidate_count': (
            evaluation.total_relevant_candidate_count
        ),
        'relevant_candidate_share': relevant_candidate_share,
        'retrieved_count': (
            float('nan') if metrics is None else metrics.retrieved_count
        ),
        'relevant_retrieved_count': (
            float('nan')
            if metrics is None
            else metrics.relevant_retrieved_count
        ),
        'hit_at_k': (
            float('nan') if metrics is None else float(metrics.hit_at_k)
        ),
        'precision_at_k': (
            float('nan') if metrics is None else metrics.precision_at_k
        ),
        'recall_at_k': (
            float('nan') if metrics is None else metrics.recall_at_k
        ),
        'maximum_recall_at_k': (
            float('nan')
            if metrics is None
            else metrics.maximum_recall_at_k
        ),
        'recall_efficiency_at_k': (
            float('nan')
            if metrics is None
            else metrics.recall_efficiency_at_k
        ),
        'expected_random_precision_at_k': expected_random_precision,
        'precision_lift_at_k': precision_lift,
        'reciprocal_rank': (
            float('nan') if metrics is None else metrics.reciprocal_rank
        ),
        'ndcg_at_k': (
            float('nan') if metrics is None else metrics.ndcg_at_k
        ),
    }


def build_split_metrics(query_summaries: pd.DataFrame) -> pd.DataFrame:
    records = []
    for split, split_queries in query_summaries.groupby(
        'split',
        observed=True,
        sort=False,
    ):
        query_count = len(split_queries)
        evaluated_mask = split_queries['status'].eq('evaluated')
        relevant_pool_mask = (
            evaluated_mask
            & split_queries['total_relevant_candidate_count'].gt(0)
        )
        status_counts = split_queries['status'].value_counts()
        records.append(
            {
                'split': str(split),
                'query_count': query_count,
                'evaluated_query_count': int(evaluated_mask.sum()),
                'evaluation_coverage': float(evaluated_mask.mean()),
                'query_outcome_incomplete_count': int(
                    status_counts.get('query_outcome_incomplete', 0)
                ),
                'query_without_future_alarms_count': int(
                    status_counts.get('query_without_future_alarms', 0)
                ),
                'no_evaluable_candidates_count': int(
                    status_counts.get('no_evaluable_candidates', 0)
                ),
                'query_with_relevant_candidates_count': int(
                    relevant_pool_mask.sum()
                ),
                'relevant_candidate_query_share': float(
                    relevant_pool_mask.mean()
                ),
                'mean_hit_at_k': split_queries['hit_at_k'].mean(),
                'mean_precision_at_k': split_queries[
                    'precision_at_k'
                ].mean(),
                'mean_recall_at_k': split_queries['recall_at_k'].mean(),
                'mean_relevant_candidate_share': split_queries[
                    'relevant_candidate_share'
                ].mean(),
                'mean_maximum_recall_at_k': split_queries[
                    'maximum_recall_at_k'
                ].mean(),
                'mean_recall_efficiency_at_k': split_queries[
                    'recall_efficiency_at_k'
                ].mean(),
                'mean_expected_random_precision_at_k': split_queries[
                    'expected_random_precision_at_k'
                ].mean(),
                'mean_precision_lift_at_k': split_queries[
                    'precision_lift_at_k'
                ].mean(),
                'mean_reciprocal_rank': split_queries[
                    'reciprocal_rank'
                ].mean(),
                'mean_ndcg_at_k': split_queries['ndcg_at_k'].mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_retrieval_splits(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    features: AlarmFeatureMatrix,
    outcomes: pd.DataFrame,
    relevance_threshold: float,
    query_splits: tuple[str, ...] = ('validation', 'test'),
    top_k: int = 5,
    policy: CandidatePolicy = 'expanding_history',
) -> RetrievalBatchEvaluation:
    '''Evaluate every query in requested splits without dropping coverage.'''
    query_rows = incidents.loc[
        incidents['split'].isin(query_splits)
    ].sort_values(
        ['start_time', 'incident_id'],
        kind='stable',
    )
    if query_rows.empty:
        raise ValueError('query_splits must select at least one incident')

    outcome_alarm_matrix = build_outcome_alarm_matrix(outcomes)
    summary_records = []
    ranked_frames = []
    for query in query_rows.itertuples(index=False):
        evaluation = evaluate_retrieval_query(
            incidents,
            documents,
            features,
            outcomes,
            query_incident_id=str(query.incident_id),
            relevance_threshold=relevance_threshold,
            top_k=top_k,
            policy=policy,
            outcome_alarm_matrix=outcome_alarm_matrix,
        )
        summary_records.append(
            build_query_evaluation_record(evaluation, str(query.split))
        )
        if not evaluation.ranked_results.empty:
            ranked_frame = evaluation.ranked_results.copy()
            ranked_frame.insert(1, 'query_split', str(query.split))
            ranked_frames.append(ranked_frame)

    query_summaries = pd.DataFrame.from_records(summary_records)
    ranked_results = (
        pd.concat(ranked_frames, ignore_index=True)
        if ranked_frames
        else _empty_ranked_results()
    )
    return RetrievalBatchEvaluation(
        query_summaries=query_summaries,
        ranked_results=ranked_results,
        split_metrics=build_split_metrics(query_summaries),
    )
