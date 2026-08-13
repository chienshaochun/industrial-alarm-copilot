'''Validation selection and locked test evaluation experiments.'''

import pandas as pd

from industrial_alarm_copilot.retrieval.evaluation import (
    build_retrieval_evaluation_index,
    build_query_evaluation_record,
    build_split_metrics,
    evaluate_retrieval_splits,
    prepare_retrieval_query_evidence,
    score_retrieval_query_evidence,
)
from industrial_alarm_copilot.retrieval.features import (
    build_retrieval_feature_variants,
)
from industrial_alarm_copilot.retrieval.outcomes import (
    build_future_alarm_outcomes,
    build_outcome_alarm_matrix,
)
from industrial_alarm_copilot.retrieval.settings import (
    RetrievalExperimentSettings,
)
from industrial_alarm_copilot.retrieval.search import (
    build_retrieval_search_index,
)


RETRIEVAL_DIAGNOSTIC_COLUMN_MAP = {
    'feature_version': 'feature_version',
    'future_horizon_hours': 'horizon_h',
    'relevance_threshold': 'threshold',
    'evaluation_coverage': 'coverage',
    'relevant_candidate_query_share': 'relevant_query_share',
    'mean_relevant_candidate_share': 'relevant_candidate_share',
    'mean_hit_at_k': 'hit_at_5',
    'mean_precision_at_k': 'precision_at_5',
    'mean_expected_random_precision_at_k': 'random_precision_at_5',
    'mean_precision_lift_at_k': 'precision_lift_at_5',
    'mean_recall_at_k': 'recall_at_5',
    'mean_maximum_recall_at_k': 'max_recall_at_5',
    'mean_recall_efficiency_at_k': 'recall_efficiency_at_5',
    'mean_reciprocal_rank': 'mrr',
    'mean_ndcg_at_k': 'ndcg_at_5',
}


def select_retrieval_diagnostics(
    experiment_results: pd.DataFrame,
) -> pd.DataFrame:
    '''Select and shorten columns needed to interpret Top-5 quality.'''
    missing_columns = set(RETRIEVAL_DIAGNOSTIC_COLUMN_MAP).difference(
        experiment_results.columns
    )
    if missing_columns:
        raise ValueError('experiment results are missing diagnostic columns')
    return experiment_results[
        list(RETRIEVAL_DIAGNOSTIC_COLUMN_MAP)
    ].rename(columns=RETRIEVAL_DIAGNOSTIC_COLUMN_MAP)


def run_validation_experiment_grid(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    events: pd.DataFrame,
    settings: RetrievalExperimentSettings,
    max_validation_queries: int | None = None,
) -> pd.DataFrame:
    '''Compare feature, horizon, and relevance settings on validation only.'''
    validation_queries = incidents.loc[
        incidents['split'].eq('validation')
    ].sort_values(
        ['start_time', 'incident_id'],
        kind='stable',
    )
    if validation_queries.empty:
        raise ValueError('at least one validation incident is required')
    if max_validation_queries is not None:
        if max_validation_queries <= 0:
            raise ValueError('max_validation_queries must be positive')
        validation_queries = validation_queries.head(max_validation_queries)

    feature_variants = build_retrieval_feature_variants(
        documents,
        incidents,
        alarm_weight=settings.alarm_weight,
        shape_weight=settings.shape_weight,
    )
    experiment_records = []
    for horizon_hours in settings.future_horizon_hours_candidates:
        outcomes = build_future_alarm_outcomes(
            incidents,
            events,
            future_horizon_hours=horizon_hours,
        )
        outcome_alarm_matrix = build_outcome_alarm_matrix(outcomes)
        evaluation_index = build_retrieval_evaluation_index(
            incidents,
            outcomes,
        )

        for feature_version, features in feature_variants.items():
            search_index = build_retrieval_search_index(
                incidents,
                documents,
                features,
            )
            evidence_by_query_id = {
                str(query.incident_id): prepare_retrieval_query_evidence(
                    incidents,
                    documents,
                    features,
                    outcomes,
                    query_incident_id=str(query.incident_id),
                    top_k=settings.top_k,
                    policy=settings.candidate_policy,
                    outcome_alarm_matrix=outcome_alarm_matrix,
                    evaluation_index=evaluation_index,
                    search_index=search_index,
                )
                for query in validation_queries.itertuples(index=False)
            }

            for threshold in settings.relevance_threshold_candidates:
                query_records = [
                    build_query_evaluation_record(
                        score_retrieval_query_evidence(
                            evidence_by_query_id[str(query.incident_id)],
                            threshold,
                        ),
                        query_split='validation',
                    )
                    for query in validation_queries.itertuples(index=False)
                ]
                query_summaries = pd.DataFrame.from_records(query_records)
                split_metrics = build_split_metrics(query_summaries).iloc[0]
                experiment_records.append(
                    {
                        'selection_split': 'validation',
                        'feature_version': feature_version,
                        'future_horizon_hours': float(horizon_hours),
                        'relevance_threshold': float(threshold),
                        'top_k': settings.top_k,
                        'candidate_policy': settings.candidate_policy,
                        'alarm_weight': settings.alarm_weight,
                        'shape_weight': (
                            0.0
                            if feature_version == 'alarm_tfidf_v1'
                            else settings.shape_weight
                        ),
                        'query_limit': max_validation_queries,
                        **split_metrics.drop(labels='split').to_dict(),
                    }
                )

    return pd.DataFrame.from_records(experiment_records)


def run_selected_test_evaluation(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    events: pd.DataFrame,
    settings: RetrievalExperimentSettings,
) -> pd.DataFrame:
    '''Evaluate the validation-selected retrieval settings on test once.'''
    if (
        settings.selected_feature_version is None
        or settings.future_horizon_hours is None
        or settings.relevance_threshold is None
    ):
        raise ValueError('selected retrieval settings are required')

    feature_variants = build_retrieval_feature_variants(
        documents,
        incidents,
        alarm_weight=settings.alarm_weight,
        shape_weight=settings.shape_weight,
    )
    try:
        selected_features = feature_variants[
            settings.selected_feature_version
        ]
    except KeyError as error:
        raise ValueError('selected feature version is unavailable') from error

    outcomes = build_future_alarm_outcomes(
        incidents,
        events,
        future_horizon_hours=settings.future_horizon_hours,
    )
    evaluation = evaluate_retrieval_splits(
        incidents,
        documents,
        selected_features,
        outcomes,
        relevance_threshold=settings.relevance_threshold,
        query_splits=('test',),
        top_k=settings.top_k,
        policy=settings.candidate_policy,
    )
    split_metrics = evaluation.split_metrics.iloc[0]
    if split_metrics['split'] != 'test':
        raise ValueError('selected evaluation must contain only test queries')

    return pd.DataFrame.from_records(
        [
            {
                'evaluation_split': 'test',
                'feature_version': settings.selected_feature_version,
                'future_horizon_hours': settings.future_horizon_hours,
                'relevance_threshold': settings.relevance_threshold,
                'top_k': settings.top_k,
                'candidate_policy': settings.candidate_policy,
                'alarm_weight': settings.alarm_weight,
                'shape_weight': (
                    0.0
                    if settings.selected_feature_version
                    == 'alarm_tfidf_v1'
                    else settings.shape_weight
                ),
                **split_metrics.drop(labels='split').to_dict(),
            }
        ]
    )
