'''Top-K multi-label forecasting evaluation with coverage diagnostics.'''

from dataclasses import dataclass

import numpy as np
import pandas as pd

from industrial_alarm_copilot.forecasting.vocabulary import (
    EncodedForecastLabels,
)


@dataclass(frozen=True)
class ForecastScoreMatrix:
    '''Model scores aligned with incident IDs and train-fitted labels.'''

    incident_ids: tuple[str, ...]
    alarm_codes: tuple[str, ...]
    scores: np.ndarray
    model_version: str


@dataclass(frozen=True)
class ForecastBatchEvaluation:
    '''Per-query Top-K results and one split-level metric row.'''

    query_metrics: pd.DataFrame
    per_label_metrics: pd.DataFrame
    split_metrics: pd.DataFrame


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_forecast_scores(
    labels: pd.DataFrame,
    encoded: EncodedForecastLabels,
    score_matrix: ForecastScoreMatrix,
    evaluation_split: str,
    top_k: int,
) -> ForecastBatchEvaluation:
    '''Evaluate known-label Top-K predictions without dropping coverage.'''
    if top_k <= 0:
        raise ValueError('top_k must be greater than zero')
    aligned_labels = labels.reset_index(drop=True)
    incident_ids = tuple(aligned_labels['incident_id'].astype(str))
    if incident_ids != encoded.incident_ids:
        raise ValueError('labels and encoded rows must be identically aligned')
    if score_matrix.incident_ids != encoded.incident_ids:
        raise ValueError('score and encoded rows must be identically aligned')
    if score_matrix.alarm_codes != encoded.alarm_codes:
        raise ValueError('score and encoded alarm codes must align')
    expected_shape = (len(incident_ids), len(encoded.alarm_codes))
    if score_matrix.scores.shape != expected_shape:
        raise ValueError('forecast score matrix shape is inconsistent')
    if not np.isfinite(score_matrix.scores).all():
        raise ValueError('forecast scores must be finite')

    split_mask = aligned_labels['split'].eq(evaluation_split).to_numpy()
    split_rows = np.flatnonzero(split_mask)
    if len(split_rows) == 0:
        raise ValueError('evaluation_split must select at least one episode')
    complete_mask = aligned_labels[
        'outcome_is_complete'
    ].astype(bool).to_numpy()
    complete_rows = split_rows[complete_mask[split_rows]]
    if len(complete_rows) == 0:
        raise ValueError('evaluation_split has no complete outcomes')

    effective_k = min(top_k, len(encoded.alarm_codes))
    code_order = np.argsort(
        np.asarray(encoded.alarm_codes, dtype=str),
        kind='stable',
    )
    ordered_scores = score_matrix.scores[complete_rows][:, code_order]
    ranked_in_code_order = np.argsort(
        -ordered_scores,
        axis=1,
        kind='stable',
    )[:, :effective_k]
    top_columns = code_order[ranked_in_code_order]

    truth = encoded.matrix[complete_rows].toarray().astype(bool)
    predictions = np.zeros_like(truth, dtype=bool)
    prediction_rows = np.arange(len(complete_rows))[:, None]
    predictions[prediction_rows, top_columns] = True
    true_positive_matrix = predictions & truth
    true_positive_count = true_positive_matrix.sum(axis=1)
    known_label_count = truth.sum(axis=1)
    precision_at_k = true_positive_count / effective_k
    recall_at_k = np.divide(
        true_positive_count,
        known_label_count,
        out=np.full(len(complete_rows), np.nan, dtype=float),
        where=known_label_count != 0,
    )
    unknown_code_counts = encoded.unknown_alarm_code_counts[complete_rows]
    unknown_event_counts = encoded.unknown_alarm_event_counts[complete_rows]

    query_metrics = pd.DataFrame(
        {
            'incident_id': aligned_labels.iloc[complete_rows][
                'incident_id'
            ].astype(str).to_numpy(),
            'split': evaluation_split,
            'model_version': score_matrix.model_version,
            'top_k': top_k,
            'known_label_count': known_label_count,
            'unknown_alarm_code_count': unknown_code_counts,
            'unknown_alarm_event_count': unknown_event_counts,
            'true_positive_count': true_positive_count,
            'hit_at_k': true_positive_count > 0,
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
        }
    )

    true_positive_total = int(true_positive_matrix.sum())
    predicted_positive_total = int(predictions.sum())
    known_positive_total = int(truth.sum())
    micro_precision = (
        true_positive_total / predicted_positive_total
        if predicted_positive_total
        else 0.0
    )
    micro_recall = (
        true_positive_total / known_positive_total
        if known_positive_total
        else 0.0
    )
    defined_recall = recall_at_k[~np.isnan(recall_at_k)]

    label_support = truth.sum(axis=0)
    supported_label_mask = label_support > 0
    label_true_positive = true_positive_matrix.sum(axis=0)
    label_predicted_positive = predictions.sum(axis=0)
    label_precision = np.divide(
        label_true_positive,
        label_predicted_positive,
        out=np.zeros(len(encoded.alarm_codes), dtype=float),
        where=label_predicted_positive != 0,
    )
    label_recall = np.divide(
        label_true_positive,
        label_support,
        out=np.zeros(len(encoded.alarm_codes), dtype=float),
        where=label_support != 0,
    )
    label_f1 = np.divide(
        2 * label_precision * label_recall,
        label_precision + label_recall,
        out=np.zeros(len(encoded.alarm_codes), dtype=float),
        where=(label_precision + label_recall) != 0,
    )
    supported_label_count = int(supported_label_mask.sum())
    macro_f1 = (
        float(label_f1[supported_label_mask].mean())
        if supported_label_count
        else float('nan')
    )
    per_label_metrics = pd.DataFrame(
        {
            'alarm_code': encoded.alarm_codes,
            'evaluation_support': label_support.astype(np.int64),
            'predicted_positive_count': label_predicted_positive.astype(
                np.int64
            ),
            'true_positive_count': label_true_positive.astype(np.int64),
            'precision_at_k': label_precision,
            'recall_at_k': label_recall,
            'f1_at_k': label_f1,
        }
    )

    split_metrics = pd.DataFrame.from_records(
        [
            {
                'split': evaluation_split,
                'model_version': score_matrix.model_version,
                'top_k': top_k,
                'episode_count': len(split_rows),
                'complete_outcome_count': len(complete_rows),
                'outcome_coverage': len(complete_rows) / len(split_rows),
                'complete_empty_known_label_count': int(
                    (known_label_count == 0).sum()
                ),
                'unknown_label_query_count': int(
                    (unknown_code_counts > 0).sum()
                ),
                'mean_hit_at_k': float(query_metrics['hit_at_k'].mean()),
                'mean_precision_at_k': float(precision_at_k.mean()),
                'mean_recall_at_k': (
                    float(defined_recall.mean())
                    if len(defined_recall)
                    else float('nan')
                ),
                'micro_precision_at_k': micro_precision,
                'micro_recall_at_k': micro_recall,
                'micro_f1_at_k': _safe_f1(
                    micro_precision,
                    micro_recall,
                ),
                'macro_f1_at_k': macro_f1,
                'evaluated_label_count': supported_label_count,
            }
        ]
    )
    return ForecastBatchEvaluation(
        query_metrics=query_metrics,
        per_label_metrics=per_label_metrics,
        split_metrics=split_metrics,
    )


def build_forecast_support_group_metrics(
    per_label_metrics: pd.DataFrame,
    grouped_train_support: pd.DataFrame,
) -> pd.DataFrame:
    '''Aggregate per-label Top-K results by train-only support groups.'''
    required_metric_columns = {
        'alarm_code',
        'evaluation_support',
        'predicted_positive_count',
        'true_positive_count',
        'f1_at_k',
    }
    if not required_metric_columns.issubset(per_label_metrics.columns):
        raise ValueError('per-label metrics are missing required columns')
    required_support_columns = {
        'alarm_code',
        'train_sample_support',
        'support_group',
    }
    if not required_support_columns.issubset(grouped_train_support.columns):
        raise ValueError('train support profile is missing required columns')

    combined = per_label_metrics.merge(
        grouped_train_support[list(required_support_columns)],
        on='alarm_code',
        how='left',
        validate='one_to_one',
    )
    if combined['support_group'].isna().any():
        raise ValueError('every forecast label must have a support group')

    records = []
    for support_group in ('rare', 'medium', 'common'):
        group = combined.loc[
            combined['support_group'].eq(support_group)
        ]
        supported = group.loc[group['evaluation_support'].gt(0)]
        true_positive_count = int(group['true_positive_count'].sum())
        predicted_positive_count = int(
            group['predicted_positive_count'].sum()
        )
        evaluation_positive_count = int(group['evaluation_support'].sum())
        micro_precision = (
            true_positive_count / predicted_positive_count
            if predicted_positive_count
            else 0.0
        )
        micro_recall = (
            true_positive_count / evaluation_positive_count
            if evaluation_positive_count
            else 0.0
        )
        records.append(
            {
                'support_group': support_group,
                'label_count': len(group),
                'evaluated_label_count': len(supported),
                'evaluation_positive_count': evaluation_positive_count,
                'predicted_positive_count': predicted_positive_count,
                'true_positive_count': true_positive_count,
                'micro_precision_at_k': micro_precision,
                'micro_recall_at_k': micro_recall,
                'micro_f1_at_k': _safe_f1(
                    micro_precision,
                    micro_recall,
                ),
                'macro_f1_at_k': (
                    float(supported['f1_at_k'].mean())
                    if len(supported)
                    else float('nan')
                ),
            }
        )
    return pd.DataFrame.from_records(records)
