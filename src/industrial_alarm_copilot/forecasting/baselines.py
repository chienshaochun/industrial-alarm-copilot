'''Deterministic future-alarm forecasting baselines.'''

from dataclasses import dataclass

import numpy as np
import pandas as pd

from industrial_alarm_copilot.forecasting.vocabulary import (
    EncodedForecastLabels,
)
from industrial_alarm_copilot.forecasting.evaluation import (
    ForecastScoreMatrix,
)


GLOBAL_FREQUENCY_MODEL_VERSION = 'global_frequency_v1'


@dataclass(frozen=True)
class GlobalFrequencyBaseline:
    '''Train-only marginal probability for every forecast alarm label.'''

    alarm_codes: tuple[str, ...]
    scores: np.ndarray
    train_sample_count: int
    model_version: str = GLOBAL_FREQUENCY_MODEL_VERSION


def _validate_label_alignment(
    labels: pd.DataFrame,
    encoded: EncodedForecastLabels,
) -> pd.DataFrame:
    aligned_labels = labels.reset_index(drop=True)
    incident_ids = tuple(aligned_labels['incident_id'].astype(str))
    if incident_ids != encoded.incident_ids:
        raise ValueError('labels and encoded rows must be identically aligned')
    if encoded.matrix.shape != (
        len(aligned_labels),
        len(encoded.alarm_codes),
    ):
        raise ValueError('encoded label matrix shape is inconsistent')
    return aligned_labels


def fit_global_frequency_baseline(
    labels: pd.DataFrame,
    encoded: EncodedForecastLabels,
) -> GlobalFrequencyBaseline:
    '''Fit marginal alarm probabilities on complete train outcomes only.'''
    aligned_labels = _validate_label_alignment(labels, encoded)
    train_mask = (
        aligned_labels['split'].eq('train')
        & aligned_labels['outcome_is_complete'].astype(bool)
    ).to_numpy()
    train_sample_count = int(train_mask.sum())
    if train_sample_count == 0:
        raise ValueError('at least one complete train label is required')

    positive_counts = np.asarray(
        encoded.matrix[train_mask].sum(axis=0)
    ).ravel()
    scores = positive_counts.astype(float) / train_sample_count
    return GlobalFrequencyBaseline(
        alarm_codes=encoded.alarm_codes,
        scores=scores,
        train_sample_count=train_sample_count,
    )


def rank_global_frequency_baseline(
    baseline: GlobalFrequencyBaseline,
    top_k: int,
) -> pd.DataFrame:
    '''Return deterministic global Top-K alarms and train frequencies.'''
    if top_k <= 0:
        raise ValueError('top_k must be greater than zero')
    if len(baseline.alarm_codes) != len(baseline.scores):
        raise ValueError('baseline alarm codes and scores must align')

    ranking = pd.DataFrame(
        {
            'alarm_code': baseline.alarm_codes,
            'model_score': baseline.scores,
        }
    ).sort_values(
        ['model_score', 'alarm_code'],
        ascending=[False, True],
        kind='stable',
    )
    ranking = ranking.head(min(top_k, len(ranking))).reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    ranking['model_version'] = baseline.model_version
    return ranking[
        ['rank', 'alarm_code', 'model_score', 'model_version']
    ]


def score_global_frequency_baseline(
    baseline: GlobalFrequencyBaseline,
    incident_ids: tuple[str, ...],
) -> ForecastScoreMatrix:
    '''Broadcast the same global train frequencies to every query.'''
    if len(set(incident_ids)) != len(incident_ids):
        raise ValueError('forecast score incident_ids must be unique')
    scores = np.broadcast_to(
        baseline.scores,
        (len(incident_ids), len(baseline.scores)),
    ).copy()
    return ForecastScoreMatrix(
        incident_ids=incident_ids,
        alarm_codes=baseline.alarm_codes,
        scores=scores,
        model_version=baseline.model_version,
    )
