'''Train-only one-vs-rest linear forecasting models.'''

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from industrial_alarm_copilot.forecasting.evaluation import ForecastScoreMatrix
from industrial_alarm_copilot.forecasting.features import ForecastFeatureMatrix
from industrial_alarm_copilot.forecasting.vocabulary import EncodedForecastLabels


LinearClassWeight = Literal['none', 'balanced']


@dataclass(frozen=True)
class FittedForecastLinearModel:
    '''A multi-label logistic model fitted on complete train rows.'''

    estimator: Any
    alarm_codes: tuple[str, ...]
    feature_version: str
    feature_names: tuple[str, ...]
    train_sample_count: int
    class_weight_mode: LinearClassWeight
    model_version: str


def _validate_alignment(features, labels, encoded) -> pd.DataFrame:
    aligned_labels = labels.reset_index(drop=True)
    incident_ids = tuple(aligned_labels['incident_id'].astype(str))
    if incident_ids != features.incident_ids or incident_ids != encoded.incident_ids:
        raise ValueError('forecast features and labels must be identically aligned')
    if features.matrix.shape[0] != len(incident_ids):
        raise ValueError('forecast feature row count is inconsistent')
    return aligned_labels


def fit_forecast_linear_model(
    features: ForecastFeatureMatrix,
    labels: pd.DataFrame,
    encoded: EncodedForecastLabels,
    class_weight_mode: LinearClassWeight,
    c: float = 1.0,
    max_iter: int = 300,
) -> FittedForecastLinearModel:
    '''Fit independent logistic classifiers on complete train outcomes.'''
    if class_weight_mode not in ('none', 'balanced'):
        raise ValueError('class_weight_mode must be none or balanced')
    if not np.isfinite(c) or c <= 0:
        raise ValueError('linear C must be finite and positive')
    if max_iter < 1:
        raise ValueError('linear max_iter must be positive')

    aligned_labels = _validate_alignment(features, labels, encoded)
    train_mask = (
        aligned_labels['split'].eq('train')
        & aligned_labels['outcome_is_complete'].astype(bool)
    ).to_numpy()
    train_sample_count = int(train_mask.sum())
    if train_sample_count == 0:
        raise ValueError('at least one complete train outcome is required')

    estimator = OneVsRestClassifier(
        LogisticRegression(
            C=float(c),
            class_weight='balanced' if class_weight_mode == 'balanced' else None,
            max_iter=int(max_iter),
            solver='liblinear',
            random_state=0,
        ),
        n_jobs=1,
    )
    estimator.fit(features.matrix[train_mask], encoded.matrix[train_mask])
    return FittedForecastLinearModel(
        estimator=estimator,
        alarm_codes=encoded.alarm_codes,
        feature_version=features.feature_version,
        feature_names=features.feature_names,
        train_sample_count=train_sample_count,
        class_weight_mode=class_weight_mode,
        model_version=f'ovr_logistic_{class_weight_mode}_v1',
    )


def score_forecast_linear_model(
    model: FittedForecastLinearModel,
    features: ForecastFeatureMatrix,
) -> ForecastScoreMatrix:
    '''Return an independent probability for every alarm and episode.'''
    if features.feature_version != model.feature_version:
        raise ValueError('forecast feature version does not match model')
    if features.feature_names != model.feature_names:
        raise ValueError('forecast feature columns do not match model')
    scores = np.asarray(model.estimator.predict_proba(features.matrix), dtype=float)
    if scores.shape != (len(features.incident_ids), len(model.alarm_codes)):
        raise ValueError('linear forecast score shape is inconsistent')
    return ForecastScoreMatrix(
        incident_ids=features.incident_ids,
        alarm_codes=model.alarm_codes,
        scores=np.clip(scores, 0.0, 1.0),
        model_version=model.model_version,
    )
