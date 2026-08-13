'''Validated settings for retrieval validation experiments.'''

from dataclasses import dataclass
import math
from typing import Any

from industrial_alarm_copilot.retrieval.candidates import (
    CandidatePolicy,
    VALID_CANDIDATE_POLICIES,
)


@dataclass(frozen=True)
class RetrievalExperimentSettings:
    '''Reproducible Top-K and validation-grid settings.'''

    top_k: int
    alarm_weight: float
    shape_weight: float
    candidate_policy: CandidatePolicy
    future_horizon_hours_candidates: tuple[float, ...]
    relevance_threshold_candidates: tuple[float, ...]
    selected_feature_version: str | None = None
    future_horizon_hours: float | None = None
    relevance_threshold: float | None = None


def _unique_numeric_values(
    values: Any,
    setting_name: str,
) -> tuple[float, ...]:
    numeric_values = tuple(float(value) for value in values)
    if not numeric_values:
        raise ValueError(f'{setting_name} cannot be empty')
    if len(set(numeric_values)) != len(numeric_values):
        raise ValueError(f'{setting_name} must contain unique values')
    return numeric_values


def parse_retrieval_settings(
    settings: dict[str, Any],
) -> RetrievalExperimentSettings:
    '''Validate raw TOML retrieval settings before an experiment.'''
    top_k = int(settings['top_k'])
    alarm_weight = float(settings['alarm_weight'])
    shape_weight = float(settings['shape_weight'])
    candidate_policy = str(settings['candidate_policy'])
    horizons = _unique_numeric_values(
        settings['future_horizon_hours_candidates'],
        'future_horizon_hours_candidates',
    )
    thresholds = _unique_numeric_values(
        settings['relevance_threshold_candidates'],
        'relevance_threshold_candidates',
    )
    selection_keys = {
        'selected_feature_version',
        'future_horizon_hours',
        'relevance_threshold',
    }
    supplied_selection_keys = selection_keys.intersection(settings)
    if supplied_selection_keys and supplied_selection_keys != selection_keys:
        raise ValueError('selected retrieval settings must be supplied together')
    selected_feature_version = settings.get('selected_feature_version')
    selected_horizon = (
        float(settings['future_horizon_hours'])
        if 'future_horizon_hours' in settings
        else None
    )
    selected_threshold = (
        float(settings['relevance_threshold'])
        if 'relevance_threshold' in settings
        else None
    )

    if top_k <= 0:
        raise ValueError('top_k must be greater than zero')
    if not all(math.isfinite(weight) for weight in (
        alarm_weight,
        shape_weight,
    )):
        raise ValueError('feature weights must be finite')
    if alarm_weight < 0 or shape_weight < 0:
        raise ValueError('feature weights must be nonnegative')
    if alarm_weight == 0 and shape_weight == 0:
        raise ValueError('at least one feature weight must be positive')
    if candidate_policy not in VALID_CANDIDATE_POLICIES:
        raise ValueError('candidate_policy is unsupported')
    if any(
        not math.isfinite(horizon) or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError('future horizons must be greater than zero')
    if any(not 0 < threshold <= 1 for threshold in thresholds):
        raise ValueError('relevance thresholds must be in (0, 1]')
    if selected_feature_version is not None:
        supported_features = {
            'alarm_tfidf_v1',
            'alarm_plus_shape_v1',
        }
        if selected_feature_version not in supported_features:
            raise ValueError('selected_feature_version is unsupported')
        if selected_horizon not in horizons:
            raise ValueError('selected future horizon must be in candidates')
        if selected_threshold not in thresholds:
            raise ValueError(
                'selected relevance threshold must be in candidates'
            )

    return RetrievalExperimentSettings(
        top_k=top_k,
        alarm_weight=alarm_weight,
        shape_weight=shape_weight,
        candidate_policy=candidate_policy,
        future_horizon_hours_candidates=horizons,
        relevance_threshold_candidates=thresholds,
        selected_feature_version=selected_feature_version,
        future_horizon_hours=selected_horizon,
        relevance_threshold=selected_threshold,
    )
