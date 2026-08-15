'''Portable forecast artifact loading and online Top-K scoring.'''

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from industrial_alarm_copilot.application.contracts import (
    ForecastPrediction,
    ObservedEpisode,
)


@dataclass(frozen=True, slots=True)
class RuntimeForecastModel:
    '''Validated transition model parameters loaded from portable JSON.'''

    model_version: str
    alarm_codes: tuple[str, ...]
    global_scores: np.ndarray
    machine_scores: dict[str, np.ndarray]
    machine_support: dict[str, int]
    transition_scores: dict[tuple[str, str], np.ndarray]
    transition_support: dict[tuple[str, str], int]


def _score_vector(values: Any, alarm_count: int, field_name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.shape != (alarm_count,):
        raise ValueError(f'{field_name} must align with alarm_codes')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError(f'{field_name} must contain finite scores in [0, 1]')
    scores.setflags(write=False)
    return scores


def parse_runtime_forecast_model(payload: dict[str, Any]) -> RuntimeForecastModel:
    '''Validate and normalize a Stage 6 transition model payload.'''
    if payload.get('model_version') != 'transition_frequency_v1':
        raise ValueError('unsupported forecast model_version')
    alarm_codes = tuple(str(code) for code in payload['alarm_codes'])
    if not alarm_codes or len(set(alarm_codes)) != len(alarm_codes):
        raise ValueError('alarm_codes must be nonempty and unique')
    alarm_count = len(alarm_codes)
    global_scores = _score_vector(
        payload['global_scores'], alarm_count, 'global_scores'
    )
    machine_scores = {
        str(record['machine_id']): _score_vector(
            record['scores'], alarm_count, 'machine_scores'
        )
        for record in payload['machine_scores']
    }
    machine_support = {
        str(record['machine_id']): int(record['sample_count'])
        for record in payload['machine_train_sample_counts']
    }
    transition_scores = {
        (str(record['machine_id']), str(record['last_alarm_code'])): (
            _score_vector(record['scores'], alarm_count, 'transition_scores')
        )
        for record in payload['transition_scores']
    }
    transition_support = {
        (str(record['machine_id']), str(record['last_alarm_code'])): int(
            record['sample_count']
        )
        for record in payload['transition_train_sample_counts']
    }
    if any(value < 0 for value in machine_support.values()):
        raise ValueError('machine support must be nonnegative')
    if any(value < 0 for value in transition_support.values()):
        raise ValueError('transition support must be nonnegative')
    return RuntimeForecastModel(
        model_version=str(payload['model_version']),
        alarm_codes=alarm_codes,
        global_scores=global_scores,
        machine_scores=machine_scores,
        machine_support=machine_support,
        transition_scores=transition_scores,
        transition_support=transition_support,
    )


def load_runtime_forecast_model(path: str | Path) -> RuntimeForecastModel:
    '''Load a portable Stage 6 model without fitting on UI data.'''
    with Path(path).open(encoding='utf-8') as model_file:
        payload = json.load(model_file)
    return parse_runtime_forecast_model(payload)


def predict_future_alarms(
    model: RuntimeForecastModel,
    observed: ObservedEpisode,
    top_k: int = 5,
    forecast_horizon_hours: float = 6.0,
) -> tuple[ForecastPrediction, ...]:
    '''Score one episode using transition, machine, then global fallback.'''
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if not math.isfinite(forecast_horizon_hours) or forecast_horizon_hours <= 0:
        raise ValueError('forecast_horizon_hours must be finite and positive')
    last_alarm_code = observed.alarm_sequence[-1].alarm_code
    state = (observed.machine_id, last_alarm_code)
    if state in model.transition_scores:
        scores = model.transition_scores[state]
        scope = 'transition'
        support = model.transition_support.get(state, 0)
    elif observed.machine_id in model.machine_scores:
        scores = model.machine_scores[observed.machine_id]
        scope = 'machine_fallback'
        support = model.machine_support.get(observed.machine_id, 0)
    else:
        scores = model.global_scores
        scope = 'global_fallback'
        support = sum(model.machine_support.values())

    ranking = sorted(
        range(len(model.alarm_codes)),
        key=lambda index: (-scores[index], model.alarm_codes[index]),
    )[: min(top_k, len(model.alarm_codes))]
    return tuple(
        ForecastPrediction(
            rank=rank,
            alarm_code=model.alarm_codes[column],
            model_score=float(scores[column]),
            model_version=model.model_version,
            forecast_horizon_hours=float(forecast_horizon_hours),
            baseline_scope=scope,
            train_support=support,
        )
        for rank, column in enumerate(ranking, start=1)
    )
