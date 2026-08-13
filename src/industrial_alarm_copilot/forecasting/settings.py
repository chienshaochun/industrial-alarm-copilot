'''Validated settings for future alarm forecasting experiments.'''

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ForecastExperimentSettings:
    '''Reproducible horizon, Top-K, and support-group settings.'''

    top_k: int
    forecast_horizon_hours_candidates: tuple[float, ...]
    selected_forecast_horizon_hours: float
    rare_max_train_support: int
    common_min_train_support: int
    minimum_machine_train_samples: int
    minimum_transition_train_samples: int
    linear_c: float
    linear_max_iter: int
    linear_class_weight_candidates: tuple[str, ...]


def parse_forecast_settings(
    settings: dict[str, Any],
) -> ForecastExperimentSettings:
    '''Validate raw TOML forecasting settings before label construction.'''
    top_k = int(settings['top_k'])
    horizons = tuple(
        float(value)
        for value in settings['forecast_horizon_hours_candidates']
    )
    selected_horizon = float(
        settings['selected_forecast_horizon_hours']
    )
    rare_max_support = int(settings['rare_max_train_support'])
    common_min_support = int(settings['common_min_train_support'])
    minimum_machine_samples = int(
        settings['minimum_machine_train_samples']
    )
    minimum_transition_samples = int(
        settings['minimum_transition_train_samples']
    )
    linear_c = float(settings['linear_c'])
    linear_max_iter = int(settings['linear_max_iter'])
    linear_class_weight_candidates = tuple(
        str(value) for value in settings['linear_class_weight_candidates']
    )

    if top_k <= 0:
        raise ValueError('forecast top_k must be greater than zero')
    if not horizons:
        raise ValueError('forecast horizon candidates cannot be empty')
    if len(set(horizons)) != len(horizons):
        raise ValueError('forecast horizon candidates must be unique')
    if any(not math.isfinite(value) or value <= 0 for value in horizons):
        raise ValueError('forecast horizons must be finite and positive')
    if selected_horizon not in horizons:
        raise ValueError('selected forecast horizon must be in candidates')
    if rare_max_support < 1:
        raise ValueError('rare support boundary must be positive')
    if common_min_support <= rare_max_support + 1:
        raise ValueError(
            'common support boundary must leave a medium support range'
        )
    if minimum_machine_samples < 1:
        raise ValueError('minimum machine train samples must be positive')
    if minimum_transition_samples < 1:
        raise ValueError('minimum transition train samples must be positive')
    if not math.isfinite(linear_c) or linear_c <= 0:
        raise ValueError('linear C must be finite and positive')
    if linear_max_iter < 1:
        raise ValueError('linear max_iter must be positive')
    if (
        not linear_class_weight_candidates
        or len(set(linear_class_weight_candidates))
        != len(linear_class_weight_candidates)
        or not set(linear_class_weight_candidates).issubset({'none', 'balanced'})
    ):
        raise ValueError('linear class weight candidates are invalid')

    return ForecastExperimentSettings(
        top_k=top_k,
        forecast_horizon_hours_candidates=horizons,
        selected_forecast_horizon_hours=selected_horizon,
        rare_max_train_support=rare_max_support,
        common_min_train_support=common_min_support,
        minimum_machine_train_samples=minimum_machine_samples,
        minimum_transition_train_samples=minimum_transition_samples,
        linear_c=linear_c,
        linear_max_iter=linear_max_iter,
        linear_class_weight_candidates=linear_class_weight_candidates,
    )
