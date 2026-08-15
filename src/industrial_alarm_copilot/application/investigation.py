'''Load reusable artifacts and orchestrate one evidence-safe investigation.'''

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_alarm_copilot.application.contracts import (
    EpisodeOption,
    InvestigationResult,
)
from industrial_alarm_copilot.application.forecast import (
    RuntimeForecastModel,
    load_runtime_forecast_model,
    predict_future_alarms,
)
from industrial_alarm_copilot.application.observed import (
    build_observed_episode,
)
from industrial_alarm_copilot.application.retrieved import (
    build_retrieved_episode_evidence,
)
from industrial_alarm_copilot.retrieval.candidates import (
    select_historical_candidates,
)
from industrial_alarm_copilot.retrieval.features import (
    build_retrieval_feature_variants,
)
from industrial_alarm_copilot.retrieval.outcomes import (
    build_future_alarm_outcomes,
)
from industrial_alarm_copilot.retrieval.pipeline import (
    load_retrieval_experiment_inputs,
)
from industrial_alarm_copilot.retrieval.search import (
    RetrievalSearchIndex,
    build_retrieval_search_index,
)
from industrial_alarm_copilot.retrieval.settings import (
    RetrievalExperimentSettings,
    parse_retrieval_settings,
)


DEFAULT_LIMITATIONS = (
    'incident 是依 30 分鐘事件間隔推導的 episode，不等同已確認的根因事件。',
    '相似事件只代表 alarm 組成與 episode 外型接近，不證明因果關係。',
    '未來 Alarm 是統計候選，不是設備故障診斷或維修指令。',
)


@dataclass(frozen=True)
class InvestigationResources:
    '''Read-only tables and fitted models reused across UI interactions.'''

    events: pd.DataFrame
    incidents: pd.DataFrame
    incident_events: pd.DataFrame
    outcomes: pd.DataFrame
    search_index: RetrievalSearchIndex
    forecast_model: RuntimeForecastModel
    retrieval_settings: RetrievalExperimentSettings
    forecast_top_k: int
    forecast_horizon_hours: float


def load_investigation_resources(
    events_path: str | Path,
    incidents_path: str | Path,
    incident_events_path: str | Path,
    forecast_model_path: str | Path,
    pipeline_settings: dict[str, Any],
) -> InvestigationResources:
    '''Load artifacts once and build only train-fitted runtime indexes.'''
    inputs = load_retrieval_experiment_inputs(
        events_path,
        incidents_path,
        incident_events_path,
    )
    retrieval_settings = parse_retrieval_settings(
        pipeline_settings['retrieval']
    )
    if retrieval_settings.selected_feature_version is None:
        raise ValueError('retrieval selection must be locked for the app')
    if retrieval_settings.future_horizon_hours is None:
        raise ValueError('retrieval future horizon must be locked for the app')
    feature_variants = build_retrieval_feature_variants(
        inputs.documents,
        inputs.incidents,
        alarm_weight=retrieval_settings.alarm_weight,
        shape_weight=retrieval_settings.shape_weight,
    )
    selected_features = feature_variants[
        retrieval_settings.selected_feature_version
    ]
    forecast_settings = pipeline_settings['forecasting']
    return InvestigationResources(
        events=inputs.events,
        incidents=inputs.incidents,
        incident_events=inputs.incident_events,
        outcomes=build_future_alarm_outcomes(
            inputs.incidents,
            inputs.events,
            retrieval_settings.future_horizon_hours,
        ),
        search_index=build_retrieval_search_index(
            inputs.incidents,
            inputs.documents,
            selected_features,
        ),
        forecast_model=load_runtime_forecast_model(forecast_model_path),
        retrieval_settings=retrieval_settings,
        forecast_top_k=int(forecast_settings['top_k']),
        forecast_horizon_hours=float(
            forecast_settings['selected_forecast_horizon_hours']
        ),
    )


class InvestigationService:
    '''Application boundary consumed by Streamlit and future APIs.'''

    def __init__(self, resources: InvestigationResources) -> None:
        self._resources = resources

    def machine_ids(self) -> tuple[str, ...]:
        '''Return stable machine choices with numeric-aware ordering.'''
        values = set(self._resources.incidents['machine_id'].astype(str))
        return tuple(
            sorted(values, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        )

    def episode_options(self, machine_id: str) -> tuple[EpisodeOption, ...]:
        '''Return newest-first selector records for one machine.'''
        selected = self._resources.incidents.loc[
            self._resources.incidents['machine_id'].astype(str).eq(
                str(machine_id)
            )
        ].sort_values('start_time', ascending=False, kind='stable')
        return tuple(
            EpisodeOption(
                incident_id=str(row.incident_id),
                machine_id=str(row.machine_id),
                split=str(row.split),
                start_time=row.start_time.to_pydatetime(),
                end_time=row.end_time.to_pydatetime(),
                event_count=int(row.event_count),
                is_upper_tail=bool(row.is_upper_tail),
            )
            for row in selected.itertuples(index=False)
        )

    def investigate(self, incident_id: str) -> InvestigationResult:
        '''Assemble observed facts, historical evidence, and predictions.'''
        resources = self._resources
        observed = build_observed_episode(
            resources.incidents,
            resources.events,
            resources.incident_events,
            incident_id,
        )
        policy = resources.retrieval_settings.candidate_policy
        candidates = select_historical_candidates(
            resources.incidents,
            incident_id,
            policy=policy,
        )
        candidate_rows = np.asarray(
            [
                resources.search_index.row_by_incident_id[str(value)]
                for value in candidates['incident_id']
            ],
            dtype=np.int64,
        )
        retrieval_results = resources.search_index.retrieve_candidate_rows(
            incident_id,
            candidate_rows,
            top_k=resources.retrieval_settings.top_k,
            policy=policy,
        )
        evidence = build_retrieved_episode_evidence(
            retrieval_results,
            resources.outcomes,
        )
        predictions = predict_future_alarms(
            resources.forecast_model,
            observed,
            top_k=resources.forecast_top_k,
            forecast_horizon_hours=resources.forecast_horizon_hours,
        )
        return InvestigationResult(
            observed=observed,
            retrieved_evidence=evidence,
            predictions=predictions,
            limitations=DEFAULT_LIMITATIONS,
        )
