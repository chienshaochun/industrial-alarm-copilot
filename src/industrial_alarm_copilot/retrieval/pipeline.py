'''Artifact loading and orchestration for retrieval experiments.'''

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_alarm_copilot.retrieval.documents import (
    build_alarm_documents,
)
from industrial_alarm_copilot.retrieval.experiments import (
    run_selected_test_evaluation,
    run_validation_experiment_grid,
)
from industrial_alarm_copilot.retrieval.settings import (
    parse_retrieval_settings,
)


@dataclass(frozen=True)
class RetrievalExperimentInputs:
    '''Aligned artifact tables used by retrieval experiments.'''

    events: pd.DataFrame
    incidents: pd.DataFrame
    incident_events: pd.DataFrame
    documents: pd.DataFrame


def load_retrieval_experiment_inputs(
    events_parquet_path: str | Path,
    incidents_parquet_path: str | Path,
    incident_events_parquet_path: str | Path,
) -> RetrievalExperimentInputs:
    '''Load compatible Stage 3 and Stage 4 artifacts.'''
    events = pd.read_parquet(events_parquet_path)
    incidents = pd.read_parquet(incidents_parquet_path)
    incident_events = pd.read_parquet(incident_events_parquet_path)
    documents = build_alarm_documents(events, incident_events)

    incident_ids = set(incidents['incident_id'].astype(str))
    document_ids = set(documents['incident_id'].astype(str))
    if incident_ids != document_ids:
        raise ValueError('every incident must have exactly one alarm document')

    return RetrievalExperimentInputs(
        events=events,
        incidents=incidents,
        incident_events=incident_events,
        documents=documents,
    )


def run_validation_from_artifacts(
    events_parquet_path: str | Path,
    incidents_parquet_path: str | Path,
    incident_events_parquet_path: str | Path,
    pipeline_settings: dict[str, Any],
    max_validation_queries: int | None = None,
) -> pd.DataFrame:
    '''Run the validation grid from reproducible processed artifacts.'''
    inputs = load_retrieval_experiment_inputs(
        events_parquet_path,
        incidents_parquet_path,
        incident_events_parquet_path,
    )
    retrieval_settings = parse_retrieval_settings(
        pipeline_settings['retrieval']
    )
    return run_validation_experiment_grid(
        inputs.incidents,
        inputs.documents,
        inputs.events,
        retrieval_settings,
        max_validation_queries=max_validation_queries,
    )


def run_selected_test_from_artifacts(
    events_parquet_path: str | Path,
    incidents_parquet_path: str | Path,
    incident_events_parquet_path: str | Path,
    pipeline_settings: dict[str, Any],
) -> pd.DataFrame:
    '''Run the locked retrieval setting on test artifacts.'''
    inputs = load_retrieval_experiment_inputs(
        events_parquet_path,
        incidents_parquet_path,
        incident_events_parquet_path,
    )
    retrieval_settings = parse_retrieval_settings(
        pipeline_settings['retrieval']
    )
    return run_selected_test_evaluation(
        inputs.incidents,
        inputs.documents,
        inputs.events,
        retrieval_settings,
    )
