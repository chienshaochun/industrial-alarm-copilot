'''Cached artifact loaders used only by the Streamlit runtime.'''

from pathlib import Path

import pandas as pd
import streamlit as st

from industrial_alarm_copilot.application.investigation import (
    InvestigationService,
    load_investigation_resources,
)
from industrial_alarm_copilot.data.runtime import load_pipeline_settings
from industrial_alarm_copilot.presentation.data import (
    build_evaluation_data,
    build_overview_data,
)


@st.cache_data(show_spinner=False)
def load_overview_data(events_path: str, incidents_path: str):
    '''Read immutable artifacts once per file modification.'''
    event_file = Path(events_path)
    incident_file = Path(incidents_path)
    return build_overview_data(
        pd.read_parquet(event_file),
        pd.read_parquet(incident_file),
    )


@st.cache_resource(show_spinner=False)
def load_investigation_service(project_root: str) -> InvestigationService:
    '''Build expensive retrieval features once per Streamlit process.'''
    root = Path(project_root)
    processed = root / 'data' / 'processed'
    settings = load_pipeline_settings(root / 'configs' / 'default.toml')
    resources = load_investigation_resources(
        processed / 'events.parquet',
        processed / 'incidents.parquet',
        processed / 'incident_events.parquet',
        processed / 'forecast_model.json',
        settings,
    )
    return InvestigationService(resources)


@st.cache_data(show_spinner=False)
def load_evaluation_data(
    retrieval_results_path: str,
    forecast_results_path: str,
    support_groups_path: str,
):
    '''Read the locked test reports without recomputing evaluation.'''
    return build_evaluation_data(
        pd.read_csv(retrieval_results_path),
        pd.read_csv(forecast_results_path),
        pd.read_csv(support_groups_path),
    )
