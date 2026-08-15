'''Cached artifact loaders used only by the Streamlit runtime.'''

from pathlib import Path

import pandas as pd
import streamlit as st

from industrial_alarm_copilot.presentation.data import build_overview_data


@st.cache_data(show_spinner=False)
def load_overview_data(events_path: str, incidents_path: str):
    '''Read immutable artifacts once per file modification.'''
    event_file = Path(events_path)
    incident_file = Path(incidents_path)
    return build_overview_data(
        pd.read_parquet(event_file),
        pd.read_parquet(incident_file),
    )
