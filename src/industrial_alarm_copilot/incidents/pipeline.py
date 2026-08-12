'''In-memory incident analysis pipeline.'''

from dataclasses import dataclass

import pandas as pd

from industrial_alarm_copilot.incidents.baselines import (
    GlobalIncidentBaseline,
    apply_incident_baseline_flags,
    fit_global_incident_baseline,
    fit_machine_incident_baselines,
)
from industrial_alarm_copilot.incidents.mapping import (
    build_incident_event_mapping,
)
from industrial_alarm_copilot.incidents.summary import build_incident_summary


@dataclass(frozen=True)
class IncidentAnalysis:
    '''Related tables and fitted baselines from one incident build.'''

    incidents: pd.DataFrame
    incident_events: pd.DataFrame
    global_baseline: GlobalIncidentBaseline
    machine_baselines: pd.DataFrame


def build_incident_analysis(
    events: pd.DataFrame,
    gap_minutes: float = 30.0,
    baseline_quantile: float = 0.95,
    minimum_incident_count: int = 200,
) -> IncidentAnalysis:
    '''Build episode summaries, mappings, and train-only baseline flags.'''
    incident_summary = build_incident_summary(
        events,
        gap_minutes=gap_minutes,
    )
    incident_events = build_incident_event_mapping(
        events,
        gap_minutes=gap_minutes,
    )
    global_baseline = fit_global_incident_baseline(
        incident_summary,
        quantile=baseline_quantile,
    )
    machine_baselines = fit_machine_incident_baselines(
        incident_summary,
        quantile=baseline_quantile,
        minimum_incident_count=minimum_incident_count,
    )
    flagged_incidents = apply_incident_baseline_flags(
        incident_summary,
        global_baseline,
        machine_baselines,
    )

    return IncidentAnalysis(
        incidents=flagged_incidents,
        incident_events=incident_events,
        global_baseline=global_baseline,
        machine_baselines=machine_baselines,
    )
