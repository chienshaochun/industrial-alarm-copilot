'''Forecasting artifact loading and locked-model evaluation.'''

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_alarm_copilot.forecasting.baselines import (
    TransitionFrequencyBaseline,
    fit_transition_frequency_baseline,
    score_transition_frequency_baseline,
)
from industrial_alarm_copilot.forecasting.evaluation import (
    ForecastBatchEvaluation,
    build_forecast_support_group_metrics,
    evaluate_forecast_scores,
)
from industrial_alarm_copilot.forecasting.labels import build_forecast_labels
from industrial_alarm_copilot.forecasting.profile import (
    assign_forecast_support_groups,
    build_forecast_alarm_support_profile,
)
from industrial_alarm_copilot.forecasting.settings import (
    ForecastExperimentSettings,
    parse_forecast_settings,
)
from industrial_alarm_copilot.forecasting.vocabulary import (
    EncodedForecastLabels,
    encode_forecast_labels,
    fit_forecast_label_vocabulary,
)
from industrial_alarm_copilot.retrieval.pipeline import (
    load_retrieval_experiment_inputs,
)


@dataclass(frozen=True)
class ForecastPreparedInputs:
    '''Aligned tables and train-fitted targets for forecast evaluation.'''

    events: pd.DataFrame
    incidents: pd.DataFrame
    labels: pd.DataFrame
    encoded: EncodedForecastLabels
    incident_context: pd.DataFrame
    grouped_support: pd.DataFrame


@dataclass(frozen=True)
class SelectedForecastRun:
    '''Locked transition model, evaluation, and support diagnostics.'''

    baseline: TransitionFrequencyBaseline
    evaluation: ForecastBatchEvaluation
    support_group_metrics: pd.DataFrame
    scope_profile: pd.DataFrame


def prepare_forecast_inputs(
    events_parquet_path: str | Path,
    incidents_parquet_path: str | Path,
    incident_events_parquet_path: str | Path,
    settings: ForecastExperimentSettings,
) -> ForecastPreparedInputs:
    '''Build 6-hour targets and train-only vocabularies from prior artifacts.'''
    retrieval_inputs = load_retrieval_experiment_inputs(
        events_parquet_path,
        incidents_parquet_path,
        incident_events_parquet_path,
    )
    labels = build_forecast_labels(
        retrieval_inputs.incidents,
        retrieval_inputs.events,
        settings.selected_forecast_horizon_hours,
    )
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)
    support = build_forecast_alarm_support_profile(vocabulary)
    grouped_support = assign_forecast_support_groups(
        support,
        settings.rare_max_train_support,
        settings.common_min_train_support,
    )

    documents = retrieval_inputs.documents.copy()
    documents['incident_id'] = documents['incident_id'].astype(str)
    document_by_id = documents.set_index('incident_id')['alarm_document']
    incident_ids = retrieval_inputs.incidents['incident_id'].astype(str)
    incident_context = pd.DataFrame(
        {
            'incident_id': incident_ids,
            'last_alarm_code': [
                str(document_by_id.loc[incident_id]).split()[-1]
                for incident_id in incident_ids
            ],
        }
    )
    return ForecastPreparedInputs(
        events=retrieval_inputs.events,
        incidents=retrieval_inputs.incidents,
        labels=labels,
        encoded=encoded,
        incident_context=incident_context,
        grouped_support=grouped_support,
    )


def run_selected_forecast_evaluation(
    inputs: ForecastPreparedInputs,
    settings: ForecastExperimentSettings,
    evaluation_split: str,
) -> SelectedForecastRun:
    '''Fit the locked train-only transition model and evaluate one split.'''
    if settings.selected_model_version != 'transition_frequency_v1':
        raise ValueError('only the locked transition model is supported')
    baseline = fit_transition_frequency_baseline(
        inputs.labels,
        inputs.encoded,
        inputs.incident_context,
        settings.minimum_machine_train_samples,
        settings.minimum_transition_train_samples,
    )
    predictions = score_transition_frequency_baseline(
        baseline,
        incident_ids=inputs.encoded.incident_ids,
        machine_ids=tuple(inputs.labels['machine_id'].astype(str)),
        last_alarm_codes=tuple(
            inputs.incident_context['last_alarm_code'].astype(str)
        ),
    )
    evaluation = evaluate_forecast_scores(
        inputs.labels,
        inputs.encoded,
        predictions.score_matrix,
        evaluation_split=evaluation_split,
        top_k=settings.top_k,
    )
    support_metrics = build_forecast_support_group_metrics(
        evaluation.per_label_metrics,
        inputs.grouped_support,
    )
    split_mask = inputs.labels['split'].eq(evaluation_split).to_numpy()
    complete_mask = inputs.labels['outcome_is_complete'].astype(bool).to_numpy()
    selected_rows = split_mask & complete_mask
    scope_profile = (
        pd.Series(predictions.baseline_scopes, name='baseline_scope')
        .loc[selected_rows]
        .value_counts()
        .rename_axis('baseline_scope')
        .reset_index(name='query_count')
    )
    return SelectedForecastRun(
        baseline=baseline,
        evaluation=evaluation,
        support_group_metrics=support_metrics,
        scope_profile=scope_profile,
    )


def run_selected_forecast_from_artifacts(
    events_parquet_path: str | Path,
    incidents_parquet_path: str | Path,
    incident_events_parquet_path: str | Path,
    pipeline_settings: dict[str, Any],
    evaluation_split: str = 'test',
) -> SelectedForecastRun:
    '''Run the locked forecast model from Stage 3 and 4 artifacts.'''
    settings = parse_forecast_settings(pipeline_settings['forecasting'])
    inputs = prepare_forecast_inputs(
        events_parquet_path,
        incidents_parquet_path,
        incident_events_parquet_path,
        settings,
    )
    return run_selected_forecast_evaluation(
        inputs,
        settings,
        evaluation_split=evaluation_split,
    )
