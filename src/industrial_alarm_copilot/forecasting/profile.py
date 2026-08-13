'''Forecast label coverage, density, and support profiles.'''

import numpy as np
import pandas as pd

from industrial_alarm_copilot.forecasting.vocabulary import (
    EncodedForecastLabels,
    ForecastLabelVocabulary,
)


def build_forecast_split_profile(
    labels: pd.DataFrame,
    encoded: EncodedForecastLabels,
) -> pd.DataFrame:
    '''Summarize outcome completeness and known/unknown targets by split.'''
    aligned_labels = labels.reset_index(drop=True)
    incident_ids = tuple(aligned_labels['incident_id'].astype(str))
    if incident_ids != encoded.incident_ids:
        raise ValueError('labels and encoded rows must be identically aligned')

    known_label_counts = np.asarray(encoded.matrix.sum(axis=1)).ravel()
    records = []
    for split, split_rows in aligned_labels.groupby(
        'split',
        observed=True,
        sort=False,
    ):
        row_numbers = split_rows.index.to_numpy(dtype=np.int64)
        complete_mask = split_rows['outcome_is_complete'].astype(bool).to_numpy()
        complete_rows = row_numbers[complete_mask]
        total_count = len(split_rows)
        complete_count = len(complete_rows)
        complete_has_future = split_rows.loc[
            split_rows['outcome_is_complete'].astype(bool),
            'has_future_alarms',
        ].astype(bool).to_numpy()
        unknown_query_mask = (
            encoded.unknown_alarm_code_counts[complete_rows] > 0
        )

        records.append(
            {
                'split': str(split),
                'episode_count': total_count,
                'complete_outcome_count': complete_count,
                'outcome_coverage': complete_count / total_count,
                'incomplete_outcome_count': total_count - complete_count,
                'complete_with_future_alarm_count': int(
                    complete_has_future.sum()
                ),
                'complete_empty_outcome_count': int(
                    (~complete_has_future).sum()
                ),
                'complete_empty_outcome_share': (
                    float((~complete_has_future).mean())
                    if complete_count
                    else float('nan')
                ),
                'mean_known_label_count': (
                    float(known_label_counts[complete_rows].mean())
                    if complete_count
                    else float('nan')
                ),
                'unknown_label_query_count': int(
                    unknown_query_mask.sum()
                ),
                'unknown_label_query_share': (
                    float(unknown_query_mask.mean())
                    if complete_count
                    else float('nan')
                ),
                'unknown_alarm_code_occurrence_count': int(
                    encoded.unknown_alarm_code_counts[complete_rows].sum()
                ),
                'unknown_alarm_event_count': int(
                    encoded.unknown_alarm_event_counts[complete_rows].sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def build_forecast_alarm_support_profile(
    vocabulary: ForecastLabelVocabulary,
) -> pd.DataFrame:
    '''Rank train labels by complete-sample support.'''
    profile = pd.DataFrame(
        {
            'alarm_code': vocabulary.alarm_codes,
            'train_sample_support': vocabulary.support,
        }
    )
    profile['train_sample_share'] = (
        profile['train_sample_support'] / vocabulary.train_sample_count
    )
    profile = profile.sort_values(
        ['train_sample_support', 'alarm_code'],
        ascending=[False, True],
        kind='stable',
    ).reset_index(drop=True)
    profile.insert(0, 'support_rank', np.arange(1, len(profile) + 1))
    return profile


def assign_forecast_support_groups(
    support_profile: pd.DataFrame,
    rare_max_train_support: int,
    common_min_train_support: int,
) -> pd.DataFrame:
    '''Assign train-only rare, medium, and common label groups.'''
    if rare_max_train_support < 1:
        raise ValueError('rare support boundary must be positive')
    if common_min_train_support <= rare_max_train_support + 1:
        raise ValueError(
            'common support boundary must leave a medium support range'
        )
    if 'train_sample_support' not in support_profile:
        raise ValueError('support profile is missing train_sample_support')

    grouped = support_profile.copy()
    support = grouped['train_sample_support'].to_numpy(dtype=np.int64)
    grouped['support_group'] = np.select(
        [
            support <= rare_max_train_support,
            support >= common_min_train_support,
        ],
        ['rare', 'common'],
        default='medium',
    )
    return grouped


def summarize_forecast_support_groups(
    grouped_support_profile: pd.DataFrame,
) -> pd.DataFrame:
    '''Summarize label count and positive mass for each support group.'''
    required_columns = {'support_group', 'train_sample_support'}
    if not required_columns.issubset(grouped_support_profile.columns):
        raise ValueError('grouped support profile is missing required columns')
    if grouped_support_profile.empty:
        raise ValueError('grouped support profile cannot be empty')

    group_order = ['rare', 'medium', 'common']
    summary = (
        grouped_support_profile.groupby(
            'support_group',
            observed=True,
            sort=False,
        )['train_sample_support']
        .agg(label_count='size', positive_sample_count='sum')
        .reindex(group_order, fill_value=0)
        .reset_index()
    )
    summary['label_share'] = (
        summary['label_count'] / len(grouped_support_profile)
    )
    total_positive_count = grouped_support_profile[
        'train_sample_support'
    ].sum()
    summary['positive_sample_share'] = (
        summary['positive_sample_count'] / total_positive_count
    )
    return summary
