'''Train-fitted alarm label vocabulary and binary target encoding.'''

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


LABEL_VOCABULARY_VERSION = 'forecast_label_v1'


@dataclass(frozen=True)
class ForecastLabelVocabulary:
    '''Alarm output space fitted from complete train outcomes only.'''

    alarm_codes: tuple[str, ...]
    support: tuple[int, ...]
    train_sample_count: int
    version: str = LABEL_VOCABULARY_VERSION


@dataclass(frozen=True)
class EncodedForecastLabels:
    '''Sparse known-label targets and explicit unknown-label diagnostics.'''

    incident_ids: tuple[str, ...]
    alarm_codes: tuple[str, ...]
    matrix: sparse.csr_matrix
    unknown_alarm_codes: tuple[tuple[str, ...], ...]
    unknown_alarm_code_counts: np.ndarray
    unknown_alarm_event_counts: np.ndarray


def _validate_label_table(labels: pd.DataFrame) -> None:
    required_columns = {
        'incident_id',
        'split',
        'outcome_is_complete',
        'future_alarm_codes',
        'future_alarm_counts',
    }
    missing_columns = required_columns.difference(labels.columns)
    if missing_columns:
        raise ValueError(
            'forecast labels missing columns: '
            + ', '.join(sorted(missing_columns))
        )
    if not labels['incident_id'].is_unique:
        raise ValueError('forecast labels incident_id must be unique')


def fit_forecast_label_vocabulary(
    labels: pd.DataFrame,
) -> ForecastLabelVocabulary:
    '''Fit a deterministic output vocabulary on complete train samples.'''
    _validate_label_table(labels)
    train_labels = labels.loc[
        labels['split'].eq('train')
        & labels['outcome_is_complete'].astype(bool)
    ]
    if train_labels.empty:
        raise ValueError('at least one complete train label is required')

    normalized_sets = [
        set(str(code) for code in alarm_codes)
        for alarm_codes in train_labels['future_alarm_codes']
    ]
    alarm_codes = tuple(
        sorted({code for codes in normalized_sets for code in codes})
    )
    if not alarm_codes:
        raise ValueError('complete train labels contain no alarm codes')
    support = tuple(
        sum(alarm_code in codes for codes in normalized_sets)
        for alarm_code in alarm_codes
    )
    return ForecastLabelVocabulary(
        alarm_codes=alarm_codes,
        support=support,
        train_sample_count=len(train_labels),
    )


def encode_forecast_labels(
    vocabulary: ForecastLabelVocabulary,
    labels: pd.DataFrame,
) -> EncodedForecastLabels:
    '''Encode known targets without expanding the train-fitted vocabulary.'''
    _validate_label_table(labels)
    column_by_alarm_code = {
        alarm_code: column_number
        for column_number, alarm_code in enumerate(vocabulary.alarm_codes)
    }
    known_alarm_codes = set(vocabulary.alarm_codes)
    row_numbers = []
    column_numbers = []
    unknown_alarm_codes = []
    unknown_alarm_code_counts = np.zeros(len(labels), dtype=np.int64)
    unknown_alarm_event_counts = np.zeros(len(labels), dtype=np.int64)

    for row_number, row in enumerate(labels.itertuples(index=False)):
        normalized_codes = set(
            str(code) for code in row.future_alarm_codes
        )
        known_codes = sorted(normalized_codes.intersection(known_alarm_codes))
        row_numbers.extend([row_number] * len(known_codes))
        column_numbers.extend(
            column_by_alarm_code[code] for code in known_codes
        )

        row_unknown_codes = tuple(
            sorted(normalized_codes.difference(known_alarm_codes))
        )
        unknown_alarm_codes.append(row_unknown_codes)
        unknown_alarm_code_counts[row_number] = len(row_unknown_codes)
        counts_by_code = {
            str(code): int(count)
            for code, count in row.future_alarm_counts
        }
        unknown_alarm_event_counts[row_number] = sum(
            counts_by_code.get(code, 0) for code in row_unknown_codes
        )

    matrix = sparse.csr_matrix(
        (
            np.ones(len(row_numbers), dtype=np.int8),
            (row_numbers, column_numbers),
        ),
        shape=(len(labels), len(vocabulary.alarm_codes)),
        dtype=np.int8,
    )
    return EncodedForecastLabels(
        incident_ids=tuple(labels['incident_id'].astype(str)),
        alarm_codes=vocabulary.alarm_codes,
        matrix=matrix,
        unknown_alarm_codes=tuple(unknown_alarm_codes),
        unknown_alarm_code_counts=unknown_alarm_code_counts,
        unknown_alarm_event_counts=unknown_alarm_event_counts,
    )
