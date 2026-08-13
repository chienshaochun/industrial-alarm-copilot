'''Train-fitted ordered alarm sequences for neural forecasting.'''

from dataclasses import dataclass

import numpy as np
import pandas as pd


SEQUENCE_INPUT_VERSION = 'alarm_machine_sequence_v1'
PAD_TOKEN_ID = 0
UNKNOWN_TOKEN_ID = 1
UNKNOWN_MACHINE_ID = 0


@dataclass(frozen=True)
class ForecastSequenceEncoder:
    '''Train-only alarm and machine vocabularies plus truncation contract.'''

    alarm_tokens: tuple[str, ...]
    machine_labels: tuple[str, ...]
    max_sequence_length: int
    version: str = SEQUENCE_INPUT_VERSION


@dataclass(frozen=True)
class EncodedForecastSequences:
    '''Right-padded, recent-history alarm sequences aligned with incidents.'''

    incident_ids: tuple[str, ...]
    token_ids: np.ndarray
    sequence_lengths: np.ndarray
    machine_ids: np.ndarray
    encoder: ForecastSequenceEncoder


def _align_documents(
    documents: pd.DataFrame,
    incidents: pd.DataFrame,
) -> pd.DataFrame:
    if not documents['incident_id'].is_unique:
        raise ValueError('sequence documents incident_id must be unique')
    if not incidents['incident_id'].is_unique:
        raise ValueError('sequence incidents incident_id must be unique')
    incident_ids = incidents['incident_id'].astype(str)
    indexed = documents.assign(
        incident_id=documents['incident_id'].astype(str)
    ).set_index('incident_id')
    if set(indexed.index) != set(incident_ids):
        raise ValueError('sequence documents and incidents must cover same IDs')
    return indexed.loc[incident_ids].reset_index()


def fit_forecast_sequence_encoder(
    documents: pd.DataFrame,
    incidents: pd.DataFrame,
    max_sequence_length: int,
) -> ForecastSequenceEncoder:
    '''Fit alarm and machine input vocabularies using train episodes only.'''
    if max_sequence_length < 1:
        raise ValueError('max_sequence_length must be positive')
    aligned_documents = _align_documents(documents, incidents)
    train_mask = incidents['split'].eq('train').to_numpy()
    if not train_mask.any():
        raise ValueError('at least one train episode is required')
    train_documents = aligned_documents.loc[train_mask, 'alarm_document']
    alarm_tokens = tuple(
        sorted(
            {
                token
                for document in train_documents.astype(str)
                for token in document.split()
            }
        )
    )
    if not alarm_tokens:
        raise ValueError('train alarm sequence vocabulary cannot be empty')
    machine_labels = tuple(
        sorted(set(incidents.loc[train_mask, 'machine_id'].astype(str)))
    )
    return ForecastSequenceEncoder(
        alarm_tokens=alarm_tokens,
        machine_labels=machine_labels,
        max_sequence_length=int(max_sequence_length),
    )


def transform_forecast_sequences(
    encoder: ForecastSequenceEncoder,
    documents: pd.DataFrame,
    incidents: pd.DataFrame,
) -> EncodedForecastSequences:
    '''Encode ordered alarms, retaining the most recent tokens when truncated.'''
    aligned_documents = _align_documents(documents, incidents)
    alarm_id_by_token = {
        token: index + 2 for index, token in enumerate(encoder.alarm_tokens)
    }
    machine_id_by_label = {
        label: index + 1 for index, label in enumerate(encoder.machine_labels)
    }
    row_count = len(incidents)
    token_ids = np.full(
        (row_count, encoder.max_sequence_length),
        PAD_TOKEN_ID,
        dtype=np.int64,
    )
    sequence_lengths = np.zeros(row_count, dtype=np.int64)
    for row_number, document in enumerate(
        aligned_documents['alarm_document'].astype(str)
    ):
        tokens = document.split()[-encoder.max_sequence_length :]
        if not tokens:
            raise ValueError('alarm sequences cannot be empty')
        encoded_tokens = [
            alarm_id_by_token.get(token, UNKNOWN_TOKEN_ID) for token in tokens
        ]
        sequence_lengths[row_number] = len(encoded_tokens)
        token_ids[row_number, : len(encoded_tokens)] = encoded_tokens
    machine_ids = np.asarray(
        [
            machine_id_by_label.get(str(machine_id), UNKNOWN_MACHINE_ID)
            for machine_id in incidents['machine_id']
        ],
        dtype=np.int64,
    )
    return EncodedForecastSequences(
        incident_ids=tuple(incidents['incident_id'].astype(str)),
        token_ids=token_ids,
        sequence_lengths=sequence_lengths,
        machine_ids=machine_ids,
        encoder=encoder,
    )
