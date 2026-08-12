'''Train-fitted feature transformations for episode retrieval.'''

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


ALARM_TFIDF_FEATURE_VERSION = 'alarm_tfidf_v1'
EPISODE_SHAPE_FEATURE_VERSION = 'episode_shape_v1'
SHAPE_SOURCE_COLUMNS = (
    'event_count',
    'duration_seconds',
    'distinct_alarm_count',
)
SHAPE_FEATURE_NAMES = tuple(
    f'shape_log1p_{column}' for column in SHAPE_SOURCE_COLUMNS
)


@dataclass(frozen=True)
class FittedAlarmTfidf:
    '''TF-IDF transformer fitted only on train episode documents.'''

    vectorizer: TfidfVectorizer
    train_incident_count: int
    feature_version: str = ALARM_TFIDF_FEATURE_VERSION


@dataclass(frozen=True)
class FittedEpisodeShape:
    '''Shape scaler fitted on log-transformed train episodes only.'''

    scaler: StandardScaler
    train_incident_count: int
    feature_version: str = EPISODE_SHAPE_FEATURE_VERSION


@dataclass(frozen=True)
class AlarmFeatureMatrix:
    '''Sparse alarm features aligned with stable incident IDs.'''

    incident_ids: tuple[str, ...]
    matrix: Any
    feature_names: tuple[str, ...]
    feature_version: str


def fit_alarm_tfidf(
    documents: pd.DataFrame,
    incidents: pd.DataFrame,
) -> FittedAlarmTfidf:
    '''Fit alarm-code vocabulary and IDF using train episodes only.'''
    if not documents['incident_id'].is_unique:
        raise ValueError('documents incident_id must be unique')
    if not incidents['incident_id'].is_unique:
        raise ValueError('incidents incident_id must be unique')

    documents_with_split = documents.merge(
        incidents[['incident_id', 'split']],
        on='incident_id',
        how='left',
        sort=False,
        validate='one_to_one',
    )
    if documents_with_split['split'].isna().any():
        raise ValueError('every document must reference an incident')

    train_documents = documents_with_split.loc[
        documents_with_split['split'].eq('train'),
        'alarm_document',
    ]
    if train_documents.empty:
        raise ValueError('at least one train alarm document is required')

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'(?u)\b\w+\b',
        norm='l2',
        use_idf=True,
        smooth_idf=True,
    )
    vectorizer.fit(train_documents)
    return FittedAlarmTfidf(
        vectorizer=vectorizer,
        train_incident_count=len(train_documents),
    )


def transform_alarm_documents(
    fitted_tfidf: FittedAlarmTfidf,
    documents: pd.DataFrame,
) -> AlarmFeatureMatrix:
    '''Transform documents without changing the train-fitted vocabulary.'''
    if not documents['incident_id'].is_unique:
        raise ValueError('documents incident_id must be unique')

    feature_matrix = fitted_tfidf.vectorizer.transform(
        documents['alarm_document']
    )
    return AlarmFeatureMatrix(
        incident_ids=tuple(documents['incident_id'].astype(str)),
        matrix=feature_matrix,
        feature_names=tuple(
            fitted_tfidf.vectorizer.get_feature_names_out()
        ),
        feature_version=fitted_tfidf.feature_version,
    )


def _log_shape_values(incidents: pd.DataFrame) -> np.ndarray:
    values = incidents.loc[
        :, list(SHAPE_SOURCE_COLUMNS)
    ].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('shape features must be finite')
    if (values < 0).any():
        raise ValueError('shape features cannot be negative')
    return np.log1p(values)


def fit_episode_shape(incidents: pd.DataFrame) -> FittedEpisodeShape:
    '''Fit log-shape standardization using train episodes only.'''
    if not incidents['incident_id'].is_unique:
        raise ValueError('incidents incident_id must be unique')

    train_mask = incidents['split'].eq('train').to_numpy()
    if not train_mask.any():
        raise ValueError('at least one train incident is required')

    log_shape_values = _log_shape_values(incidents)
    scaler = StandardScaler()
    scaler.fit(log_shape_values[train_mask])
    return FittedEpisodeShape(
        scaler=scaler,
        train_incident_count=int(train_mask.sum()),
    )


def transform_episode_shape(
    fitted_shape: FittedEpisodeShape,
    incidents: pd.DataFrame,
) -> AlarmFeatureMatrix:
    '''Transform episode shape without refitting on validation or test.'''
    if not incidents['incident_id'].is_unique:
        raise ValueError('incidents incident_id must be unique')

    shape_matrix = fitted_shape.scaler.transform(
        _log_shape_values(incidents)
    )
    return AlarmFeatureMatrix(
        incident_ids=tuple(incidents['incident_id'].astype(str)),
        matrix=shape_matrix,
        feature_names=SHAPE_FEATURE_NAMES,
        feature_version=fitted_shape.feature_version,
    )
