'''Train-fitted feature transformations for episode retrieval.'''

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize


ALARM_TFIDF_FEATURE_VERSION = 'alarm_tfidf_v1'
EPISODE_SHAPE_FEATURE_VERSION = 'episode_shape_v1'
ALARM_PLUS_SHAPE_FEATURE_VERSION = 'alarm_plus_shape_v1'
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
    feature_parameters: tuple[tuple[str, float], ...] = ()


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


def combine_alarm_and_shape_features(
    alarm_features: AlarmFeatureMatrix,
    shape_features: AlarmFeatureMatrix,
    alarm_weight: float = 1.0,
    shape_weight: float = 1.0,
) -> AlarmFeatureMatrix:
    '''Align, weight, and combine alarm composition with episode shape.'''
    weights = np.asarray([alarm_weight, shape_weight], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('feature weights must be finite and nonnegative')
    if not weights.any():
        raise ValueError('at least one feature weight must be positive')
    if len(set(alarm_features.incident_ids)) != len(
        alarm_features.incident_ids
    ):
        raise ValueError('alarm feature incident_ids must be unique')
    if len(set(shape_features.incident_ids)) != len(
        shape_features.incident_ids
    ):
        raise ValueError('shape feature incident_ids must be unique')
    if set(alarm_features.incident_ids) != set(shape_features.incident_ids):
        raise ValueError('alarm and shape features must cover same incidents')

    shape_row_by_id = {
        incident_id: row_number
        for row_number, incident_id in enumerate(shape_features.incident_ids)
    }
    aligned_shape_rows = [
        shape_row_by_id[incident_id]
        for incident_id in alarm_features.incident_ids
    ]

    alarm_block = normalize(
        alarm_features.matrix,
        norm='l2',
        copy=True,
    )
    shape_block = normalize(
        shape_features.matrix[aligned_shape_rows],
        norm='l2',
        copy=True,
    )
    combined_matrix = sparse.hstack(
        [
            sparse.csr_matrix(alarm_block) * float(alarm_weight),
            sparse.csr_matrix(shape_block) * float(shape_weight),
        ],
        format='csr',
    )
    combined_matrix = normalize(combined_matrix, norm='l2', copy=False)

    return AlarmFeatureMatrix(
        incident_ids=alarm_features.incident_ids,
        matrix=combined_matrix,
        feature_names=(
            tuple(f'alarm_{name}' for name in alarm_features.feature_names)
            + shape_features.feature_names
        ),
        feature_version=ALARM_PLUS_SHAPE_FEATURE_VERSION,
        feature_parameters=(
            ('alarm_weight', float(alarm_weight)),
            ('shape_weight', float(shape_weight)),
        ),
    )


def build_retrieval_feature_variants(
    documents: pd.DataFrame,
    incidents: pd.DataFrame,
    alarm_weight: float = 1.0,
    shape_weight: float = 1.0,
) -> dict[str, AlarmFeatureMatrix]:
    '''Build comparable train-fitted alarm-only and alarm-plus-shape variants.'''
    document_ids = set(documents['incident_id'].astype(str))
    incident_ids = set(incidents['incident_id'].astype(str))
    if document_ids != incident_ids:
        raise ValueError('documents and incidents must cover same IDs')

    fitted_alarm = fit_alarm_tfidf(documents, incidents)
    alarm_features = transform_alarm_documents(
        fitted_alarm,
        documents,
    )
    fitted_shape = fit_episode_shape(incidents)
    shape_features = transform_episode_shape(fitted_shape, incidents)
    combined_features = combine_alarm_and_shape_features(
        alarm_features,
        shape_features,
        alarm_weight=alarm_weight,
        shape_weight=shape_weight,
    )
    return {
        alarm_features.feature_version: alarm_features,
        combined_features.feature_version: combined_features,
    }
