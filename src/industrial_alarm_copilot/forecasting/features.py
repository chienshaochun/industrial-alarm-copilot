'''Train-fitted episode features for future alarm forecasting.'''

from dataclasses import dataclass

import pandas as pd
from scipy import sparse

from industrial_alarm_copilot.retrieval.features import (
    build_retrieval_feature_variants,
)


FORECAST_ALARM_SHAPE_FEATURE_VERSION = 'forecast_alarm_shape_v1'


@dataclass(frozen=True)
class ForecastFeatureMatrix:
    '''Sparse episode features aligned with stable incident IDs.'''

    incident_ids: tuple[str, ...]
    matrix: sparse.csr_matrix
    feature_names: tuple[str, ...]
    feature_version: str


def build_forecast_alarm_shape_features(
    documents: pd.DataFrame,
    incidents: pd.DataFrame,
) -> ForecastFeatureMatrix:
    '''Build train-fitted alarm TF-IDF and episode-shape features.'''
    if not documents['incident_id'].is_unique:
        raise ValueError('forecast documents incident_id must be unique')
    if not incidents['incident_id'].is_unique:
        raise ValueError('forecast incidents incident_id must be unique')

    incident_ids = tuple(incidents['incident_id'].astype(str))
    document_ids = set(documents['incident_id'].astype(str))
    if document_ids != set(incident_ids):
        raise ValueError('forecast documents and incidents must cover same IDs')

    aligned_documents = (
        documents.assign(incident_id=documents['incident_id'].astype(str))
        .set_index('incident_id')
        .loc[list(incident_ids)]
        .reset_index()
    )
    variants = build_retrieval_feature_variants(
        aligned_documents,
        incidents,
        alarm_weight=1.0,
        shape_weight=1.0,
    )
    combined = variants['alarm_plus_shape_v1']
    if tuple(combined.incident_ids) != incident_ids:
        raise ValueError('forecast features must align with incident rows')
    return ForecastFeatureMatrix(
        incident_ids=combined.incident_ids,
        matrix=sparse.csr_matrix(combined.matrix),
        feature_names=combined.feature_names,
        feature_version=FORECAST_ALARM_SHAPE_FEATURE_VERSION,
    )
