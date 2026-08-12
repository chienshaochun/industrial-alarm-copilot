'''Train-fitted feature transformations for episode retrieval.'''

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ALARM_TFIDF_FEATURE_VERSION = 'alarm_tfidf_v1'


@dataclass(frozen=True)
class FittedAlarmTfidf:
    '''TF-IDF transformer fitted only on train episode documents.'''

    vectorizer: TfidfVectorizer
    train_incident_count: int
    feature_version: str = ALARM_TFIDF_FEATURE_VERSION


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
