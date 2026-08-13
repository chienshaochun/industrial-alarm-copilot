'''Alarm-plus-shape feature combination tests.'''

import numpy as np
import pandas as pd

from industrial_alarm_copilot.retrieval.features import (
    ALARM_PLUS_SHAPE_FEATURE_VERSION,
    ALARM_TFIDF_FEATURE_VERSION,
    build_retrieval_feature_variants,
    combine_alarm_and_shape_features,
    fit_alarm_tfidf,
    fit_episode_shape,
    transform_alarm_documents,
    transform_episode_shape,
)


def test_combine_alarm_and_shape_features_aligns_incident_ids():
    incidents = pd.DataFrame(
        {
            'incident_id': ['inc_a', 'inc_b'],
            'split': ['train', 'train'],
            'event_count': [1, 9],
            'duration_seconds': [0, 99],
            'distinct_alarm_count': [1, 3],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': ['inc_a', 'inc_b'],
            'alarm_document': ['1', '98'],
        }
    )
    fitted_alarm = fit_alarm_tfidf(documents, incidents)
    alarm_features = transform_alarm_documents(
        fitted_alarm,
        documents,
    )
    fitted_shape = fit_episode_shape(incidents)
    shape_features = transform_episode_shape(
        fitted_shape,
        incidents.iloc[::-1].reset_index(drop=True),
    )

    combined = combine_alarm_and_shape_features(
        alarm_features,
        shape_features,
        alarm_weight=1.0,
        shape_weight=1.0,
    )

    assert combined.incident_ids == ('inc_a', 'inc_b')
    assert combined.feature_version == ALARM_PLUS_SHAPE_FEATURE_VERSION
    assert combined.feature_parameters == (
        ('alarm_weight', 1.0),
        ('shape_weight', 1.0),
    )
    assert combined.matrix.shape == (2, 5)
    np.testing.assert_allclose(
        np.linalg.norm(combined.matrix.toarray(), axis=1),
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        combined.matrix.toarray()[:, -3:],
        [
            [-1 / np.sqrt(6)] * 3,
            [1 / np.sqrt(6)] * 3,
        ],
    )


def test_build_retrieval_feature_variants_uses_same_incident_population():
    incidents = pd.DataFrame(
        {
            'incident_id': ['train_a', 'train_b', 'validation'],
            'split': ['train', 'train', 'validation'],
            'event_count': [1, 3, 2],
            'duration_seconds': [0, 20, 10],
            'distinct_alarm_count': [1, 2, 1],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': ['validation', 'train_b', 'train_a'],
            'alarm_document': ['98', '11 98', '11'],
        }
    )

    variants = build_retrieval_feature_variants(
        documents,
        incidents,
        alarm_weight=1.0,
        shape_weight=1.0,
    )

    assert set(variants) == {
        ALARM_TFIDF_FEATURE_VERSION,
        ALARM_PLUS_SHAPE_FEATURE_VERSION,
    }
    assert variants[ALARM_TFIDF_FEATURE_VERSION].incident_ids == (
        'validation',
        'train_b',
        'train_a',
    )
    assert variants[ALARM_PLUS_SHAPE_FEATURE_VERSION].incident_ids == (
        'validation',
        'train_b',
        'train_a',
    )
