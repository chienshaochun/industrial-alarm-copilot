'''Train-only episode shape feature tests.'''

import numpy as np
import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.features import (
    SHAPE_FEATURE_NAMES,
    fit_episode_shape,
    transform_episode_shape,
)


def test_episode_shape_scaler_fits_train_only_after_log1p():
    incidents = pd.DataFrame(
        {
            'incident_id': ['train_small', 'train_large', 'validation_huge'],
            'split': ['train', 'train', 'validation'],
            'event_count': [1, 9, 999999],
            'duration_seconds': [0, 99, 999999],
            'distinct_alarm_count': [1, 3, 999],
        }
    )

    fitted = fit_episode_shape(incidents)
    transformed = transform_episode_shape(fitted, incidents)

    assert fitted.train_incident_count == 2
    assert transformed.feature_names == SHAPE_FEATURE_NAMES
    assert transformed.incident_ids == (
        'train_small',
        'train_large',
        'validation_huge',
    )
    np.testing.assert_allclose(
        transformed.matrix[:2],
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    )
    assert (transformed.matrix[2] > 1).all()


def test_episode_shape_rejects_negative_physical_values():
    incidents = pd.DataFrame(
        {
            'incident_id': ['invalid'],
            'split': ['train'],
            'event_count': [1],
            'duration_seconds': [-1],
            'distinct_alarm_count': [1],
        }
    )

    with pytest.raises(ValueError, match='cannot be negative'):
        fit_episode_shape(incidents)
