'''Train-fitted forecasting episode feature tests.'''

import numpy as np
import pandas as pd
import pytest

from industrial_alarm_copilot.forecasting.features import (
    build_forecast_alarm_shape_features,
)


def test_build_forecast_features_are_train_fitted_and_machine_free():
    incidents = pd.DataFrame(
        {
            'incident_id': ['train_a', 'train_b', 'validation'],
            'machine_id': ['4', '9', '4'],
            'split': ['train', 'train', 'validation'],
            'event_count': [2, 1, 3],
            'duration_seconds': [30.0, 0.0, 60.0],
            'distinct_alarm_count': [2, 1, 2],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'alarm_document': ['11 98', '26', '11 137 137'],
        }
    )

    features = build_forecast_alarm_shape_features(
        documents,
        incidents,
    )

    assert features.incident_ids == ('train_a', 'train_b', 'validation')
    assert features.feature_version == 'forecast_alarm_shape_v1'
    assert features.matrix.shape[0] == 3
    assert 'alarm_11' in features.feature_names
    assert 'alarm_26' in features.feature_names
    assert 'alarm_137' not in features.feature_names
    assert not any('machine' in name for name in features.feature_names)
    row_norms = np.sqrt(features.matrix.multiply(features.matrix).sum(axis=1))
    assert np.asarray(row_norms).ravel().tolist() == pytest.approx(
        [1.0, 1.0, 1.0]
    )
