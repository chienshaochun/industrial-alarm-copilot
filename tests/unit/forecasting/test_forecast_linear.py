'''One-vs-rest linear forecasting tests.'''

import numpy as np
import pandas as pd
from scipy import sparse

from industrial_alarm_copilot.forecasting.features import ForecastFeatureMatrix
from industrial_alarm_copilot.forecasting.linear import (
    fit_forecast_linear_model,
    score_forecast_linear_model,
)
from industrial_alarm_copilot.forecasting.vocabulary import (
    encode_forecast_labels,
    fit_forecast_label_vocabulary,
)


def _inputs():
    labels = pd.DataFrame(
        {
            'incident_id': ['a', 'b', 'c', 'd', 'query'],
            'split': ['train', 'train', 'train', 'train', 'validation'],
            'outcome_is_complete': [True] * 5,
            'future_alarm_codes': [('11',), ('11',), ('98',), ('98',), ('11',)],
            'future_alarm_counts': [
                ((code, 1),) for code in ['11', '11', '98', '98', '11']
            ],
        }
    )
    vocabulary = fit_forecast_label_vocabulary(labels)
    encoded = encode_forecast_labels(vocabulary, labels)
    features = ForecastFeatureMatrix(
        incident_ids=tuple(labels['incident_id']),
        matrix=sparse.csr_matrix(
            np.asarray(
                [[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.5, 0.0]]
            )
        ),
        feature_names=('alarm_11', 'alarm_98'),
        feature_version='forecast_alarm_shape_v1',
    )
    return features, labels, encoded


def test_forecast_linear_model_uses_train_and_scores_all_rows():
    features, labels, encoded = _inputs()

    model = fit_forecast_linear_model(features, labels, encoded, 'none')
    scores = score_forecast_linear_model(model, features)

    assert model.train_sample_count == 4
    assert model.model_version == 'ovr_logistic_none_v1'
    assert scores.scores.shape == (5, 2)
    alarm_11 = scores.alarm_codes.index('11')
    alarm_98 = scores.alarm_codes.index('98')
    assert scores.scores[-1, alarm_11] > scores.scores[-1, alarm_98]


def test_forecast_linear_balanced_mode_is_explicit_and_finite():
    features, labels, encoded = _inputs()

    model = fit_forecast_linear_model(features, labels, encoded, 'balanced')
    scores = score_forecast_linear_model(model, features)

    assert model.class_weight_mode == 'balanced'
    assert model.model_version == 'ovr_logistic_balanced_v1'
    assert np.isfinite(scores.scores).all()
