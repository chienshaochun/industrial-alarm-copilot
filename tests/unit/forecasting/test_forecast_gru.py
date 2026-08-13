'''Lightweight GRU fit and scoring tests.'''

import numpy as np
import pandas as pd

from industrial_alarm_copilot.forecasting.gru import (
    fit_forecast_gru,
    score_forecast_gru,
)
from industrial_alarm_copilot.forecasting.sequences import (
    fit_forecast_sequence_encoder,
    transform_forecast_sequences,
)
from industrial_alarm_copilot.forecasting.vocabulary import (
    encode_forecast_labels,
    fit_forecast_label_vocabulary,
)


def test_gru_fits_complete_train_and_scores_all_rows():
    incidents = pd.DataFrame(
        {
            'incident_id': ['a', 'b', 'c', 'd', 'query'],
            'machine_id': ['4', '4', '9', '9', '4'],
            'split': ['train', 'train', 'train', 'train', 'validation'],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': incidents['incident_id'],
            'alarm_document': ['11 11', '11 26', '98 98', '98 26', '11 11'],
        }
    )
    labels = incidents.copy()
    labels['outcome_is_complete'] = True
    labels['future_alarm_codes'] = [('11',), ('11',), ('98',), ('98',), ('11',)]
    labels['future_alarm_counts'] = [
        ((code, 1),) for code in ['11', '11', '98', '98', '11']
    ]
    vocabulary = fit_forecast_label_vocabulary(labels)
    targets = encode_forecast_labels(vocabulary, labels)
    encoder = fit_forecast_sequence_encoder(documents, incidents, 4)
    sequences = transform_forecast_sequences(encoder, documents, incidents)

    model = fit_forecast_gru(
        sequences,
        labels,
        targets,
        embedding_dim=4,
        hidden_dim=6,
        machine_embedding_dim=2,
        batch_size=2,
        epochs=2,
        weight_mode='balanced_capped',
    )
    scores = score_forecast_gru(model, sequences, batch_size=2)

    assert model.train_sample_count == 4
    assert len(model.loss_history) == 2
    assert model.model_version == 'gru_balanced_capped_v1'
    assert scores.scores.shape == (5, 2)
    assert np.isfinite(scores.scores).all()
    assert ((scores.scores >= 0) & (scores.scores <= 1)).all()
