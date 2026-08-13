'''Forecast sequence encoding tests.'''

import pandas as pd

from industrial_alarm_copilot.forecasting.sequences import (
    UNKNOWN_MACHINE_ID,
    UNKNOWN_TOKEN_ID,
    fit_forecast_sequence_encoder,
    transform_forecast_sequences,
)


def test_sequence_encoder_is_train_fitted_and_keeps_recent_order():
    incidents = pd.DataFrame(
        {
            'incident_id': ['train', 'validation'],
            'machine_id': ['4', 'unseen'],
            'split': ['train', 'validation'],
        }
    )
    documents = pd.DataFrame(
        {
            'incident_id': ['validation', 'train'],
            'alarm_document': ['11 999', '11 26 98'],
        }
    )

    encoder = fit_forecast_sequence_encoder(documents, incidents, 2)
    encoded = transform_forecast_sequences(encoder, documents, incidents)

    assert encoder.alarm_tokens == ('11', '26', '98')
    assert encoder.machine_labels == ('4',)
    assert encoded.incident_ids == ('train', 'validation')
    assert encoded.sequence_lengths.tolist() == [2, 2]
    assert encoded.token_ids[0].tolist() == [3, 4]
    assert encoded.token_ids[1].tolist() == [2, UNKNOWN_TOKEN_ID]
    assert encoded.machine_ids.tolist() == [1, UNKNOWN_MACHINE_ID]
