'''Train-only forecasting label vocabulary tests.'''

import pandas as pd

from industrial_alarm_copilot.forecasting.vocabulary import (
    encode_forecast_labels,
    fit_forecast_label_vocabulary,
)


def _build_labels():
    return pd.DataFrame(
        {
            'incident_id': [
                'train_complete',
                'train_empty',
                'train_incomplete',
                'validation_complete',
            ],
            'split': ['train', 'train', 'train', 'validation'],
            'outcome_is_complete': [True, True, False, True],
            'future_alarm_codes': [
                ('98', '11', '98'),
                (),
                ('137',),
                ('26', '98'),
            ],
            'future_alarm_counts': [
                (('98', 2), ('11', 1)),
                (),
                (('137', 1),),
                (('26', 3), ('98', 1)),
            ],
        }
    )


def test_fit_forecast_label_vocabulary_uses_complete_train_only():
    vocabulary = fit_forecast_label_vocabulary(_build_labels())

    assert vocabulary.alarm_codes == ('11', '98')
    assert vocabulary.support == (1, 1)
    assert vocabulary.train_sample_count == 2
    assert vocabulary.version == 'forecast_label_v1'


def test_encode_forecast_labels_preserves_unknown_diagnostics():
    labels = _build_labels()
    vocabulary = fit_forecast_label_vocabulary(labels)

    encoded = encode_forecast_labels(vocabulary, labels)

    assert encoded.incident_ids == tuple(labels['incident_id'])
    assert encoded.alarm_codes == ('11', '98')
    assert encoded.matrix.toarray().tolist() == [
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 1],
    ]
    assert encoded.unknown_alarm_codes == (
        (),
        (),
        ('137',),
        ('26',),
    )
    assert encoded.unknown_alarm_code_counts.tolist() == [0, 0, 1, 1]
    assert encoded.unknown_alarm_event_counts.tolist() == [0, 0, 1, 3]
