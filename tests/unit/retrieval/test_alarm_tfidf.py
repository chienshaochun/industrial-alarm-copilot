'''Train-only alarm TF-IDF feature unit tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.retrieval.features import (
    fit_alarm_tfidf,
    transform_alarm_documents,
)


def test_alarm_tfidf_fits_train_only_and_keeps_single_digit_codes():
    documents = pd.DataFrame(
        {
            'incident_id': pd.Series(
                ['inc_train_a', 'inc_validation', 'inc_train_b'],
                dtype='string',
            ),
            'alarm_document': pd.Series(
                ['1 98 98', '999 999', '1 26'],
                dtype='string',
            ),
        }
    )
    incidents = pd.DataFrame(
        {
            'incident_id': pd.Series(
                ['inc_train_a', 'inc_train_b', 'inc_validation'],
                dtype='string',
            ),
            'split': ['train', 'train', 'validation'],
        }
    )

    fitted = fit_alarm_tfidf(documents, incidents)
    features = transform_alarm_documents(fitted, documents)

    assert fitted.train_incident_count == 2
    assert features.feature_version == 'alarm_tfidf_v1'
    assert features.feature_names == ('1', '26', '98')
    assert '999' not in fitted.vectorizer.vocabulary_
    assert features.incident_ids == (
        'inc_train_a',
        'inc_validation',
        'inc_train_b',
    )
    assert features.matrix.shape == (3, 3)
    assert features.matrix[1].nnz == 0

    token_1 = fitted.vectorizer.vocabulary_['1']
    token_98 = fitted.vectorizer.vocabulary_['98']
    assert features.matrix[0, token_98] > features.matrix[0, token_1]
    assert fitted.vectorizer.idf_[token_1] == pytest.approx(1.0)
