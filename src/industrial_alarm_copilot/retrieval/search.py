'''Exact, time-safe Top-K retrieval for alarm episodes.'''

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from industrial_alarm_copilot.retrieval.candidates import (
    CandidatePolicy,
    select_historical_candidates,
)
from industrial_alarm_copilot.retrieval.features import AlarmFeatureMatrix


RETRIEVAL_RESULT_COLUMNS = [
    'query_incident_id',
    'rank',
    'candidate_incident_id',
    'similarity_score',
    'feature_version',
    'candidate_policy',
    'candidate_machine_id',
    'candidate_start_time',
    'candidate_end_time',
    'same_machine',
    'shared_alarm_codes',
]


def _alarm_code_set(document: str) -> set[str]:
    return set(str(document).split())


@dataclass(frozen=True)
class RetrievalSearchIndex:
    '''Reusable feature and evidence arrays for exact repeated search.'''

    features: AlarmFeatureMatrix
    incident_ids: np.ndarray
    row_by_incident_id: dict[str, int]
    feature_rows: np.ndarray
    machine_ids: np.ndarray
    start_times: np.ndarray
    end_times: np.ndarray
    alarm_code_sets: tuple[frozenset[str], ...]

    def retrieve_candidate_rows(
        self,
        query_incident_id: str,
        candidate_rows: np.ndarray,
        top_k: int = 5,
        policy: CandidatePolicy = 'expanding_history',
    ) -> pd.DataFrame:
        '''Rank prefiltered incident rows with exact cosine similarity.'''
        if top_k <= 0:
            raise ValueError('top_k must be greater than zero')
        try:
            query_row = self.row_by_incident_id[str(query_incident_id)]
        except KeyError as error:
            raise KeyError(
                f'unknown query incident_id: {query_incident_id}'
            ) from error

        candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
        if not len(candidate_rows):
            return pd.DataFrame(columns=RETRIEVAL_RESULT_COLUMNS)
        if (
            (candidate_rows < 0).any()
            or (candidate_rows >= len(self.incident_ids)).any()
        ):
            raise IndexError('candidate row is outside incident index')

        similarities = cosine_similarity(
            self.features.matrix[self.feature_rows[query_row]],
            self.features.matrix[self.feature_rows[candidate_rows]],
        ).ravel()
        candidate_ids = self.incident_ids[candidate_rows]
        candidate_end_times = self.end_times[candidate_rows]
        order = np.lexsort(
            (
                candidate_ids,
                -candidate_end_times.astype('datetime64[ns]').astype('int64'),
                -similarities,
            )
        )[:top_k]
        ranked_rows = candidate_rows[order]
        ranked_similarities = similarities[order]
        query_alarm_codes = self.alarm_code_sets[query_row]

        results = pd.DataFrame(
            {
                'query_incident_id': str(query_incident_id),
                'rank': np.arange(1, len(ranked_rows) + 1),
                'candidate_incident_id': self.incident_ids[ranked_rows],
                'similarity_score': ranked_similarities,
                'feature_version': self.features.feature_version,
                'candidate_policy': policy,
                'candidate_machine_id': self.machine_ids[ranked_rows],
                'candidate_start_time': self.start_times[ranked_rows],
                'candidate_end_time': self.end_times[ranked_rows],
                'same_machine': (
                    self.machine_ids[ranked_rows]
                    == self.machine_ids[query_row]
                ),
                'shared_alarm_codes': [
                    tuple(
                        sorted(
                            query_alarm_codes
                            & self.alarm_code_sets[row_number]
                        )
                    )
                    for row_number in ranked_rows
                ],
            }
        )
        return results[RETRIEVAL_RESULT_COLUMNS]


def build_retrieval_search_index(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    features: AlarmFeatureMatrix,
) -> RetrievalSearchIndex:
    '''Align incident metadata, documents, and feature rows once.'''
    if not incidents['incident_id'].is_unique:
        raise ValueError('incidents incident_id must be unique')
    if not documents['incident_id'].is_unique:
        raise ValueError('documents incident_id must be unique')
    if len(set(features.incident_ids)) != len(features.incident_ids):
        raise ValueError('feature incident_ids must be unique')

    incident_ids = incidents['incident_id'].astype(str).to_numpy()
    required_ids = set(incident_ids)
    feature_row_by_id = {
        str(incident_id): row_number
        for row_number, incident_id in enumerate(features.incident_ids)
    }
    if not required_ids.issubset(feature_row_by_id):
        raise ValueError('every incident must have a feature row')

    document_table = documents.copy()
    document_table['incident_id'] = document_table['incident_id'].astype(str)
    document_table = document_table.set_index('incident_id')
    if not required_ids.issubset(document_table.index):
        raise ValueError('every incident must have an alarm document')
    alarm_documents = document_table.loc[
        incident_ids,
        'alarm_document',
    ]

    return RetrievalSearchIndex(
        features=features,
        incident_ids=incident_ids,
        row_by_incident_id={
            incident_id: row_number
            for row_number, incident_id in enumerate(incident_ids)
        },
        feature_rows=np.asarray(
            [feature_row_by_id[incident_id] for incident_id in incident_ids],
            dtype=np.int64,
        ),
        machine_ids=incidents['machine_id'].astype(str).to_numpy(),
        start_times=incidents['start_time'].to_numpy(dtype='datetime64[ns]'),
        end_times=incidents['end_time'].to_numpy(dtype='datetime64[ns]'),
        alarm_code_sets=tuple(
            frozenset(_alarm_code_set(document))
            for document in alarm_documents
        ),
    )


def retrieve_similar_episodes(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    features: AlarmFeatureMatrix,
    query_incident_id: str,
    top_k: int = 5,
    policy: CandidatePolicy = 'expanding_history',
) -> pd.DataFrame:
    '''Rank time-safe historical episodes by exact cosine similarity.'''
    search_index = build_retrieval_search_index(
        incidents,
        documents,
        features,
    )
    candidates = select_historical_candidates(
        incidents,
        query_incident_id,
        policy=policy,
    )
    candidate_rows = np.asarray(
        [
            search_index.row_by_incident_id[str(incident_id)]
            for incident_id in candidates['incident_id']
        ],
        dtype=np.int64,
    )
    return search_index.retrieve_candidate_rows(
        query_incident_id,
        candidate_rows,
        top_k=top_k,
        policy=policy,
    )
