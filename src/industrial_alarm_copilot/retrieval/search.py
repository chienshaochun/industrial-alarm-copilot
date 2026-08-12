'''Exact, time-safe Top-K retrieval for alarm episodes.'''

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


def retrieve_similar_episodes(
    incidents: pd.DataFrame,
    documents: pd.DataFrame,
    features: AlarmFeatureMatrix,
    query_incident_id: str,
    top_k: int = 5,
    policy: CandidatePolicy = 'expanding_history',
) -> pd.DataFrame:
    '''Rank time-safe historical episodes by exact cosine similarity.'''
    if top_k <= 0:
        raise ValueError('top_k must be greater than zero')
    if len(set(features.incident_ids)) != len(features.incident_ids):
        raise ValueError('feature incident_ids must be unique')
    if not documents['incident_id'].is_unique:
        raise ValueError('documents incident_id must be unique')

    row_by_incident_id = {
        incident_id: row_number
        for row_number, incident_id in enumerate(features.incident_ids)
    }
    required_ids = set(incidents['incident_id'].astype(str))
    missing_feature_ids = required_ids.difference(row_by_incident_id)
    if missing_feature_ids:
        raise ValueError('every incident must have a feature row')

    document_by_incident_id = documents.set_index('incident_id')[
        'alarm_document'
    ]
    if not required_ids.issubset(set(document_by_incident_id.index.astype(str))):
        raise ValueError('every incident must have an alarm document')

    candidates = select_historical_candidates(
        incidents,
        query_incident_id,
        policy=policy,
    )
    if candidates.empty:
        return pd.DataFrame(columns=RETRIEVAL_RESULT_COLUMNS)

    query = incidents.loc[
        incidents['incident_id'].eq(query_incident_id)
    ].iloc[0]
    query_row = row_by_incident_id[str(query_incident_id)]
    candidate_rows = [
        row_by_incident_id[str(incident_id)]
        for incident_id in candidates['incident_id']
    ]
    similarities = cosine_similarity(
        features.matrix[query_row],
        features.matrix[candidate_rows],
    ).ravel()

    ranked = candidates[
        ['incident_id', 'machine_id', 'start_time', 'end_time']
    ].copy()
    ranked['similarity_score'] = similarities
    ranked = ranked.sort_values(
        ['similarity_score', 'end_time', 'incident_id'],
        ascending=[False, False, True],
        kind='stable',
    ).head(top_k).reset_index(drop=True)

    query_alarm_codes = _alarm_code_set(
        document_by_incident_id.loc[query_incident_id]
    )
    results = pd.DataFrame(
        {
            'query_incident_id': str(query_incident_id),
            'rank': range(1, len(ranked) + 1),
            'candidate_incident_id': ranked[
                'incident_id'
            ].astype(str).to_numpy(),
            'similarity_score': ranked['similarity_score'].to_numpy(),
            'feature_version': features.feature_version,
            'candidate_policy': policy,
            'candidate_machine_id': ranked[
                'machine_id'
            ].astype(str).to_numpy(),
            'candidate_start_time': ranked['start_time'].to_numpy(),
            'candidate_end_time': ranked['end_time'].to_numpy(),
            'same_machine': ranked['machine_id'].astype(str).eq(
                str(query['machine_id'])
            ).to_numpy(),
            'shared_alarm_codes': [
                tuple(
                    sorted(
                        query_alarm_codes
                        & _alarm_code_set(
                            document_by_incident_id.loc[incident_id]
                        )
                    )
                )
                for incident_id in ranked['incident_id']
            ],
        }
    )
    return results[RETRIEVAL_RESULT_COLUMNS]
