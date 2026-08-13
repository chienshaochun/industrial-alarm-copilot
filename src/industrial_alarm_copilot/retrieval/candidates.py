'''Time-safe historical candidate selection for episode retrieval.'''

from typing import Literal

import pandas as pd


CandidatePolicy = Literal['expanding_history', 'train_only']
VALID_CANDIDATE_POLICIES = {'expanding_history', 'train_only'}


def select_historical_candidates(
    incidents: pd.DataFrame,
    query_incident_id: str,
    policy: CandidatePolicy = 'expanding_history',
) -> pd.DataFrame:
    '''Return episodes that had ended before the query episode began.'''
    if not incidents['incident_id'].is_unique:
        raise ValueError('incidents incident_id must be unique')
    if policy not in VALID_CANDIDATE_POLICIES:
        raise ValueError(f'unsupported candidate policy: {policy}')

    query_rows = incidents.loc[
        incidents['incident_id'].eq(query_incident_id)
    ]
    if query_rows.empty:
        raise KeyError(f'unknown query incident_id: {query_incident_id}')

    query_start_time = query_rows.iloc[0]['start_time']
    candidate_mask = (
        incidents['end_time'].lt(query_start_time)
        & incidents['incident_id'].ne(query_incident_id)
    )
    if policy == 'train_only':
        candidate_mask &= incidents['split'].eq('train')

    return incidents.loc[candidate_mask].copy()
