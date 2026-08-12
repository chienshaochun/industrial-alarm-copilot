'''Time-safe candidate preparation for offline retrieval evaluation.'''

import pandas as pd

from industrial_alarm_copilot.retrieval.candidates import (
    CandidatePolicy,
    select_historical_candidates,
)


def select_outcome_evaluable_candidates(
    incidents: pd.DataFrame,
    outcomes: pd.DataFrame,
    query_incident_id: str,
    policy: CandidatePolicy = 'expanding_history',
) -> pd.DataFrame:
    '''Return historical candidates whose complete outcomes predate query.'''
    if not outcomes['incident_id'].is_unique:
        raise ValueError('outcomes incident_id must be unique')

    candidates = select_historical_candidates(
        incidents,
        query_incident_id,
        policy=policy,
    )
    query_rows = incidents.loc[
        incidents['incident_id'].eq(query_incident_id)
    ]
    query_start_time = query_rows.iloc[0]['start_time']

    outcome_columns = outcomes[
        ['incident_id', 'outcome_end_time', 'outcome_is_complete']
    ].copy()
    candidates_with_outcomes = candidates.merge(
        outcome_columns,
        on='incident_id',
        how='left',
        sort=False,
        validate='one_to_one',
    )
    if candidates_with_outcomes['outcome_is_complete'].isna().any():
        raise ValueError('every candidate must have an outcome')

    evaluation_mask = (
        candidates_with_outcomes['outcome_is_complete'].astype(bool)
        & candidates_with_outcomes['outcome_end_time'].lt(query_start_time)
    )
    return candidates_with_outcomes.loc[
        evaluation_mask,
        candidates.columns,
    ].copy()
