'''Adapt time-safe retrieval outputs into presentation evidence.'''

import pandas as pd

from industrial_alarm_copilot.application.contracts import (
    RetrievedEpisodeEvidence,
)


def build_retrieved_episode_evidence(
    retrieval_results: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[RetrievedEpisodeEvidence, ...]:
    '''Align ranked historical episodes with their observed future alarms.'''
    ranked = retrieval_results.copy()
    ranked['candidate_incident_id'] = ranked[
        'candidate_incident_id'
    ].astype(str)
    outcome_table = outcomes[
        [
            'incident_id',
            'outcome_is_complete',
            'future_horizon_hours',
            'future_alarm_codes',
        ]
    ].copy()
    outcome_table['incident_id'] = outcome_table['incident_id'].astype(str)
    aligned = ranked.merge(
        outcome_table,
        left_on='candidate_incident_id',
        right_on='incident_id',
        how='left',
        sort=False,
        validate='one_to_one',
        indicator=True,
    )
    if not aligned['_merge'].eq('both').all():
        missing_ids = aligned.loc[
            aligned['_merge'].ne('both'),
            'candidate_incident_id',
        ]
        raise ValueError(
            'retrieved incident is missing a future outcome: '
            + ', '.join(missing_ids)
        )
    aligned = aligned.sort_values('rank', kind='stable')
    actual_ranks = aligned['rank'].tolist()
    expected_ranks = list(range(1, len(aligned) + 1))
    if actual_ranks != expected_ranks:
        raise ValueError('rank must be unique and contiguous from one')

    return tuple(
        RetrievedEpisodeEvidence(
            rank=int(row.rank),
            incident_id=str(row.candidate_incident_id),
            machine_id=str(row.candidate_machine_id),
            start_time=row.candidate_start_time.to_pydatetime(),
            end_time=row.candidate_end_time.to_pydatetime(),
            similarity_score=float(row.similarity_score),
            shared_alarm_codes=tuple(
                str(code) for code in row.shared_alarm_codes
            ),
            future_alarm_codes=tuple(
                str(code) for code in row.future_alarm_codes
            ),
            outcome_is_complete=bool(row.outcome_is_complete),
            future_horizon_hours=float(row.future_horizon_hours),
        )
        for row in aligned.itertuples(index=False)
    )
