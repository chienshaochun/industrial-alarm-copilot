'''Retrieved historical evidence adapter tests.'''

import pandas as pd
import pytest

from industrial_alarm_copilot.application.retrieved import (
    build_retrieved_episode_evidence,
)


def test_build_retrieved_evidence_aligns_outcomes_and_restores_rank_order():
    retrieval_results = pd.DataFrame(
        {
            'rank': [2, 1],
            'candidate_incident_id': ['inc_b', 'inc_a'],
            'candidate_machine_id': ['9', '4'],
            'candidate_start_time': pd.to_datetime(
                ['2019-11-04 08:00:00', '2019-11-03 14:21:00']
            ),
            'candidate_end_time': pd.to_datetime(
                ['2019-11-04 08:15:00', '2019-11-03 15:08:00']
            ),
            'similarity_score': [0.61, 0.73],
            'shared_alarm_codes': [('98',), ('11', '98')],
        }
    )
    outcomes = pd.DataFrame(
        {
            'incident_id': ['inc_a', 'inc_b'],
            'outcome_is_complete': [True, False],
            'future_horizon_hours': [6.0, 6.0],
            'future_alarm_codes': [('26', '98'), ()],
        }
    )

    evidence = build_retrieved_episode_evidence(
        retrieval_results,
        outcomes,
    )

    assert [item.rank for item in evidence] == [1, 2]
    assert [item.incident_id for item in evidence] == ['inc_a', 'inc_b']
    assert evidence[0].future_alarm_codes == ('26', '98')
    assert evidence[0].outcome_is_complete is True
    assert evidence[0].future_horizon_hours == 6.0
    assert evidence[1].future_alarm_codes == ()
    assert evidence[1].outcome_is_complete is False


def _retrieval_inputs():
    retrieval_results = pd.DataFrame(
        {
            'rank': [1, 2],
            'candidate_incident_id': ['inc_a', 'inc_b'],
            'candidate_machine_id': ['4', '9'],
            'candidate_start_time': pd.to_datetime(
                ['2019-11-03 14:21:00', '2019-11-04 08:00:00']
            ),
            'candidate_end_time': pd.to_datetime(
                ['2019-11-03 15:08:00', '2019-11-04 08:15:00']
            ),
            'similarity_score': [0.73, 0.61],
            'shared_alarm_codes': [('11', '98'), ('98',)],
        }
    )
    outcomes = pd.DataFrame(
        {
            'incident_id': ['inc_a', 'inc_b'],
            'outcome_is_complete': [True, True],
            'future_horizon_hours': [6.0, 6.0],
            'future_alarm_codes': [('26',), ('11',)],
        }
    )
    return retrieval_results, outcomes


def test_build_retrieved_evidence_rejects_missing_outcome():
    retrieval_results, outcomes = _retrieval_inputs()
    outcomes = outcomes.loc[outcomes['incident_id'].ne('inc_b')]

    with pytest.raises(ValueError, match='inc_b'):
        build_retrieved_episode_evidence(retrieval_results, outcomes)


@pytest.mark.parametrize(
    'invalid_ranks',
    [
        [1, 1],
        [1, 3],
    ],
    ids=['duplicate', 'noncontiguous'],
)
def test_build_retrieved_evidence_rejects_invalid_rank_sequence(
    invalid_ranks,
):
    retrieval_results, outcomes = _retrieval_inputs()
    retrieval_results['rank'] = invalid_ranks

    with pytest.raises(ValueError, match='rank'):
        build_retrieved_episode_evidence(retrieval_results, outcomes)
