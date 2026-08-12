'''Proxy relevance labels for offline episode retrieval evaluation.'''

from collections.abc import Iterable

import pandas as pd


RELEVANCE_COLUMNS = [
    'query_outcome_is_complete',
    'query_has_future_alarms',
    'candidate_outcome_is_complete',
    'candidate_outcome_available_before_query',
    'evaluation_eligible',
    'outcome_jaccard',
    'relevance_threshold',
    'is_relevant',
]


def jaccard_alarm_codes(
    left_codes: Iterable[str],
    right_codes: Iterable[str],
) -> float:
    '''Return set overlap, treating two empty sets as no relevance signal.'''
    left_set = set(left_codes)
    right_set = set(right_codes)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def label_retrieval_relevance(
    retrieval_results: pd.DataFrame,
    outcomes: pd.DataFrame,
    incidents: pd.DataFrame,
    relevance_threshold: float,
) -> pd.DataFrame:
    '''Attach time-safe future-outcome relevance labels to retrieval rows.'''
    if not 0 < relevance_threshold <= 1:
        raise ValueError(
            'relevance_threshold must be greater than zero and at most one'
        )
    if not outcomes['incident_id'].is_unique:
        raise ValueError('outcomes incident_id must be unique')
    if not incidents['incident_id'].is_unique:
        raise ValueError('incidents incident_id must be unique')

    indexed_outcomes = outcomes.copy()
    indexed_outcomes['incident_id'] = indexed_outcomes[
        'incident_id'
    ].astype(str)
    indexed_outcomes = indexed_outcomes.set_index('incident_id')

    indexed_incidents = incidents.copy()
    indexed_incidents['incident_id'] = indexed_incidents[
        'incident_id'
    ].astype(str)
    indexed_incidents = indexed_incidents.set_index('incident_id')

    required_ids = set(
        retrieval_results['query_incident_id'].astype(str)
    ) | set(retrieval_results['candidate_incident_id'].astype(str))
    if not required_ids.issubset(indexed_outcomes.index):
        raise ValueError('every retrieval incident must have an outcome')
    query_ids = set(retrieval_results['query_incident_id'].astype(str))
    if not query_ids.issubset(indexed_incidents.index):
        raise ValueError('every query must reference an incident')

    records = []
    relevance_labels = []
    for result in retrieval_results.itertuples(index=False):
        query_id = str(result.query_incident_id)
        candidate_id = str(result.candidate_incident_id)
        query_outcome = indexed_outcomes.loc[query_id]
        candidate_outcome = indexed_outcomes.loc[candidate_id]
        query_start_time = indexed_incidents.loc[query_id, 'start_time']

        query_complete = bool(query_outcome['outcome_is_complete'])
        query_has_future = bool(query_outcome['has_future_alarms'])
        candidate_complete = bool(
            candidate_outcome['outcome_is_complete']
        )
        candidate_available = bool(
            candidate_outcome['outcome_end_time'] < query_start_time
        )
        evaluation_eligible = (
            query_complete
            and query_has_future
            and candidate_complete
            and candidate_available
        )

        if evaluation_eligible:
            outcome_jaccard = jaccard_alarm_codes(
                query_outcome['future_alarm_codes'],
                candidate_outcome['future_alarm_codes'],
            )
            relevance_label = outcome_jaccard >= relevance_threshold
        else:
            outcome_jaccard = float('nan')
            relevance_label = pd.NA

        records.append(
            {
                'query_outcome_is_complete': query_complete,
                'query_has_future_alarms': query_has_future,
                'candidate_outcome_is_complete': candidate_complete,
                'candidate_outcome_available_before_query': (
                    candidate_available
                ),
                'evaluation_eligible': evaluation_eligible,
                'outcome_jaccard': outcome_jaccard,
                'relevance_threshold': float(relevance_threshold),
            }
        )
        relevance_labels.append(relevance_label)

    labeled_results = retrieval_results.reset_index(drop=True).copy()
    relevance_values = pd.DataFrame.from_records(
        records,
        columns=RELEVANCE_COLUMNS[:-1],
    )
    labeled_results = pd.concat(
        [labeled_results, relevance_values],
        axis='columns',
    )
    labeled_results['is_relevant'] = pd.array(
        relevance_labels,
        dtype='boolean',
    )
    return labeled_results[
        list(retrieval_results.columns) + RELEVANCE_COLUMNS
    ]
