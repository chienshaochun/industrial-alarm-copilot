'''Binary ranking metrics for retrieval evaluation.'''

from collections.abc import Iterable
from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class BinaryRankingMetrics:
    '''Metrics for one query at one K cutoff.'''

    k: int
    retrieved_count: int
    relevant_retrieved_count: int
    total_relevant_candidate_count: int
    hit_at_k: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def _discounted_gain(relevance_labels: tuple[bool, ...]) -> float:
    return sum(
        1 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(relevance_labels, start=1)
        if is_relevant
    )


def compute_binary_ranking_metrics(
    relevance_by_rank: Iterable[bool],
    k: int,
    total_relevant_candidate_count: int,
) -> BinaryRankingMetrics:
    '''Compute per-query metrics without hiding missing retrieval slots.'''
    if k <= 0:
        raise ValueError('k must be greater than zero')
    if total_relevant_candidate_count < 0:
        raise ValueError(
            'total_relevant_candidate_count cannot be negative'
        )

    raw_labels = tuple(relevance_by_rank)
    if any(pd.isna(label) for label in raw_labels):
        raise ValueError('relevance labels must be fully evaluable')
    labels = tuple(bool(label) for label in raw_labels)
    if len(labels) > k:
        raise ValueError('relevance_by_rank cannot contain more than k rows')

    relevant_retrieved_count = sum(labels)
    if relevant_retrieved_count > total_relevant_candidate_count:
        raise ValueError(
            'retrieved relevant count cannot exceed total relevant count'
        )

    first_relevant_rank = next(
        (
            rank
            for rank, is_relevant in enumerate(labels, start=1)
            if is_relevant
        ),
        None,
    )
    reciprocal_rank = (
        0.0 if first_relevant_rank is None else 1 / first_relevant_rank
    )
    recall_at_k = (
        float('nan')
        if total_relevant_candidate_count == 0
        else relevant_retrieved_count / total_relevant_candidate_count
    )

    ideal_relevant_count = min(total_relevant_candidate_count, k)
    ideal_labels = (True,) * ideal_relevant_count
    ideal_discounted_gain = _discounted_gain(ideal_labels)
    ndcg_at_k = (
        float('nan')
        if ideal_discounted_gain == 0
        else _discounted_gain(labels) / ideal_discounted_gain
    )

    return BinaryRankingMetrics(
        k=k,
        retrieved_count=len(labels),
        relevant_retrieved_count=relevant_retrieved_count,
        total_relevant_candidate_count=total_relevant_candidate_count,
        hit_at_k=bool(relevant_retrieved_count),
        precision_at_k=relevant_retrieved_count / k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg_at_k,
    )
