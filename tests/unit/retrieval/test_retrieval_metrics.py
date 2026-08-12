'''Binary retrieval metric tests.'''

import math

import pytest

from industrial_alarm_copilot.retrieval.metrics import (
    compute_binary_ranking_metrics,
)


def test_compute_binary_ranking_metrics_uses_rank_and_full_candidate_count():
    metrics = compute_binary_ranking_metrics(
        [False, True, True, False],
        k=5,
        total_relevant_candidate_count=4,
    )

    actual_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, 5))

    assert metrics.hit_at_k is True
    assert metrics.precision_at_k == pytest.approx(2 / 5)
    assert metrics.recall_at_k == pytest.approx(2 / 4)
    assert metrics.reciprocal_rank == pytest.approx(1 / 2)
    assert metrics.ndcg_at_k == pytest.approx(actual_dcg / ideal_dcg)
    assert metrics.retrieved_count == 4


def test_compute_binary_ranking_metrics_marks_no_relevant_pool_undefined():
    metrics = compute_binary_ranking_metrics(
        [False, False],
        k=3,
        total_relevant_candidate_count=0,
    )

    assert metrics.hit_at_k is False
    assert metrics.precision_at_k == 0.0
    assert metrics.reciprocal_rank == 0.0
    assert math.isnan(metrics.recall_at_k)
    assert math.isnan(metrics.ndcg_at_k)
