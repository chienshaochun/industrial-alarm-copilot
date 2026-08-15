'''Investigation orchestration tests.'''

from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse

from industrial_alarm_copilot.application.forecast import (
    RuntimeForecastModel,
)
from industrial_alarm_copilot.application.investigation import (
    InvestigationResources,
    InvestigationService,
)
from industrial_alarm_copilot.retrieval.features import AlarmFeatureMatrix
from industrial_alarm_copilot.retrieval.search import (
    build_retrieval_search_index,
)
from industrial_alarm_copilot.retrieval.settings import (
    RetrievalExperimentSettings,
)


def _resources() -> InvestigationResources:
    incidents = pd.DataFrame(
        {
            'incident_id': ['old', 'query'],
            'machine_id': ['1', '1'],
            'split': ['train', 'validation'],
            'start_time': pd.to_datetime(['2020-01-01 08:00', '2020-01-02 08:00']),
            'end_time': pd.to_datetime(['2020-01-01 08:05', '2020-01-02 08:10']),
            'duration_seconds': [300.0, 600.0],
            'event_count': [1, 2],
            'distinct_alarm_count': [1, 2],
            'is_upper_tail': [False, True],
            'is_high_event_count': [False, True],
            'is_high_duration_seconds': [False, False],
            'is_high_distinct_alarm_count': [False, True],
        }
    )
    events = pd.DataFrame(
        {
            'source_row': [0, 1, 2],
            'timestamp': pd.to_datetime(
                ['2020-01-01 08:00', '2020-01-02 08:00', '2020-01-02 08:10']
            ),
            'alarm_code': ['11', '11', '98'],
            'machine_id': ['1', '1', '1'],
            'gap_seconds': [np.nan, 86100.0, 600.0],
        }
    )
    mapping = pd.DataFrame(
        {
            'incident_id': ['old', 'query', 'query'],
            'source_row': [0, 1, 2],
            'event_position': [0, 0, 1],
        }
    )
    documents = pd.DataFrame(
        {'incident_id': ['old', 'query'], 'alarm_document': ['11', '11 98']}
    )
    features = AlarmFeatureMatrix(
        incident_ids=('old', 'query'),
        matrix=sparse.csr_matrix([[1.0, 0.0], [1.0, 1.0]]),
        feature_names=('11', '98'),
        feature_version='alarm_tfidf_v1',
    )
    outcomes = pd.DataFrame(
        {
            'incident_id': ['old', 'query'],
            'outcome_is_complete': [True, False],
            'future_horizon_hours': [6.0, 6.0],
            'future_alarm_codes': [('98',), ()],
        }
    )
    model = RuntimeForecastModel(
        model_version='transition_frequency_v1',
        alarm_codes=('11', '98'),
        global_scores=np.array([0.2, 0.8]),
        machine_scores={'1': np.array([0.3, 0.7])},
        machine_support={'1': 100},
        transition_scores={('1', '98'): np.array([0.9, 0.1])},
        transition_support={('1', '98'): 20},
    )
    settings = RetrievalExperimentSettings(
        top_k=5,
        alarm_weight=1.0,
        shape_weight=1.0,
        candidate_policy='expanding_history',
        future_horizon_hours_candidates=(6.0,),
        relevance_threshold_candidates=(0.3,),
        selected_feature_version='alarm_tfidf_v1',
        future_horizon_hours=6.0,
        relevance_threshold=0.3,
    )
    return InvestigationResources(
        events=events,
        incidents=incidents,
        incident_events=mapping,
        outcomes=outcomes,
        search_index=build_retrieval_search_index(
            incidents, documents, features
        ),
        forecast_model=model,
        retrieval_settings=settings,
        forecast_top_k=2,
        forecast_horizon_hours=6.0,
    )


def test_service_lists_options_and_assembles_separate_sections():
    service = InvestigationService(_resources())

    assert service.machine_ids() == ('1',)
    options = service.episode_options('1')
    assert [option.incident_id for option in options] == ['query', 'old']

    result = service.investigate('query')

    assert result.observed.incident_id == 'query'
    assert result.observed.upper_tail_flags == (
        'high_event_count',
        'high_alarm_diversity',
    )
    assert [item.incident_id for item in result.retrieved_evidence] == ['old']
    assert result.retrieved_evidence[0].future_alarm_codes == ('98',)
    assert [item.alarm_code for item in result.predictions] == ['11', '98']
    assert result.predictions[0].baseline_scope == 'transition'
    assert len(result.limitations) == 3


def test_service_never_retrieves_an_episode_after_the_query():
    service = InvestigationService(_resources())

    result = service.investigate('old')

    assert result.retrieved_evidence == ()
