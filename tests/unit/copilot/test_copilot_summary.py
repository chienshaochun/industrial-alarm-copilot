'''Evidence-constrained summary tests.'''

from datetime import datetime

from industrial_alarm_copilot.application.contracts import (
    AlarmEventFact,
    ForecastPrediction,
    InvestigationResult,
    ObservedEpisode,
    RetrievedEpisodeEvidence,
)
from industrial_alarm_copilot.copilot.summary import (
    EvidenceConstrainedLLMGenerator,
    TemplateSummaryGenerator,
)


def _result() -> InvestigationResult:
    return InvestigationResult(
        observed=ObservedEpisode(
            incident_id='query',
            machine_id='4',
            split='test',
            start_time=datetime(2020, 1, 2, 8),
            end_time=datetime(2020, 1, 2, 8, 5),
            duration_seconds=300.0,
            event_count=2,
            distinct_alarm_count=2,
            is_upper_tail=False,
            upper_tail_flags=(),
            alarm_sequence=(
                AlarmEventFact(datetime(2020, 1, 2, 8), '11', None),
                AlarmEventFact(datetime(2020, 1, 2, 8, 5), '98', 300.0),
            ),
        ),
        retrieved_evidence=(
            RetrievedEpisodeEvidence(
                rank=1,
                incident_id='evidence-1',
                machine_id='4',
                start_time=datetime(2019, 12, 1, 8),
                end_time=datetime(2019, 12, 1, 8, 2),
                similarity_score=0.8,
                shared_alarm_codes=('98',),
                future_alarm_codes=('26',),
                outcome_is_complete=True,
                future_horizon_hours=6.0,
            ),
        ),
        predictions=(
            ForecastPrediction(
                rank=1,
                alarm_code='26',
                model_score=0.6,
                model_version='transition_frequency_v1',
                forecast_horizon_hours=6.0,
                baseline_scope='transition',
                train_support=100,
            ),
        ),
        limitations=('不是根因診斷。',),
    )


def test_template_keeps_information_classes_separate_and_cited():
    summary = TemplateSummaryGenerator().generate(_result())

    assert '設備 4' in summary.observed_facts[0]
    assert summary.historical_evidence[0].incident_id == 'evidence-1'
    assert 'Alarm 26' in summary.prediction_context[0]
    assert summary.limitations == ('不是根因診斷。',)
    assert not summary.used_fallback


def test_llm_adapter_accepts_only_known_structured_citations():
    generator = EvidenceConstrainedLLMGenerator(
        invoke=lambda payload: {
            'overview': 'Alarm 組合與一筆歷史 episode 接近，請查閱證據。',
            'cited_incident_ids': ['evidence-1'],
            'cited_alarm_codes': ['98', '26'],
        },
        provider_name='fake-llm',
    )

    summary = generator.generate(_result())

    assert summary.generator_name == 'fake-llm'
    assert [note.incident_id for note in summary.historical_evidence] == [
        'evidence-1'
    ]
    assert not summary.used_fallback


def test_llm_adapter_falls_back_when_it_cites_unknown_evidence():
    generator = EvidenceConstrainedLLMGenerator(
        invoke=lambda payload: {
            'overview': '這是一段無法驗證的內容。',
            'cited_incident_ids': ['invented-episode'],
            'cited_alarm_codes': ['999'],
        },
        provider_name='fake-llm',
    )

    summary = generator.generate(_result())

    assert summary.generator_name == 'deterministic_template_v1'
    assert summary.used_fallback
    assert summary.fallback_reason == 'EvidenceGuardrailError'
    assert summary.historical_evidence[0].incident_id == 'evidence-1'
