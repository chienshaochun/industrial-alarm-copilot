'''Deterministic summaries and a provider-neutral optional LLM adapter.'''

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from industrial_alarm_copilot.application.contracts import (
    InvestigationResult,
)
from industrial_alarm_copilot.copilot.contracts import (
    CopilotSummary,
    EvidenceNote,
)


class SummaryGenerator(Protocol):
    '''Port implemented by offline templates or an optional LLM provider.'''

    def generate(self, result: InvestigationResult) -> CopilotSummary:
        '''Generate a presentation-safe summary.'''


def _observed_facts(result: InvestigationResult) -> tuple[str, ...]:
    episode = result.observed
    tail_text = (
        '超過 train-only P95 統計基線'
        if episode.is_upper_tail
        else '未超過 train-only P95 統計基線'
    )
    return (
        f'設備 {episode.machine_id}，episode {episode.incident_id}。',
        f'觀察區間 {episode.start_time:%Y-%m-%d %H:%M:%S} 至 '
        f'{episode.end_time:%Y-%m-%d %H:%M:%S}。',
        f'共 {episode.event_count} 筆 Alarm、'
        f'{episode.distinct_alarm_count} 種代碼，{tail_text}。',
    )


def _historical_notes(result: InvestigationResult) -> tuple[EvidenceNote, ...]:
    return tuple(
        EvidenceNote(
            incident_id=item.incident_id,
            text=(
                f'相似度 {item.similarity_score:.3f}；共同 Alarm '
                f'{", ".join(item.shared_alarm_codes) or "無"}；其後 '
                f'{item.future_horizon_hours:g} 小時觀察到 '
                f'{", ".join(item.future_alarm_codes) or "無 Alarm"}。'
            ),
        )
        for item in result.retrieved_evidence
    )


def _prediction_context(result: InvestigationResult) -> tuple[str, ...]:
    return tuple(
        f'Top {item.rank}: Alarm {item.alarm_code}，分數 '
        f'{item.model_score:.3f}，基線層級 {item.baseline_scope}，'
        f'train support {item.train_support}。'
        for item in result.predictions
    )


class TemplateSummaryGenerator:
    '''Always-available, auditable summary requiring no external service.'''

    def generate(self, result: InvestigationResult) -> CopilotSummary:
        return CopilotSummary(
            overview=(
                '以下內容依序區分已觀察事實、相似歷史證據與統計預測；'
                '請勿把相似性或預測分數解讀成根因。'
            ),
            observed_facts=_observed_facts(result),
            historical_evidence=_historical_notes(result),
            prediction_context=_prediction_context(result),
            limitations=result.limitations,
            generator_name='deterministic_template_v1',
        )


class EvidenceGuardrailError(ValueError):
    '''Raised when generated structured output references unknown evidence.'''


class EvidenceConstrainedLLMGenerator:
    '''Provider-neutral LLM adapter with deterministic safety fallback.'''

    def __init__(
        self,
        invoke: Callable[[dict[str, Any]], Mapping[str, Any]],
        provider_name: str,
        fallback: SummaryGenerator | None = None,
    ) -> None:
        self._invoke = invoke
        self._provider_name = provider_name
        self._fallback = fallback or TemplateSummaryGenerator()

    def _payload(self, result: InvestigationResult) -> dict[str, Any]:
        template = TemplateSummaryGenerator().generate(result)
        return {
            'task': (
                '僅根據結構化內容撰寫簡短繁體中文概述；不得新增根因、'
                '維修指令、episode ID 或 Alarm code。'
            ),
            'observed_facts': template.observed_facts,
            'historical_evidence': [
                {'incident_id': note.incident_id, 'text': note.text}
                for note in template.historical_evidence
            ],
            'predictions': template.prediction_context,
            'limitations': template.limitations,
            'required_output': {
                'overview': 'string',
                'cited_incident_ids': 'list[string]',
                'cited_alarm_codes': 'list[string]',
            },
        }

    def _validated_summary(
        self,
        output: Mapping[str, Any],
        result: InvestigationResult,
    ) -> CopilotSummary:
        overview = str(output['overview']).strip()
        if not overview or len(overview) > 800:
            raise EvidenceGuardrailError('overview must contain 1-800 chars')
        cited_ids = tuple(str(value) for value in output['cited_incident_ids'])
        allowed_ids = {
            item.incident_id for item in result.retrieved_evidence
        }
        if not set(cited_ids).issubset(allowed_ids):
            raise EvidenceGuardrailError('summary cited an unknown episode')
        cited_codes = tuple(str(value) for value in output['cited_alarm_codes'])
        allowed_codes = {
            event.alarm_code for event in result.observed.alarm_sequence
        }
        allowed_codes.update(
            code
            for evidence in result.retrieved_evidence
            for code in (
                evidence.shared_alarm_codes + evidence.future_alarm_codes
            )
        )
        allowed_codes.update(
            prediction.alarm_code for prediction in result.predictions
        )
        if not set(cited_codes).issubset(allowed_codes):
            raise EvidenceGuardrailError('summary cited an unknown Alarm code')

        template = TemplateSummaryGenerator().generate(result)
        selected_notes = tuple(
            note
            for note in template.historical_evidence
            if note.incident_id in set(cited_ids)
        )
        return CopilotSummary(
            overview=overview,
            observed_facts=template.observed_facts,
            historical_evidence=selected_notes,
            prediction_context=template.prediction_context,
            limitations=template.limitations,
            generator_name=self._provider_name,
        )

    def generate(self, result: InvestigationResult) -> CopilotSummary:
        try:
            return self._validated_summary(
                self._invoke(self._payload(result)),
                result,
            )
        except Exception as error:
            fallback = self._fallback.generate(result)
            return CopilotSummary(
                overview=fallback.overview,
                observed_facts=fallback.observed_facts,
                historical_evidence=fallback.historical_evidence,
                prediction_context=fallback.prediction_context,
                limitations=fallback.limitations,
                generator_name=fallback.generator_name,
                used_fallback=True,
                fallback_reason=type(error).__name__,
            )
