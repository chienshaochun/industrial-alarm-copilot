'''Presentation contracts for deterministic and optional LLM summaries.'''

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceNote:
    '''A statement explicitly attached to one retrieved episode.'''

    incident_id: str
    text: str


@dataclass(frozen=True, slots=True)
class CopilotSummary:
    '''Four deliberately separated information classes for the UI.'''

    overview: str
    observed_facts: tuple[str, ...]
    historical_evidence: tuple[EvidenceNote, ...]
    prediction_context: tuple[str, ...]
    limitations: tuple[str, ...]
    generator_name: str
    used_fallback: bool = False
    fallback_reason: str | None = None
