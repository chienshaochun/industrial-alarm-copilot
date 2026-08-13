'''Immutable outputs shared by application services and presentation code.'''

from dataclasses import dataclass
from datetime import datetime
import math


VALID_BASELINE_SCOPES = frozenset(
    {'transition', 'machine_fallback', 'global_fallback'}
)


@dataclass(frozen=True, slots=True)
class AlarmEventFact:
    '''One observed alarm event inside the selected episode.'''

    timestamp: datetime
    alarm_code: str
    gap_seconds: float | None

    def __post_init__(self) -> None:
        if self.gap_seconds is not None and (
            not math.isfinite(self.gap_seconds) or self.gap_seconds < 0
        ):
            raise ValueError('gap_seconds must be finite and nonnegative')


@dataclass(frozen=True, slots=True)
class ObservedEpisode:
    '''Facts calculated directly from the selected derived episode.'''

    incident_id: str
    machine_id: str
    split: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    event_count: int
    distinct_alarm_count: int
    is_upper_tail: bool
    upper_tail_flags: tuple[str, ...]
    alarm_sequence: tuple[AlarmEventFact, ...]

    def __post_init__(self) -> None:
        if self.event_count < 1:
            raise ValueError('event_count must be positive')
        if not 1 <= self.distinct_alarm_count <= self.event_count:
            raise ValueError(
                'distinct_alarm_count must be between one and event_count'
            )
        if len(self.alarm_sequence) != self.event_count:
            raise ValueError('alarm_sequence length must equal event_count')


@dataclass(frozen=True, slots=True)
class RetrievedEpisodeEvidence:
    '''One time-safe historical episode returned as cited evidence.'''

    rank: int
    incident_id: str
    machine_id: str
    start_time: datetime
    end_time: datetime
    similarity_score: float
    shared_alarm_codes: tuple[str, ...]
    future_alarm_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError('rank must be positive')
        if (
            not math.isfinite(self.similarity_score)
            or self.similarity_score < -1.000001
            or self.similarity_score > 1.000001
        ):
            raise ValueError(
                'similarity_score must be within the cosine range'
            )


@dataclass(frozen=True, slots=True)
class ForecastPrediction:
    '''One ranked future-alarm candidate with model provenance.'''

    rank: int
    alarm_code: str
    model_score: float
    model_version: str
    forecast_horizon_hours: float
    baseline_scope: str
    train_support: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError('rank must be positive')
        if self.train_support < 0:
            raise ValueError('train_support must be nonnegative')
        if self.baseline_scope not in VALID_BASELINE_SCOPES:
            raise ValueError('baseline_scope is not supported')


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    '''Presentation-safe investigation sections that cannot be mixed.'''

    observed: ObservedEpisode
    retrieved_evidence: tuple[RetrievedEpisodeEvidence, ...]
    predictions: tuple[ForecastPrediction, ...]
    limitations: tuple[str, ...]
