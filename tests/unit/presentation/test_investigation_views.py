'''Investigation table adapter tests.'''

from tests.unit.copilot.test_copilot_summary import _result

from industrial_alarm_copilot.presentation.investigation_views import (
    forecast_prediction_frame,
    observed_event_frame,
    retrieved_evidence_frame,
)


def test_investigation_frames_keep_information_classes_separate():
    result = _result()

    observed = observed_event_frame(result)
    evidence = retrieved_evidence_frame(result)
    predictions = forecast_prediction_frame(result)

    assert observed['alarm_code'].tolist() == ['11', '98']
    assert 'similarity_score' not in observed
    assert evidence['incident_id'].tolist() == ['evidence-1']
    assert evidence['future_alarm_codes'].tolist() == ['26']
    assert predictions['alarm_code'].tolist() == ['26']
    assert predictions['baseline_scope'].tolist() == ['transition']
