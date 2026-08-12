'''Prepare-incidents CLI unit tests.'''

import industrial_alarm_copilot.__main__ as cli
from industrial_alarm_copilot.incidents.artifacts import (
    IncidentArtifactPaths,
)


def test_prepare_incidents_command_passes_paths_settings_and_version(
    monkeypatch,
    tmp_path,
    capsys,
):
    events_path = tmp_path / 'events.parquet'
    config_path = tmp_path / 'default.toml'
    output_dir = tmp_path / 'processed'
    settings = {'incidents': {'gap_minutes': 30}}
    received = {}

    monkeypatch.setattr(cli, 'load_pipeline_settings', lambda path: settings)
    monkeypatch.setattr(cli, 'get_git_commit', lambda path: 'abc1234')

    def fake_prepare_incident_artifacts(**kwargs):
        received.update(kwargs)
        return IncidentArtifactPaths(
            incidents_parquet=output_dir / 'incidents.parquet',
            incident_events_parquet=output_dir / 'incident_events.parquet',
            baselines_json=output_dir / 'incident_baselines.json',
        )

    monkeypatch.setattr(
        cli,
        'prepare_incident_artifacts',
        fake_prepare_incident_artifacts,
    )

    exit_code = cli.main(
        [
            'prepare-incidents',
            '--events-parquet',
            str(events_path),
            '--config',
            str(config_path),
            '--output-dir',
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert received == {
        'events_parquet_path': events_path,
        'output_dir': output_dir,
        'pipeline_settings': settings,
        'code_version': 'abc1234',
    }
    output = capsys.readouterr().out
    assert str(output_dir / 'incidents.parquet') in output
    assert str(output_dir / 'incident_events.parquet') in output
    assert str(output_dir / 'incident_baselines.json') in output
    assert 'abc1234' in output
