"""Industrial Alarm Copilot 命令列入口。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from industrial_alarm_copilot.data.pipeline import prepare_data_artifacts
from industrial_alarm_copilot.data.runtime import (
    get_git_commit,
    load_pipeline_settings,
)
from industrial_alarm_copilot.incidents.artifacts import (
    prepare_incident_artifacts,
)
from industrial_alarm_copilot.retrieval.pipeline import (
    run_selected_test_from_artifacts,
    run_validation_from_artifacts,
)
from industrial_alarm_copilot.retrieval.experiments import (
    select_retrieval_diagnostics,
)
from industrial_alarm_copilot.retrieval.artifacts import (
    write_retrieval_test_artifacts,
    write_retrieval_validation_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """建立命令列參數解析器。"""
    parser = argparse.ArgumentParser(prog="industrial-alarm-copilot")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_data = commands.add_parser(
        "prepare-data",
        help="由 ALPI raw CSV 產生 processed events artifacts",
    )
    prepare_data.add_argument(
        "--raw-csv",
        type=Path,
        default=Path("data/raw/alarms.csv"),
    )
    prepare_data.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
    )
    prepare_data.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )

    prepare_incidents = commands.add_parser(
        'prepare-incidents',
        help='Build incident analysis artifacts from processed events',
    )
    prepare_incidents.add_argument(
        '--events-parquet',
        type=Path,
        default=Path('data/processed/events.parquet'),
    )
    prepare_incidents.add_argument(
        '--config',
        type=Path,
        default=Path('configs/default.toml'),
    )
    prepare_incidents.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
    )

    validate_retrieval = commands.add_parser(
        'validate-retrieval',
        help='Run the retrieval experiment grid on validation episodes',
    )
    validate_retrieval.add_argument(
        '--events-parquet',
        type=Path,
        default=Path('data/processed/events.parquet'),
    )
    validate_retrieval.add_argument(
        '--incidents-parquet',
        type=Path,
        default=Path('data/processed/incidents.parquet'),
    )
    validate_retrieval.add_argument(
        '--incident-events-parquet',
        type=Path,
        default=Path('data/processed/incident_events.parquet'),
    )
    validate_retrieval.add_argument(
        '--config',
        type=Path,
        default=Path('configs/default.toml'),
    )
    validate_retrieval.add_argument(
        '--max-validation-queries',
        type=int,
        default=None,
    )
    validate_retrieval.add_argument(
        '--diagnostics-only',
        action='store_true',
        help='Print compact Top-5 diagnostics as CSV',
    )
    validate_retrieval.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Optionally write validation CSV and metadata artifacts',
    )

    test_retrieval = commands.add_parser(
        'test-retrieval',
        help='Evaluate the locked retrieval setting on test episodes',
    )
    test_retrieval.add_argument(
        '--events-parquet',
        type=Path,
        default=Path('data/processed/events.parquet'),
    )
    test_retrieval.add_argument(
        '--incidents-parquet',
        type=Path,
        default=Path('data/processed/incidents.parquet'),
    )
    test_retrieval.add_argument(
        '--incident-events-parquet',
        type=Path,
        default=Path('data/processed/incident_events.parquet'),
    )
    test_retrieval.add_argument(
        '--config',
        type=Path,
        default=Path('configs/default.toml'),
    )
    test_retrieval.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed'),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """執行 Industrial Alarm Copilot CLI。"""
    args = build_parser().parse_args(argv)

    if args.command == "prepare-data":
        settings = load_pipeline_settings(args.config)
        code_version = get_git_commit(Path.cwd())
        artifact_paths = prepare_data_artifacts(
            raw_csv_path=args.raw_csv,
            output_dir=args.output_dir,
            pipeline_settings=settings,
            code_version=code_version,
        )
        print(f"events artifact: {artifact_paths.events_parquet}")
        print(f"metadata artifact: {artifact_paths.metadata_json}")
        print(f"code version: {code_version}")

    if args.command == 'prepare-incidents':
        settings = load_pipeline_settings(args.config)
        code_version = get_git_commit(Path.cwd())
        artifact_paths = prepare_incident_artifacts(
            events_parquet_path=args.events_parquet,
            output_dir=args.output_dir,
            pipeline_settings=settings,
            code_version=code_version,
        )
        print(f'incidents artifact: {artifact_paths.incidents_parquet}')
        print(
            'incident events artifact: '
            f'{artifact_paths.incident_events_parquet}'
        )
        print(f'baseline metadata: {artifact_paths.baselines_json}')
        print(f'code version: {code_version}')

    if args.command == 'validate-retrieval':
        settings = load_pipeline_settings(args.config)
        experiment_results = run_validation_from_artifacts(
            events_parquet_path=args.events_parquet,
            incidents_parquet_path=args.incidents_parquet,
            incident_events_parquet_path=args.incident_events_parquet,
            pipeline_settings=settings,
            max_validation_queries=args.max_validation_queries,
        )
        if args.diagnostics_only:
            diagnostics = select_retrieval_diagnostics(experiment_results)
            print(diagnostics.to_csv(index=False), end='')
        else:
            print(experiment_results.to_string(index=False))
        if args.output_dir is not None:
            code_version = get_git_commit(Path.cwd())
            artifact_paths = write_retrieval_validation_artifacts(
                experiment_results,
                source_paths={
                    'events': args.events_parquet,
                    'incidents': args.incidents_parquet,
                    'incident_events': args.incident_events_parquet,
                },
                retrieval_settings=settings['retrieval'],
                code_version=code_version,
                output_dir=args.output_dir,
                query_limit=args.max_validation_queries,
            )
            print(f'retrieval results artifact: {artifact_paths.results_csv}')
            print(
                'retrieval metadata artifact: '
                f'{artifact_paths.metadata_json}'
            )
            print(f'code version: {code_version}')
        if args.max_validation_queries is not None:
            print(
                'smoke-run validation query limit: '
                f'{args.max_validation_queries}'
            )

    if args.command == 'test-retrieval':
        settings = load_pipeline_settings(args.config)
        test_results = run_selected_test_from_artifacts(
            events_parquet_path=args.events_parquet,
            incidents_parquet_path=args.incidents_parquet,
            incident_events_parquet_path=args.incident_events_parquet,
            pipeline_settings=settings,
        )
        diagnostics = select_retrieval_diagnostics(test_results)
        print(diagnostics.to_csv(index=False), end='')

        code_version = get_git_commit(Path.cwd())
        artifact_paths = write_retrieval_test_artifacts(
            test_results,
            source_paths={
                'events': args.events_parquet,
                'incidents': args.incidents_parquet,
                'incident_events': args.incident_events_parquet,
            },
            retrieval_settings=settings['retrieval'],
            code_version=code_version,
            output_dir=args.output_dir,
        )
        print(f'retrieval test artifact: {artifact_paths.results_csv}')
        print(
            'retrieval test metadata: '
            f'{artifact_paths.metadata_json}'
        )
        print(f'code version: {code_version}')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
