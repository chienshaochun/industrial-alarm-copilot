'''Run all Streamlit pages using only the committed deployment snapshot.'''

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ENV = 'INDUSTRIAL_ALARM_ARTIFACT_DIR'
PROJECT_ROOT_ENV = 'INDUSTRIAL_ALARM_PROJECT_ROOT'


def _assert_page(
    app: AppTest,
    expected_title: str,
    minimum_metric_count: int,
) -> None:
    if app.exception:
        messages = '; '.join(str(item.value) for item in app.exception)
        raise RuntimeError(f'Streamlit page failed: {messages}')
    if app.error:
        messages = '; '.join(str(item.value) for item in app.error)
        raise RuntimeError(f'Streamlit page rendered an error: {messages}')
    titles = [item.value for item in app.title]
    if expected_title not in titles:
        raise RuntimeError(
            f'expected title {expected_title!r}, received {titles!r}'
        )
    if len(app.metric) < minimum_metric_count:
        raise RuntimeError(
            f'{expected_title} rendered only {len(app.metric)} metrics'
        )


def main() -> None:
    os.environ[ARTIFACT_ENV] = 'data/deployment'
    os.environ[PROJECT_ROOT_ENV] = str(PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)

    overview = AppTest.from_file(
        PROJECT_ROOT / 'app.py',
        default_timeout=30,
    ).run()
    _assert_page(overview, 'Industrial Alarm Copilot', 5)

    investigation = AppTest.from_string(
        'from industrial_alarm_copilot.presentation.pages.investigation '
        'import render_investigation_page\n'
        'render_investigation_page()',
        default_timeout=60,
    ).run()
    _assert_page(investigation, '事件調查工作台', 5)

    evaluation = AppTest.from_string(
        'from industrial_alarm_copilot.presentation.pages.evaluation '
        'import render_evaluation_page\n'
        'render_evaluation_page()',
        default_timeout=30,
    ).run()
    _assert_page(evaluation, '模型評估', 10)

    print('deployment smoke test: 3 pages passed')


if __name__ == '__main__':
    main()
