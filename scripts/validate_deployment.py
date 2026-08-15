'''Run all Streamlit pages using only the committed deployment snapshot.'''

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ENV = 'INDUSTRIAL_ALARM_ARTIFACT_DIR'


def _assert_page(app: AppTest, expected_title: str) -> None:
    if app.exception:
        messages = '; '.join(str(item.value) for item in app.exception)
        raise RuntimeError(f'Streamlit page failed: {messages}')
    titles = [item.value for item in app.title]
    if expected_title not in titles:
        raise RuntimeError(
            f'expected title {expected_title!r}, received {titles!r}'
        )


def main() -> None:
    os.environ[ARTIFACT_ENV] = 'data/deployment'

    overview = AppTest.from_file(
        PROJECT_ROOT / 'app.py',
        default_timeout=30,
    ).run()
    _assert_page(overview, 'Industrial Alarm Copilot')

    investigation = AppTest.from_string(
        'from industrial_alarm_copilot.presentation.pages.investigation '
        'import render_investigation_page\n'
        'render_investigation_page()',
        default_timeout=60,
    ).run()
    _assert_page(investigation, '事件調查工作台')

    evaluation = AppTest.from_string(
        'from industrial_alarm_copilot.presentation.pages.evaluation '
        'import render_evaluation_page\n'
        'render_evaluation_page()',
        default_timeout=30,
    ).run()
    _assert_page(evaluation, '模型評估')

    print('deployment smoke test: 3 pages passed')


if __name__ == '__main__':
    main()
