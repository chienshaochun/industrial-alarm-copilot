'''Streamlit entry-point smoke test.'''

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_default_page_renders_without_exception():
    app_path = Path(__file__).resolve().parents[3] / 'app.py'
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == 'Industrial Alarm Copilot'


def test_streamlit_investigation_page_renders_without_exception():
    app = AppTest.from_string(
        'from industrial_alarm_copilot.presentation.pages.investigation '
        'import render_investigation_page\n'
        'render_investigation_page()',
        default_timeout=60,
    ).run()

    assert not app.exception
    assert app.title[0].value == '事件調查工作台'


def test_streamlit_evaluation_page_renders_without_exception():
    app = AppTest.from_string(
        'from industrial_alarm_copilot.presentation.pages.evaluation '
        'import render_evaluation_page\n'
        'render_evaluation_page()',
        default_timeout=30,
    ).run()

    assert not app.exception
    assert app.title[0].value == '模型評估'
