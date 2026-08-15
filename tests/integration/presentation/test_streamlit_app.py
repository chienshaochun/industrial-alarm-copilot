'''Streamlit entry-point smoke test.'''

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_default_page_renders_without_exception():
    app_path = Path(__file__).resolve().parents[3] / 'app.py'
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == 'Industrial Alarm Copilot'
