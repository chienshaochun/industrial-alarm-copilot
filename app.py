'''Streamlit entry point for the Industrial Alarm Copilot.'''

import os
from pathlib import Path

import streamlit as st


os.environ.setdefault(
    'INDUSTRIAL_ALARM_PROJECT_ROOT',
    str(Path(__file__).resolve().parent),
)

from industrial_alarm_copilot.presentation.pages.evaluation import (
    render_evaluation_page,
)
from industrial_alarm_copilot.presentation.pages.investigation import (
    render_investigation_page,
)
from industrial_alarm_copilot.presentation.pages.overview import (
    render_overview_page,
)
from industrial_alarm_copilot.presentation.theme import apply_app_theme


st.set_page_config(
    page_title='Industrial Alarm Copilot',
    page_icon='🏭',
    layout='wide',
)
apply_app_theme()

navigation = st.navigation(
    [
        st.Page(
            render_overview_page,
            title='資料總覽',
            icon=':material/dashboard:',
            default=True,
        ),
        st.Page(
            render_investigation_page,
            title='事件調查工作台',
            icon=':material/manage_search:',
        ),
        st.Page(
            render_evaluation_page,
            title='模型評估',
            icon=':material/analytics:',
        ),
    ]
)
navigation.run()
