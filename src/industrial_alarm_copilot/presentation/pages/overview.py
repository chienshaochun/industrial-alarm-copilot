'''Dataset overview page.'''

from pathlib import Path

import plotly.express as px
import streamlit as st

from industrial_alarm_copilot.presentation.data import (
    missing_artifacts,
    required_artifact_paths,
    resolve_project_root,
)
from industrial_alarm_copilot.presentation.runtime import load_overview_data


PROJECT_ROOT = resolve_project_root()


def _missing_artifact_message(missing: tuple[str, ...]) -> None:
    st.error('尚未找到應用所需的 processed artifacts：' + ', '.join(missing))
    st.code(
        'python -m industrial_alarm_copilot prepare-data\n'
        'python -m industrial_alarm_copilot prepare-incidents\n'
        'python -m industrial_alarm_copilot test-forecast',
        language='powershell',
    )


def render_overview_page() -> None:
    st.markdown('<div class="copilot-kicker">ALPI Portfolio Project</div>', unsafe_allow_html=True)
    st.title('Industrial Alarm Copilot')
    st.caption('把 444,834 筆離散 Alarm 轉換為可調查、可檢索、可預測的 episode。')

    missing = missing_artifacts(PROJECT_ROOT)
    if missing:
        _missing_artifact_message(missing)
        return
    paths = required_artifact_paths(PROJECT_ROOT)
    overview = load_overview_data(
        str(paths['events']),
        str(paths['incidents']),
    )

    columns = st.columns(4)
    columns[0].metric('Alarm events', f'{overview.event_count:,}')
    columns[1].metric('Derived episodes', f'{overview.incident_count:,}')
    columns[2].metric('設備', f'{overview.machine_count}')
    columns[3].metric('Alarm codes', f'{overview.alarm_code_count}')

    st.markdown('### 資料涵蓋期間')
    period_start, period_end = st.columns(2)
    period_start.metric('資料起點', f'{overview.start_time:%Y-%m-%d}')
    period_start.caption(f'時間：{overview.start_time:%H:%M:%S.%f}'[:-3])
    period_end.metric('資料終點', f'{overview.end_time:%Y-%m-%d}')
    period_end.caption(f'時間：{overview.end_time:%H:%M:%S.%f}'[:-3])
    st.caption(
        '完整時間範圍：'
        f'{overview.start_time:%Y-%m-%d %H:%M:%S.%f}'[:-3]
        + ' → '
        + f'{overview.end_time:%Y-%m-%d %H:%M:%S.%f}'[:-3]
    )

    st.markdown('### Alarm 活動量')
    monthly_figure = px.area(
        overview.monthly_events,
        x='month_start',
        y='event_count',
        labels={'month_start': '月份', 'event_count': '事件數'},
        color_discrete_sequence=['#2878b5'],
    )
    monthly_figure.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(monthly_figure, width='stretch')

    left, right = st.columns(2)
    with left:
        st.markdown('### 最常見 Alarm code')
        alarm_figure = px.bar(
            overview.top_alarms.sort_values('event_count'),
            x='event_count',
            y='alarm_code',
            orientation='h',
            labels={'event_count': '事件數', 'alarm_code': 'Alarm code'},
            color_discrete_sequence=['#2878b5'],
        )
        alarm_figure.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(alarm_figure, width='stretch')
    with right:
        st.markdown('### 各設備 Alarm 量')
        machine_figure = px.bar(
            overview.machine_events,
            x='machine_id',
            y='event_count',
            labels={'machine_id': '設備', 'event_count': '事件數'},
            color_discrete_sequence=['#e58b35'],
        )
        machine_figure.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(machine_figure, width='stretch')

    st.markdown('### Chronological split')
    st.dataframe(
        overview.split_incidents,
        width='stretch',
        hide_index=True,
        column_config={
            'split': '資料切分',
            'incident_count': st.column_config.NumberColumn(
                'Episode 數', format='%d'
            ),
        },
    )
    st.markdown(
        '<div class="copilot-note">所有模型與檢索特徵只以 train '
        '資料擬合；validation 用於選型，test 只進行一次最終驗收。</div>',
        unsafe_allow_html=True,
    )
