'''Interactive evidence-safe episode investigation workbench.'''

from pathlib import Path

import plotly.express as px
import streamlit as st

from industrial_alarm_copilot.copilot.summary import (
    TemplateSummaryGenerator,
)
from industrial_alarm_copilot.presentation.charts import (
    render_interactive_chart,
)
from industrial_alarm_copilot.presentation.data import (
    missing_artifacts,
    resolve_project_root,
)
from industrial_alarm_copilot.presentation.investigation_views import (
    forecast_prediction_frame,
    observed_event_frame,
    retrieved_evidence_frame,
)
from industrial_alarm_copilot.presentation.runtime import (
    load_investigation_service,
)


PROJECT_ROOT = resolve_project_root()


def _episode_label(option) -> str:
    tail = '｜P95 upper-tail' if option.is_upper_tail else ''
    return (
        f'{option.start_time:%Y-%m-%d %H:%M}｜{option.event_count} events｜'
        f'{option.split}{tail}'
    )


def _render_observed(result) -> None:
    events = observed_event_frame(result)
    figure = px.scatter(
        events,
        x='timestamp',
        y='alarm_code',
        color='alarm_code',
        custom_data=['event_position', 'gap_seconds'],
        labels={'timestamp': '時間', 'alarm_code': 'Alarm code'},
    )
    figure.update_traces(
        marker={'size': 11},
        hovertemplate=(
            '<b>Alarm code %{y}</b><br>'
            '時間：%{x|%Y-%m-%d %H:%M:%S.%L}<br>'
            '事件順序：%{customdata[0]}<br>'
            '距前一筆：%{customdata[1]:.3f} 秒<extra></extra>'
        ),
    )
    figure.update_layout(
        showlegend=False,
    )
    render_interactive_chart(figure, key='investigation-observed-events')
    st.dataframe(
        events,
        width='stretch',
        hide_index=True,
        column_config={
            'event_position': '順序',
            'timestamp': st.column_config.DatetimeColumn('時間'),
            'alarm_code': 'Alarm code',
            'gap_seconds': st.column_config.NumberColumn(
                '距前一筆（秒）', format='%.3f'
            ),
        },
    )


def _render_evidence(result) -> None:
    evidence = retrieved_evidence_frame(result)
    if evidence.empty:
        st.info('此 episode 之前沒有符合時間條件的歷史候選。')
        return
    st.dataframe(
        evidence,
        width='stretch',
        hide_index=True,
        column_config={
            'rank': '排名',
            'incident_id': '證據 episode ID',
            'machine_id': '設備',
            'start_time': st.column_config.DatetimeColumn('起始時間'),
            'similarity_score': st.column_config.ProgressColumn(
                '相似度', min_value=0.0, max_value=1.0, format='%.3f'
            ),
            'shared_alarm_codes': '共同 Alarm',
            'future_alarm_codes': '其後觀察 Alarm',
            'outcome_is_complete': '觀察窗完整',
            'future_horizon_hours': '觀察窗（小時）',
        },
    )
    st.caption('候選必須在被調查 episode 開始前已結束；episode ID 即為摘要引用來源。')


def _render_predictions(result) -> None:
    predictions = forecast_prediction_frame(result)
    figure = px.bar(
        predictions.sort_values('model_score'),
        x='model_score',
        y='alarm_code',
        orientation='h',
        color='baseline_scope',
        custom_data=['train_support'],
        labels={
            'model_score': '模型分數',
            'alarm_code': 'Alarm code',
            'baseline_scope': '基線層級',
        },
    )
    figure.update_traces(
        hovertemplate=(
            '<b>Alarm code %{y}</b><br>'
            '模型分數：%{x:.3f}<br>'
            'Train support：%{customdata[0]:,}<extra></extra>'
        )
    )
    render_interactive_chart(figure, key='investigation-forecast')
    st.dataframe(
        predictions,
        width='stretch',
        hide_index=True,
        column_config={
            'rank': '排名',
            'alarm_code': 'Alarm code',
            'model_score': st.column_config.NumberColumn(
                '分數', format='%.3f'
            ),
            'model_version': '模型',
            'forecast_horizon_hours': '預測窗（小時）',
            'baseline_scope': '基線層級',
            'train_support': 'Train support',
        },
    )
    st.warning('模型分數是相對排序依據，不是設備會故障的校準機率。')


def _render_summary(result) -> None:
    summary = TemplateSummaryGenerator().generate(result)
    st.markdown(summary.overview)
    st.markdown('#### 已觀察事實')
    for fact in summary.observed_facts:
        st.markdown(f'- {fact}')
    st.markdown('#### 相似歷史證據')
    if not summary.historical_evidence:
        st.markdown('- 無可引用的歷史 episode。')
    for note in summary.historical_evidence:
        st.markdown(f'- `{note.incident_id}`：{note.text}')
    st.markdown('#### 統計預測')
    for prediction in summary.prediction_context:
        st.markdown(f'- {prediction}')
    st.markdown('#### 使用限制')
    for limitation in summary.limitations:
        st.markdown(f'- {limitation}')
    st.caption(f'摘要產生器：{summary.generator_name}（離線、可重現）')


def render_investigation_page() -> None:
    st.markdown('<div class="copilot-kicker">Evidence-first Workflow</div>', unsafe_allow_html=True)
    st.title('事件調查工作台')
    st.caption('先選設備與 episode，再分開檢視觀察、歷史證據與統計預測。')

    missing = missing_artifacts(PROJECT_ROOT)
    if missing:
        st.error('缺少 processed artifacts：' + ', '.join(missing))
        return
    with st.spinner('第一次載入會建立 train-fitted retrieval index…'):
        service = load_investigation_service(str(PROJECT_ROOT))

    selector_left, selector_right = st.columns([1, 3])
    with selector_left:
        machine_id = st.selectbox(
            '設備', service.machine_ids(), key='investigation_machine'
        )
    options = service.episode_options(machine_id)
    labels = {_episode_label(option): option for option in options}
    with selector_right:
        selected_label = st.selectbox(
            'Derived episode',
            tuple(labels),
            key='investigation_episode',
        )
    result = service.investigate(labels[selected_label].incident_id)

    episode = result.observed
    metrics = st.columns(5)
    metrics[0].metric('Episode ID', episode.incident_id[:12] + '…')
    metrics[1].metric('Alarm events', episode.event_count)
    metrics[2].metric('Distinct codes', episode.distinct_alarm_count)
    metrics[3].metric('Duration', f'{episode.duration_seconds / 60:.1f} min')
    metrics[4].metric(
        '統計基線', 'Upper-tail' if episode.is_upper_tail else 'Normal range'
    )

    main, side = st.columns([3, 1])
    with main:
        observed_tab, evidence_tab, forecast_tab, summary_tab = st.tabs(
            ['已觀察事實', '相似歷史事件', '未來 Alarm 預測', 'Copilot 摘要']
        )
        with observed_tab:
            _render_observed(result)
        with evidence_tab:
            _render_evidence(result)
        with forecast_tab:
            _render_predictions(result)
        with summary_tab:
            _render_summary(result)
    with side:
        st.markdown('### 快速判讀')
        st.markdown(f'**資料切分**  \n`{episode.split}`')
        st.markdown(
            f'**相似證據**  \n{len(result.retrieved_evidence)} 筆'
        )
        st.markdown(
            f'**Forecast scope**  \n'
            f'`{result.predictions[0].baseline_scope}`'
        )
        if episode.upper_tail_flags:
            st.markdown('**觸發旗標**')
            for flag in episode.upper_tail_flags:
                st.markdown(f'- `{flag}`')
        else:
            st.markdown('**觸發旗標**  \n無')
        st.markdown(
            '<div class="copilot-note">這個頁面提供調查線索，不輸出根因與'
            '維修決策。</div>',
            unsafe_allow_html=True,
        )
