'''Shared Plotly presentation contract tests.'''

import plotly.graph_objects as go

from industrial_alarm_copilot.presentation.charts import (
    PLOTLY_CHART_CONFIG,
    apply_chart_style,
)


def test_chart_config_keeps_recovery_controls_visible():
    assert PLOTLY_CHART_CONFIG['displayModeBar'] is True
    assert PLOTLY_CHART_CONFIG['doubleClick'] == 'reset'
    assert PLOTLY_CHART_CONFIG['displaylogo'] is False
    assert PLOTLY_CHART_CONFIG['modeBarButtonsToRemove'] == [
        'select2d',
        'lasso2d',
    ]


def test_apply_chart_style_preserves_data_and_adds_readable_surface():
    figure = go.Figure(go.Bar(x=['A'], y=[3]))

    styled = apply_chart_style(figure)

    assert styled is figure
    assert styled.data[0].y == (3,)
    assert styled.layout.paper_bgcolor == 'rgba(0, 0, 0, 0)'
    assert styled.layout.hoverlabel.bgcolor == '#0f172a'
    assert styled.layout.margin.t == 42
