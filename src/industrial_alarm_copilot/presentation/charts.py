'''Shared Plotly interaction and presentation settings.'''

from typing import Final

import streamlit as st


PLOTLY_CHART_CONFIG: Final[dict[str, object]] = {
    'displayModeBar': True,
    'displaylogo': False,
    'doubleClick': 'reset',
    'scrollZoom': False,
    'responsive': True,
    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'industrial-alarm-copilot-chart',
        'scale': 2,
    },
}


def apply_chart_style(figure):
    '''Apply a readable industrial dashboard style to a Plotly figure.'''
    figure.update_layout(
        font={'color': '#334155', 'family': 'Inter, Segoe UI, sans-serif'},
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(248, 250, 252, 0.72)',
        hoverlabel={
            'bgcolor': '#0f172a',
            'bordercolor': '#38bdf8',
            'font': {'color': '#f8fafc'},
        },
        modebar={
            'bgcolor': 'rgba(255, 255, 255, 0.92)',
            'color': '#64748b',
            'activecolor': '#2878b5',
        },
        margin={'l': 8, 'r': 8, 't': 42, 'b': 8},
    )
    figure.update_xaxes(
        automargin=True,
        gridcolor='rgba(148, 163, 184, 0.18)',
        linecolor='rgba(100, 116, 139, 0.30)',
    )
    figure.update_yaxes(
        automargin=True,
        gridcolor='rgba(148, 163, 184, 0.18)',
        linecolor='rgba(100, 116, 139, 0.30)',
    )
    return figure


def render_interactive_chart(figure, *, key: str) -> None:
    '''Render a chart with visible recovery controls and usage guidance.'''
    st.plotly_chart(
        apply_chart_style(figure),
        width='stretch',
        key=key,
        config=PLOTLY_CHART_CONFIG,
    )
    st.caption(
        '圖表操作：拖曳可縮放；雙擊圖面或點右上角的重設座標軸按鈕，'
        '即可回到原始範圍。'
    )
