'''Model evaluation and limitations page.'''

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from industrial_alarm_copilot.presentation.runtime import load_evaluation_data
from industrial_alarm_copilot.presentation.data import (
    resolve_artifact_directory,
    resolve_project_root,
)


PROJECT_ROOT = resolve_project_root()


def _percentage(value: object) -> str:
    return f'{float(value):.1%}'


def _evaluation_paths() -> dict[str, Path]:
    artifact_directory = resolve_artifact_directory(PROJECT_ROOT)
    return {
        'retrieval': artifact_directory / 'retrieval_test_results.csv',
        'forecast': artifact_directory / 'forecast_test_results.csv',
        'support': artifact_directory / 'forecast_test_support_groups.csv',
    }


def _render_retrieval_metrics(metrics: dict[str, object]) -> None:
    st.markdown('### Similar-episode retrieval｜鎖定 test 結果')
    columns = st.columns(5)
    columns[0].metric('Hit@5', _percentage(metrics['mean_hit_at_k']))
    columns[1].metric(
        'Precision@5', _percentage(metrics['mean_precision_at_k'])
    )
    columns[2].metric('Recall@5', _percentage(metrics['mean_recall_at_k']))
    columns[3].metric(
        'Precision lift', f"{float(metrics['mean_precision_lift_at_k']):.2f}×"
    )
    columns[4].metric('NDCG@5', f"{float(metrics['mean_ndcg_at_k']):.3f}")

    values = pd.DataFrame(
        {
            'metric': ['Hit@5', 'Precision@5', 'MRR', 'NDCG@5'],
            'value': [
                metrics['mean_hit_at_k'],
                metrics['mean_precision_at_k'],
                metrics['mean_reciprocal_rank'],
                metrics['mean_ndcg_at_k'],
            ],
        }
    )
    chart = px.bar(
        values,
        x='metric',
        y='value',
        text_auto='.1%',
        range_y=[0, 1],
        color_discrete_sequence=['#2878b5'],
    )
    chart.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(chart, width='stretch')
    st.markdown(
        '<div class="copilot-note">Recall@5 很低不是隱藏掉的失敗：每個 query '
        '可有大量「未來結果相似」的歷史候選，而 UI 只展示 5 筆。Top-5 的目的'
        '是提供少量可讀證據；Precision lift 用來比較它是否優於隨機抽取。</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"設定：{metrics['feature_version']}｜{metrics['candidate_policy']}｜"
        f"horizon {float(metrics['future_horizon_hours']):g}h｜"
        f"threshold {float(metrics['relevance_threshold']):.1f}｜"
        f"test queries {int(metrics['query_count']):,}"
    )


def _render_forecast_metrics(
    metrics: dict[str, object],
    support_groups: pd.DataFrame,
) -> None:
    st.markdown('### Next-alarm forecasting｜鎖定 test 結果')
    columns = st.columns(5)
    columns[0].metric('Hit@5', _percentage(metrics['mean_hit_at_k']))
    columns[1].metric(
        'Precision@5', _percentage(metrics['mean_precision_at_k'])
    )
    columns[2].metric('Recall@5', _percentage(metrics['mean_recall_at_k']))
    columns[3].metric('Micro F1@5', _percentage(metrics['micro_f1_at_k']))
    columns[4].metric('Macro F1@5', _percentage(metrics['macro_f1_at_k']))

    chart = px.bar(
        support_groups,
        x='support_group',
        y=['micro_f1_at_k', 'macro_f1_at_k'],
        barmode='group',
        labels={
            'support_group': 'Train support 群組',
            'value': 'F1@5',
            'variable': '指標',
        },
        color_discrete_sequence=['#2878b5', '#e58b35'],
    )
    chart.update_layout(
        yaxis_tickformat='.0%',
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(chart, width='stretch')
    st.dataframe(
        support_groups,
        width='stretch',
        hide_index=True,
        column_config={
            'support_group': 'Train support 群組',
            'label_count': 'Alarm code 數',
            'evaluated_label_count': 'Test 有出現的 code 數',
            'evaluation_positive_count': '實際正例',
            'predicted_positive_count': '預測正例',
            'true_positive_count': '命中正例',
            'micro_precision_at_k': st.column_config.NumberColumn(
                'Micro precision', format='percent'
            ),
            'micro_recall_at_k': st.column_config.NumberColumn(
                'Micro recall', format='percent'
            ),
            'micro_f1_at_k': st.column_config.NumberColumn(
                'Micro F1', format='percent'
            ),
            'macro_f1_at_k': st.column_config.NumberColumn(
                'Macro F1', format='percent'
            ),
        },
    )
    st.warning(
        'Rare Alarm 在 test 的 F1 仍為 0；整體 Hit@5 高，不代表長尾 Alarm '
        '已被解決。這是目前模型最重要的限制。'
    )
    st.caption(
        f"模型：{metrics['model_version']}｜Top-{int(metrics['top_k'])}｜"
        f"test episodes {int(metrics['episode_count']):,}｜"
        f"outcome coverage {float(metrics['outcome_coverage']):.1%}"
    )


def render_evaluation_page() -> None:
    st.markdown('<div class="copilot-kicker">Honest Model Card</div>', unsafe_allow_html=True)
    st.title('模型評估')
    st.caption('只呈現鎖定設定在 test split 的一次性結果，同時揭露失敗模式。')
    paths = _evaluation_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        st.error('缺少評估 artifacts：' + ', '.join(missing))
        st.code(
            'python -m industrial_alarm_copilot test-retrieval\n'
            'python -m industrial_alarm_copilot test-forecast',
            language='powershell',
        )
        return
    evaluation = load_evaluation_data(
        str(paths['retrieval']),
        str(paths['forecast']),
        str(paths['support']),
    )
    _render_retrieval_metrics(evaluation.retrieval)
    st.divider()
    _render_forecast_metrics(
        evaluation.forecasting,
        evaluation.support_groups,
    )
    with st.expander('如何閱讀這些指標？'):
        st.markdown(
            '- **Hit@5**：Top-5 中至少命中一個相關項目的 episode 比例。\n'
            '- **Precision@5**：展示的 5 個項目中，有多少是相關的。\n'
            '- **Recall@5**：所有相關項目中，被這 5 個位置找回多少。\n'
            '- **Micro F1**：常見 Alarm 影響較大；**Macro F1** 讓每個 '
            'Alarm code 權重相同，因此更能暴露長尾問題。'
        )
