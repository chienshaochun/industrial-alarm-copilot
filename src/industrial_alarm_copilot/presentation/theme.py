'''Shared visual tokens for a restrained industrial dashboard.'''

import streamlit as st


def apply_app_theme() -> None:
    st.markdown(
        '''
        <style>
        :root {
          --copilot-navy: #15243a;
          --copilot-blue: #2878b5;
          --copilot-orange: #e58b35;
          --copilot-muted: #64748b;
        }
        [data-testid="stMetric"] {
          border: 1px solid rgba(100, 116, 139, 0.24);
          border-radius: 12px;
          padding: 14px 16px;
          background: rgba(248, 250, 252, 0.58);
        }
        [data-testid="stMetricValue"] { color: var(--copilot-navy); }
        .copilot-kicker {
          color: var(--copilot-blue);
          font-weight: 700;
          letter-spacing: .08em;
          text-transform: uppercase;
          font-size: .78rem;
        }
        .copilot-note {
          border-left: 4px solid var(--copilot-orange);
          background: rgba(229, 139, 53, .08);
          padding: .8rem 1rem;
          border-radius: 0 8px 8px 0;
        }
        </style>
        ''',
        unsafe_allow_html=True,
    )
