# Project Plan

The Industrial Alarm Copilot project is divided into eight independently
verifiable stages. Each stage should finish with tests or other validation and
one focused Git commit.

## Stages

1. **Repository and dataset foundation** — initialize the repository, include
   the ALPI raw dataset, and document provenance and licensing. **Complete.**
2. **Product and technical design** — define the target user, MVP workflow,
   architecture, non-goals, and acceptance criteria.
3. **Data analysis and pipeline** — profile and validate the dataset, prevent
   temporal leakage, and build reproducible preprocessing.
4. **Incident-analysis baseline** — group related alarm events into windows and
   implement deterministic statistical baselines.
5. **Similar-episode retrieval** — retrieve historical alarm windows, cite the
   evidence, and measure retrieval quality.
6. **Next-alarm forecasting** — train baseline forecasting models and evaluate
   common and rare alarms separately.
7. **AI copilot and web application** — add evidence-constrained summaries and
   an interactive Streamlit interface.
8. **Engineering and portfolio delivery** — add automated tests, CI, Docker,
   deployment instructions, screenshots, and a short product demo.

## Delivery rules

- Use chronological splits for time-dependent evaluation.
- Keep observed facts, model predictions, and generated summaries distinct.
- Establish deterministic baselines before adding an LLM.
- Cite retrieved historical episodes in user-facing AI summaries.
- Complete validation and a focused commit before starting the next stage.

