# Industrial Alarm Copilot

An AI-assisted industrial alarm analysis project built on the public,
real-world **Alarm Logs in Packaging Industry (ALPI)** dataset.

The project is intentionally separate from the supporting Codex skills in this
workspace. It will combine deterministic alarm analysis, similar-episode
retrieval, next-alarm forecasting, evidence-backed AI summaries, and measurable
evaluation in a small interactive application.

## Current status

Repository and dataset bootstrap complete. Application implementation is the
next milestone.

## Dataset snapshot

The checked-in raw dataset is located at [`data/raw/alarms.csv`](data/raw/alarms.csv).

| Property | Value |
| --- | ---: |
| Rows | 444,834 |
| Machines | 20 |
| Alarm codes | 154 |
| Columns | `timestamp`, `alarm`, `serial` |
| Collection period | 2019-02-21 to 2020-06-17 |
| Raw CSV SHA-256 | `53bd4414a6fb5b6875a9535f1be622dfb2dfba69de407d071a52d0d304160d1a` |

The data are real industrial alarm sequences and have a highly imbalanced
alarm-code distribution. This makes the project suitable for temporal
validation, rare-alarm evaluation, multi-label forecasting, and cross-machine
generalization experiments.

## Planned product slice

1. Explore alarm timelines and machine-level distributions.
2. Build incident windows from temporally related alarm events.
3. Retrieve similar historical alarm episodes.
4. Forecast the next alarms in a future time window.
5. Generate evidence-backed summaries that cite historical episodes.
6. Evaluate forecasting and retrieval quality without temporal leakage.

The dataset does not provide alarm descriptions, root causes, or maintenance
procedures. The application will therefore avoid inventing operational advice
and will clearly separate observed evidence, model predictions, and AI-generated
summaries.

## Repository layout

```text
industrial-alarm-copilot/
|-- data/
|   |-- original/alpi-v1/  # Publisher README and preprocessing script
|   |-- processed/         # Generated derivatives; ignored by Git
|   `-- raw/alarms.csv     # Original ALPI alarm events
|-- LICENSE                # MIT license for project-authored code
|-- NOTICE.md              # Dataset attribution and license boundary
`-- README.md
```

## Dataset attribution

> Dalle Pezze, Davide; Tosato, Diego; Masiero, Chiara; Susto, Gian
> Antonio; Beghi, Alessandro (2021), "ALARM LOGS IN PACKAGING INDUSTRY
> (ALPI)", Mendeley Data, V1, doi: 10.17632/4nhx2x67cd.1.

- Dataset record: https://data.mendeley.com/datasets/4nhx2x67cd/1
- DOI: https://doi.org/10.17632/4nhx2x67cd.1
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

See [`NOTICE.md`](NOTICE.md) and [`data/README.md`](data/README.md) for the
license boundary and provenance details.

## License

Project-authored source code and documentation are released under the MIT
License. The ALPI dataset and publisher-provided files retain their original
CC BY 4.0 license and attribution requirements.

