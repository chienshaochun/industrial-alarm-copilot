# Data

## Source

This project uses version 1 of the Alarm Logs in Packaging Industry (ALPI)
dataset:

> Dalle Pezze, Davide; Tosato, Diego; Masiero, Chiara; Susto, Gian
> Antonio; Beghi, Alessandro (2021), "ALARM LOGS IN PACKAGING INDUSTRY
> (ALPI)", Mendeley Data, V1, doi: 10.17632/4nhx2x67cd.1.

- Record: https://data.mendeley.com/datasets/4nhx2x67cd/1
- DOI: https://doi.org/10.17632/4nhx2x67cd.1
- License: CC BY 4.0
- Downloaded: 2026-08-07

## Included files

- `raw/alarms.csv`: original event-level alarm log extracted without changes
- `original/alpi-v1/dataset.py`: publisher-provided preprocessing code
- `original/alpi-v1/readme.md`: publisher-provided dataset documentation
- `processed/`: local generated derivatives; ignored by Git

The publisher's precomputed JSON, NumPy, and pickle derivatives are not
included because they duplicate the raw data and can be regenerated with the
provided preprocessing script.

## Raw schema

| Column | Meaning |
| --- | --- |
| `timestamp` | Alarm occurrence time |
| `alarm` | Anonymized alarm code |
| `serial` | Anonymized machine identifier |

## Integrity

`raw/alarms.csv` SHA-256:

```text
53bd4414a6fb5b6875a9535f1be622dfb2dfba69de407d071a52d0d304160d1a
```

The dataset is third-party content and is not relicensed under the project's
MIT License. See the repository-level `NOTICE.md`.
