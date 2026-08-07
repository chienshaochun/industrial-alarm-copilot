# 資料說明

## 資料來源

本專案使用 Alarm Logs in Packaging Industry（ALPI）version 1：

> Dalle Pezze, Davide; Tosato, Diego; Masiero, Chiara; Susto, Gian
> Antonio; Beghi, Alessandro (2021), "ALARM LOGS IN PACKAGING INDUSTRY
> (ALPI)", Mendeley Data, V1, doi: 10.17632/4nhx2x67cd.1.

- 資料集頁面：https://data.mendeley.com/datasets/4nhx2x67cd/1
- DOI：https://doi.org/10.17632/4nhx2x67cd.1
- 授權：CC BY 4.0
- 下載日期：2026-08-07

## 納入的檔案

- `raw/alarms.csv`：未修改的原始事件層級警報紀錄
- `original/alpi-v1/dataset.py`：發布者提供的資料前處理程式
- `original/alpi-v1/readme.md`：發布者提供的資料集說明
- `processed/`：本機產生的衍生資料，不納入 Git

發布者預先產生的 JSON、NumPy 與 pickle 檔案未納入本專案，因為它們與原始資料重複，並可使用發布者提供的程式重新產生。

## 原始資料欄位

| 欄位 | 說明 |
| --- | --- |
| `timestamp` | 警報發生時間 |
| `alarm` | 匿名化的警報代碼 |
| `serial` | 匿名化的設備識別碼 |

## 完整性驗證

`raw/alarms.csv` SHA-256：

```text
53bd4414a6fb5b6875a9535f1be622dfb2dfba69de407d071a52d0d304160d1a
```

此資料集屬於第三方內容，不會重新授權為本專案的 MIT License。詳細界線請參閱 Repository 根目錄的 `NOTICE.md`。
