# Industrial Alarm Copilot

這是一個以公開且來自真實工業環境的 **Alarm Logs in Packaging Industry
(ALPI)** 資料集為基礎，開發的 AI 輔助工業警報分析專案。

本專案與 workspace 內既有的 Codex Skills 分開維護。最終產品會在一個小型互動式應用中，整合可重現的警報分析、相似事件檢索、後續警報預測、具歷史證據的 AI 摘要，以及可量化的評估流程。

## 目前進度

第 1 至第 3 階段已完成本機驗收：Repository 與資料集準備、產品與技術設計，以及可重現的資料分析與處理管線。下一階段將使用 30 分鐘 incident gap 建立警報事件時間窗與統計 baseline。

目前已實作：

- 嚴格驗證 ALPI 欄位、缺值、timestamp 與識別碼；
- 保留原始列追溯資訊與完全重複事件標記；
- 在同一設備內計算相鄰警報時間間隔；
- 每台設備分開執行 70%／15%／15% chronological split；
- 產生 Zstandard 壓縮的 `events.parquet` 與可追溯 metadata；
- 以完整 444,834 筆 ALPI 資料執行自動化契約驗收。

## 資料集摘要

納入版本控制的原始資料位於 [`data/raw/alarms.csv`](data/raw/alarms.csv)。

| 項目 | 數值 |
| --- | ---: |
| 資料筆數 | 444,834 |
| 設備數量 | 20 |
| 警報代碼數量 | 154 |
| 欄位 | `timestamp`, `alarm`, `serial` |
| 收集期間 | 2019-02-21 至 2020-06-17 |
| 原始 CSV SHA-256 | `53bd4414a6fb5b6875a9535f1be622dfb2dfba69de407d071a52d0d304160d1a` |

這些資料是真實工業設備產生的警報序列，警報類別分布高度不平衡，適合用來實作時間序列驗證、罕見警報評估、多標籤預測與跨設備泛化實驗。

## 快速開始

本專案使用 Python 3.12。第一次建立環境：

```powershell
conda create --name industrial-alarm-copilot python=3.12
conda activate industrial-alarm-copilot
python -m pip install --editable ".[dev]"
```

執行所有單元與整合測試：

```powershell
python -m pytest
```

由原始 CSV 重新產生 processed artifacts：

```powershell
python -m industrial_alarm_copilot prepare-data
```

預設輸出：

```text
data/processed/events.parquet
data/processed/events.metadata.json
```

衍生 artifacts 不納入 Git。metadata 會記錄原始 CSV SHA-256、pipeline 設定、產生時間、Git commit、欄位型別及 split 時間邊界。

## 第 3 階段主要發現

- 警報分布高度不平衡：最常見代碼占 20.40%，前十種占 88.75%；
- 前五台設備占 53.75% 的事件，原始事件量不能直接解讀為故障率；
- 83 個完全相同事件群組涉及 166 列，完整事件視圖全部保留；
- 同設備相鄰警報的中位 gap 為 0.98 分鐘，90% 分位數為 25.36 分鐘；
- 第一版 incident gap 設為 30 分鐘，並保留 15／30／60 分鐘敏感度比較；
- processed events 共 444,834 列、9 欄，本次產生的 Parquet 約 9.34 MiB。

時間範圍、事件數與 gap 都只描述警報記錄，不能直接視為設備運轉、停機、故障持續或修復時間。

## 預計實作範圍

1. 探索警報時間軸與各設備的資料分布。
2. 將時間上相關的警報組合為事件時間窗（incident window）。
3. 檢索相似的歷史警報事件。
4. 預測未來時間窗內可能出現的後續警報。
5. 產生引用歷史事件的證據型摘要。
6. 在避免時間洩漏的前提下評估預測與檢索品質。

資料集未提供警報說明、實際根因或維修程序，因此應用不會虛構操作建議，並會明確區分觀察到的事實、模型預測與 AI 產生的摘要。

## Repository 結構

```text
industrial-alarm-copilot/
|-- configs/default.toml   # incident gap 與 split 設定
|-- data/
|   |-- original/alpi-v1/  # 發布者原始文件與前處理程式，不修改
|   |-- processed/         # 本機衍生 artifacts，不納入 Git
|   `-- raw/alarms.csv     # 原始 ALPI 警報事件
|-- docs/                  # 規格、設計、EDA 與分析決策
|-- src/industrial_alarm_copilot/
|   `-- data/              # loader、validation、EDA、gap、split、pipeline
|-- tests/
|   |-- unit/
|   `-- integration/
|-- pyproject.toml         # Python 依賴與 pytest 設定
|-- LICENSE                # 專案自行開發內容的 MIT License
|-- NOTICE.md              # 第三方資料來源與授權界線
`-- README.md
```

## 專案文件

- [產品規格](docs/PRODUCT_SPEC.md)
- [技術設計](docs/TECHNICAL_DESIGN.md)
- [資料契約](docs/DATA_CONTRACT.md)
- [探索性資料分析](docs/EDA_REPORT.md)
- [Incident gap 分析](docs/INCIDENT_GAP_ANALYSIS.md)
- [時間序列切分策略](docs/SPLIT_STRATEGY.md)
- [第 3 階段驗收](docs/STAGE_3_ACCEPTANCE.md)

## 資料集引用

> Dalle Pezze, Davide; Tosato, Diego; Masiero, Chiara; Susto, Gian
> Antonio; Beghi, Alessandro (2021), "ALARM LOGS IN PACKAGING INDUSTRY
> (ALPI)", Mendeley Data, V1, doi: 10.17632/4nhx2x67cd.1.

- 資料集頁面：https://data.mendeley.com/datasets/4nhx2x67cd/1
- DOI：https://doi.org/10.17632/4nhx2x67cd.1
- 資料集授權：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

授權界線與來源細節請參閱 [`NOTICE.md`](NOTICE.md) 與
[`data/README.md`](data/README.md)。

## 授權

本專案自行開發的程式碼與文件採用 MIT License。ALPI 資料集與發布者提供的檔案維持原本的 CC BY 4.0 授權及姓名標示要求。

