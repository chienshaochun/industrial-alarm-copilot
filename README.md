# Industrial Alarm Copilot

這是一個以公開且來自真實工業環境的 **Alarm Logs in Packaging Industry
(ALPI)** 資料集為基礎，開發的 AI 輔助工業警報分析專案。

本專案與 workspace 內既有的 Codex Skills 分開維護。最終產品會在一個小型互動式應用中，整合可重現的警報分析、相似事件檢索、後續警報預測、具歷史證據的 AI 摘要，以及可量化的評估流程。

## 目前進度

Repository 與資料集初始化已完成，目前正在進行產品需求與技術設計。

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
|-- data/
|   |-- original/alpi-v1/  # 發布者的 README 與前處理程式
|   |-- processed/         # 本機產生的衍生資料，不納入 Git
|   `-- raw/alarms.csv     # 原始 ALPI 警報事件
|-- docs/                  # 產品規格與專案計畫
|-- LICENSE                # 專案自行開發內容的 MIT License
|-- NOTICE.md              # 第三方資料來源與授權界線
`-- README.md
```

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

