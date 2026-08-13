# Industrial Alarm Copilot

這是一個以公開且來自真實工業環境的 **Alarm Logs in Packaging Industry
(ALPI)** 資料集為基礎，開發的 AI 輔助工業警報分析專案。

本專案與 workspace 內既有的 Codex Skills 分開維護。最終產品會在一個小型互動式應用中，整合可重現的警報分析、相似事件檢索、後續警報預測、具歷史證據的 AI 摘要，以及可量化的評估流程。

## 目前進度

第 1 至第 5 階段已完成本機驗收：Repository 與資料集準備、產品與技術設計、可重現的資料處理管線、derived alarm episode、deterministic statistical baseline，以及時間安全的相似歷史事件檢索。下一階段將建立後續警報預測模型。

目前已實作：

- 嚴格驗證 ALPI 欄位、缺值、timestamp 與識別碼；
- 保留原始列追溯資訊與完全重複事件標記；
- 在同一設備內計算相鄰警報時間間隔；
- 每台設備分開執行 70%／15%／15% chronological split；
- 產生 Zstandard 壓縮的 `events.parquet` 與可追溯 metadata；
- 以 30 分鐘 gap 將 444,834 筆 events 建立為 38,153 個 derived episodes；
- 產生穩定 incident ID、event mapping、episode 時間與 alarm 組成摘要；
- 以 train-only global／machine P95 建立三種 upper-tail flags；
- 設備歷史不足時使用全域 fallback，避免不穩定的專屬門檻；
- 使用 train-fitted alarm TF-IDF 與 episode shape 建立 exact cosine Top-5 檢索；
- 候選 episode 與其 6 小時 outcome 都必須在 query 開始前完整可得；
- 以 validation 的 18 組實驗凍結 feature、horizon 與 relevance threshold；
- 以未參與選型的 test episodes 完成一次最終檢索驗收；
- 以 85 項測試驗收目前的完整資料、incident 與 retrieval 流程。

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

再由 processed events 建立 episode 與 baseline artifacts：

    python -m industrial_alarm_copilot prepare-incidents

額外輸出：

    data/processed/incidents.parquet
    data/processed/incident_events.parquet
    data/processed/incident_baselines.json

衍生 artifacts 不納入 Git。metadata 會記錄來源 SHA-256、pipeline 設定、產生時間、Git commit、欄位型別、split 統計與 baseline 門檻。

重現 validation 實驗或執行已凍結的 test 設定：

```powershell
python -m industrial_alarm_copilot validate-retrieval
python -m industrial_alarm_copilot test-retrieval
```

Retrieval CSV 與 metadata 同樣寫入 `data/processed/`，不納入 Git。正式 test 指令只能讀取設定檔中的凍結值，不接受臨時指定 feature、horizon、threshold 或 Top-K。

## 第 3 階段主要發現

- 警報分布高度不平衡：最常見代碼占 20.40%，前十種占 88.75%；
- 前五台設備占 53.75% 的事件，原始事件量不能直接解讀為故障率；
- 83 個完全相同事件群組涉及 166 列，完整事件視圖全部保留；
- 同設備相鄰警報的中位 gap 為 0.98 分鐘，90% 分位數為 25.36 分鐘；
- 第一版 incident gap 設為 30 分鐘，並保留 15／30／60 分鐘敏感度比較；
- processed events 共 444,834 列、9 欄，本次產生的 Parquet 約 9.34 MiB。

時間範圍、事件數與 gap 都只描述警報記錄，不能直接視為設備運轉、停機、故障持續或修復時間。

## 第 4 階段主要結果

- 30 分鐘正式設定產生 38,153 個 derived episodes；
- train／validation／test 分別為 25,936／5,853／6,364 個 episodes；
- 15／30／60 分鐘敏感度分析分別產生 68,093／38,153／19,039 個 episodes；
- 全域 train P95 為 42 筆 events、9,040.763 秒跨度與 6 種 alarm codes；
- 19 台設備使用專屬 train baseline，設備 19 因只有 11 個 train episodes 而使用全域 fallback；
- 任一 upper-tail flag 比例為 train 7.90%、validation 7.47%、test 6.82%。

Derived episode 與 upper-tail flag 都是統計分析單位，不是真實故障、危險或維修標籤。

## 第 5 階段主要結果

- 使用 `alarm_plus_shape_v1`，結合 train-fitted alarm TF-IDF 與 episode event count、duration、alarm diversity；
- 固定 Top-5、`expanding_history`、6 小時 future horizon 與 Jaccard threshold 0.3；
- Test evaluation coverage 為 86.61%，Hit@5 為 77.98%；
- Test Precision@5 為 38.94%，隨機基準為 18.68%，平均 precision lift 為 2.45 倍；
- Test MRR 為 0.547，NDCG@5 為 0.392；
- Validation 與 test 表現接近，未觀察到明顯的時間外推崩落。

Recall@5 的絕對值為 0.000367，主因是 proxy relevance 會產生大量 relevant 歷史候選，而產品固定只回傳五筆；其理論最大 Recall@5 也只有 0.003059。因此本階段同時報告 Precision@5、Hit@5、precision lift、MRR、NDCG@5 與 recall efficiency，不單獨以 Recall@5 判定檢索成敗。

這些 relevance labels 來自後續匿名 alarm code 集合的 Jaccard 重疊，不是真實相似故障或根因標註。Test 結果只支持 ALPI 內的時間泛化，不代表可直接泛化到其他工廠。

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
|-- configs/default.toml   # incident gap、baseline 與 split 設定
|-- data/
|   |-- original/alpi-v1/  # 發布者原始文件與前處理程式，不修改
|   |-- processed/         # 本機衍生 artifacts，不納入 Git
|   `-- raw/alarms.csv     # 原始 ALPI 警報事件
|-- docs/                  # 規格、設計、EDA 與分析決策
|-- src/industrial_alarm_copilot/
|   |-- incidents/         # episode、mapping、baseline 與 artifact pipeline
|   |-- retrieval/         # 特徵、時間安全檢索、評估與 artifacts
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
- [Derived Alarm Episode 契約](docs/INCIDENT_CONTRACT.md)
- [Incident gap 敏感度分析](docs/INCIDENT_SENSITIVITY_ANALYSIS.md)
- [Incident 統計基線分析](docs/INCIDENT_BASELINE_ANALYSIS.md)
- [第 4 階段驗收](docs/STAGE_4_ACCEPTANCE.md)
- [相似 Episode 檢索契約](docs/RETRIEVAL_CONTRACT.md)
- [Retrieval validation 分析](docs/RETRIEVAL_VALIDATION_ANALYSIS.md)
- [Retrieval test 分析](docs/RETRIEVAL_TEST_ANALYSIS.md)
- [第 5 階段驗收](docs/STAGE_5_ACCEPTANCE.md)

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

