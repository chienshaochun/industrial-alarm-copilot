# Industrial Alarm Copilot 技術設計

## 1. 設計目標

本技術設計將產品規格轉換成一個可在個人電腦上完成、測試與展示的 MVP。系統以單機 Python 應用為主，不導入微服務、訊息佇列或外部資料庫，避免讓部署複雜度超過產品本身的價值。

主要原則：

- 資料處理、模型與介面彼此分離。
- 沒有 LLM API Key 時，核心分析與摘要仍可操作。
- 所有時間相關功能都必須防止使用未來資料。
- 先建立 deterministic baseline，再加入機器學習與 LLM。
- 產生的資料、模型與評估結果都要能由指令重新建立。

## 2. 技術選擇

| 類別 | 選擇 | 用途與理由 |
| --- | --- | --- |
| 語言 | Python 3.12 | 與既有 Skills 一致，適合資料分析、模型與 Web App |
| 套件管理 | `pyproject.toml` | 集中管理專案資訊、依賴與工具設定 |
| Web UI | Streamlit | 快速建立互動式資料與模型展示介面 |
| 資料處理 | pandas、PyArrow | 444,834 筆資料可在單機處理，並以 Parquet 保存衍生資料 |
| 視覺化 | Plotly | 呈現可互動的警報時間軸與統計圖表 |
| 機器學習 | scikit-learn | 建立相似事件檢索、多標籤分類與可重現 baseline |
| 模型保存 | joblib | 保存 scikit-learn 模型與前處理物件 |
| 測試 | pytest | 單元、整合與端對端 smoke test |
| 程式品質 | Ruff | 統一 lint 與 formatting |
| CI | GitHub Actions | 在 Pull Request 執行測試與靜態檢查 |

第一版不使用資料庫。原始資料、Parquet 衍生資料、模型 artifact 與 JSON 評估報告即可支援唯讀 Demo；若未來加入使用者回饋或多次實驗管理，再評估 SQLite。

## 3. 系統架構

```text
Streamlit UI
    |
    v
Investigation Service（應用流程協調）
    |-- Incident Analyzer
    |-- Similar Episode Retriever
    |-- Alarm Forecaster
    `-- Summary Generator
            |
            |-- Deterministic Template（預設）
            `-- Optional LLM Adapter
    |
    v
Processed Parquet + Retrieval Index + Model Artifacts + Evaluation Reports
    ^
    |
Offline Data Pipeline
    ^
    |
ALPI data/raw/alarms.csv
```

Streamlit 僅負責輸入、狀態與畫面呈現，不能直接實作資料切分、事件建立或模型邏輯。應用流程由 `InvestigationService` 協調，使核心功能可以在沒有 Streamlit 的情況下接受測試與 CLI 呼叫。

## 4. 離線資料流程

```text
raw/alarms.csv
    |
    v
Schema validation + timestamp parsing + stable sorting
    |
    v
events.parquet
    |
    v
Chronological split metadata
    |
    v
Incident window construction
    |
    v
incidents.parquet + incident_events.parquet
    |
    |-- Feature extraction --> retrieval index
    `-- Future labels ------> forecasting model
                                |
                                v
                   metrics.json + model artifacts
```

所有衍生 artifact 都必須附帶：

- 原始 CSV 的 SHA-256；
- pipeline 設定摘要；
- 產生時間；
- train、validation、test 的時間邊界；
- 程式版本或 Git commit ID。

## 5. 建議目錄結構

```text
industrial-alarm-copilot/
|-- app.py
|-- pyproject.toml
|-- configs/
|   `-- default.toml
|-- data/
|   |-- raw/
|   |-- original/
|   `-- processed/
|-- artifacts/
|   |-- retrieval/
|   `-- forecasting/
|-- reports/
|-- src/industrial_alarm_copilot/
|   |-- data/
|   |   |-- loader.py
|   |   |-- schema.py
|   |   `-- splitter.py
|   |-- incidents/
|   |   |-- builder.py
|   |   `-- features.py
|   |-- retrieval/
|   |   |-- index.py
|   |   `-- evaluation.py
|   |-- forecasting/
|   |   |-- baselines.py
|   |   |-- model.py
|   |   `-- evaluation.py
|   |-- copilot/
|   |   |-- ports.py
|   |   `-- summary.py
|   |-- application/
|   |   `-- investigation.py
|   `-- presentation/
|       |-- navigation.py
|       `-- pages/
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- e2e/
`-- docs/
```

`data/processed/`、`artifacts/` 與執行產生的 `reports/` 預設不納入 Git；Repository 只保留建立它們所需的程式、設定與小型範例報告。

## 6. 核心資料契約

### 6.1 AlarmEvent

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `timestamp` | UTC-naive `datetime64[ns]` | 原始資料未提供時區，不自行推測時區 |
| `alarm_code` | category / string | 由原始 `alarm` 正規化而來 |
| `machine_id` | category / string | 由原始 `serial` 正規化而來 |
| `source_row` | integer | 對應原始資料列，便於追溯 |

### 6.2 Incident

| 欄位 | 說明 |
| --- | --- |
| `incident_id` | 由設備、時間範圍與設定版本產生的穩定 ID |
| `machine_id` | 設備識別碼 |
| `start_time`, `end_time` | 事件起訖時間 |
| `duration_seconds` | 事件持續時間 |
| `event_count` | 警報事件數量 |
| `distinct_alarm_count` | 不同警報代碼數量 |
| `alarm_counts` | 各警報代碼的計數表示 |

### 6.3 InvestigationResult

輸出分成三個不可混淆的區塊：

1. `observed_facts`：由選定事件直接計算的事實；
2. `retrieved_evidence`：歷史事件 ID、時間、相似度與後續警報；
3. `predictions`：模型、分數、預測時間窗與模型版本。

摘要產生器只能根據這個結構輸出內容，不可直接讀取未經選取的原始資料。

## 7. 事件建立策略

資料先依 `machine_id`、`timestamp`、`source_row` 穩定排序。同一設備的相鄰警報若時間差超過 `incident_gap_minutes`，就開始新的事件。

`incident_gap_minutes` 必須：

- 放在設定檔而非寫死在程式中；
- 由第 3 階段的事件間隔分布分析決定預設值；
- 在 artifact metadata 中留下紀錄；
- 透過測試涵蓋剛好等於門檻、跨設備與相同 timestamp 等邊界情況。

## 8. 相似事件檢索

第一版使用可解釋的特徵與 cosine similarity：

- 警報代碼 TF-IDF 或正規化計數；
- 事件持續時間；
- 警報總數與不同警報數；
- 可選的警報順序轉移特徵。

以 scikit-learn `NearestNeighbors` 建立檢索器。每次查詢先依時間過濾候選事件，候選事件的 `end_time` 必須早於查詢事件的 `start_time`，再進行 Top-K 排序。

檢索評估的 relevant 定義為：歷史事件後續時間窗內的警報集合，與查詢事件真實未來警報集合達到設定的重疊條件。評估輸出 Hit@K、Precision@K、Recall@K、MRR 與 NDCG@K。

## 9. 後續警報預測

輸入是觀察時間窗或事件的歷史特徵，輸出是未來 `forecast_horizon_minutes` 內各警報是否出現的多標籤向量。

依序建立並比較：

1. 全域最常見警報 baseline；
2. 每台設備的歷史頻率 baseline；
3. 依目前警報條件化的轉移 baseline；
4. scikit-learn One-vs-Rest 線性模型。

主要指標為 Precision@K、Recall@K 與 macro F1，並一併輸出每個警報的 support。資料切分禁止使用隨機 shuffle。

## 10. 時間切分與防止洩漏

主要評估採每台設備內的 chronological split，預設比例可由設定檔調整，初始候選為 70% train、15% validation、15% test。

強制規則：

- scaler、TF-IDF 與模型只能在 train fit；
- validation 用於設定選擇，test 僅用於最終報告；
- 查詢 test 事件時，檢索候選不得來自該事件之後；
- 產生某事件的特徵時，不得使用其預測時間窗內的事件；
- split 邊界與查詢 cutoff 必須出現在評估報告。

跨設備 cold-start evaluation 可作為第二階段擴充，不列入第一版 MVP 驗收。

## 11. Copilot 摘要層

摘要層定義共同介面，提供兩種實作：

- `TemplateSummaryGenerator`：不需 API Key，依結構化結果產生固定格式摘要；
- `LLMSummaryGenerator`：選配，將相同結構化證據轉換成較自然的說明。

LLM 輸出需通過結構驗證，並遵守：

- 不為匿名警報代碼補充不存在的語意；
- 不宣稱模型預測是事實；
- 每項歷史推論都引用 incident ID；
- 證據不足時輸出限制說明；
- LLM 失敗時自動回退到 deterministic template。

## 12. Streamlit 介面

採用 `st.navigation` 建立下列頁面：

1. **資料總覽**：資料品質、設備與警報分布；
2. **事件調查**：設備／時間篩選、時間軸與事件清單；
3. **相似事件**：檢索結果、證據與後續警報比較；
4. **預測與摘要**：Top-K 預測、限制說明與 Copilot 摘要；
5. **模型評估**：retrieval 與 forecasting metrics。

使用 `st.cache_data` 快取不可變的資料轉換，使用 `st.cache_resource` 載入模型與檢索器，使用 `st.session_state` 保存使用者目前選定的設備與事件。快取函式仍放在 presentation adapter，不滲入 domain logic。

## 13. 測試策略

### 單元測試

- schema validation 與錯誤資料；
- timestamp 排序與相同時間處理；
- incident gap 邊界；
- 時間切分與候選過濾；
- Top-K 與各項 metrics；
- 摘要中的證據引用與 fallback。

### 整合測試

- raw CSV sample 到 incidents artifact；
- incidents 到 retrieval results；
- train split 到 forecasting predictions；
- investigation service 的完整結構化輸出。

### 端對端測試

- 使用固定小型 fixture 執行完整 pipeline；
- 啟動 Streamlit smoke test；
- 不需要網路與 LLM API Key 即可通過主要測試。

## 14. 執行與可重現性

預計提供以下命令介面：

```text
python -m industrial_alarm_copilot prepare-data
python -m industrial_alarm_copilot build-retrieval
python -m industrial_alarm_copilot train-forecast
python -m industrial_alarm_copilot evaluate
streamlit run app.py
```

實際 CLI 參數會在各功能實作階段定義，但所有命令都應支援明確設定檔與輸出目錄，且重複執行相同版本與設定時可重現結果。

## 15. 已排除的替代方案

- **FastAPI + 獨立前端**：對目前唯讀單機 Demo 過重，若未來需要外部 API 再拆分。
- **向量資料庫**：事件量在單機可處理，第一版使用 scikit-learn index 即可。
- **SQLite**：目前沒有使用者帳號或回饋寫入需求，先以 Parquet 與 JSON artifact 為主。
- **深度學習模型**：先用可解釋 baseline 建立可信評估，再決定是否有增加複雜度的價值。
- **Agent 自主維修建議**：資料沒有警報語意與 SOP，不具備可靠依據。

## 16. 第 2 階段技術設計驗收

- 技術選擇能支援產品規格中的每項 MVP 功能；
- 資料流與模組責任明確；
- 防止時間洩漏的規則可以轉換成自動化測試；
- 沒有 LLM 時仍存在完整操作路徑；
- 目錄結構足以開始第 3 階段，但尚未過度設計部署架構。

## 17. 官方技術參考

- Streamlit multipage apps：https://docs.streamlit.io/develop/concepts/multipage-apps/overview
- Streamlit caching and state：https://docs.streamlit.io/develop/api-reference/caching-and-state
- scikit-learn OneVsRestClassifier：https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
- Plotly Python：https://plotly.com/python/

