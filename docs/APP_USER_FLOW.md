# Streamlit 應用使用流程契約

## 1. 文件目的

本文件定義 Industrial Alarm Copilot 第 7 階段的頁面責任、操作流程、跨頁狀態與資訊邊界。Streamlit 介面只負責選擇、呈現與互動，不直接實作資料處理、檢索、預測或摘要規則。

MVP 必須讓設備可靠度或維護工程師在 90 秒內完成一次歷史 Episode 調查，且不需要資料庫、網路或 LLM API Key。

## 2. 三頁式資訊架構

側邊導覽固定包含下列三頁：

1. `資料總覽`
2. `事件調查工作台`
3. `模型評估`

事件調查、相似事件、未來預測與 Copilot 摘要集中在同一個工作台，避免切換頁面時失去目前選定的設備與 Episode。

## 3. 資料總覽

### 3.1 使用者問題

資料包含什麼、涵蓋多久、哪些設備與 Alarm 最活躍，以及資料是否存在明顯長尾不平衡？

### 3.2 顯示內容

- 事件紀錄數；
- 設備數；
- Alarm code 數；
- 資料起訖時間；
- 設備與日期篩選器；
- 每月 Alarm 活動趨勢；
- 各設備事件量；
- Top Alarm Codes；
- Common、Medium、Rare label 分布；
- 資料品質驗證狀態；
- Alarm 與設備代碼匿名化聲明。

### 3.3 邊界

本頁只呈現描述性統計，不顯示單一 Episode 的相似事件、未來預測或 Copilot 結論。

## 4. 事件調查工作台

### 4.1 使用者問題

選定的 Episode 實際發生了什麼、歷史上是否出現相似模式，以及接下來六小時可能出現哪些 Alarm？

### 4.2 選擇流程

1. 選擇 `machine_id`；
2. 從該設備的 Episode 清單選擇 `incident_id`；
3. 顯示 artifact 載入與相容性狀態；
4. 所有頁籤共用相同的選定 Episode。

Episode 選項至少顯示開始時間、結束時間與事件數，避免使用者只能辨識雜湊 ID。

### 4.3 頂部摘要卡

- Event count；
- Distinct alarm count；
- Duration；
- Upper-tail 統計狀態。

Upper-tail 只能描述相對 train baseline 的統計位置，不得命名為故障、危險或異常根因。

### 4.4 工作台頁籤

#### 已觀察事實

- Episode 起訖時間；
- Alarm 時間軸；
- 依時間排序的 Alarm Sequence；
- 每筆事件與前一事件的間隔；
- Alarm frequency；
- Incident ID、machine ID 與 split。

#### 相似歷史事件

- Top-5 歷史 Episode；
- Rank、similarity score 與 shared alarm codes；
- 歷史 Episode 的 machine ID、起訖時間與 sequence；
- 歷史 Episode 後續六小時的已觀察 Alarm；
- 可供摘要引用的完整 Incident ID。

每個候選必須符合既有時間安全政策，不得把 query Episode 之後的事件當作歷史證據。

#### 未來 Alarm 預測

- Top-5 alarm codes；
- Rank 與 model score；
- 6 小時 forecast horizon；
- Model version；
- Baseline scope：`transition`、`machine_fallback` 或 `global_fallback`；
- 對應的 train support；
- 「統計候選，不是故障機率」聲明。

#### Copilot 摘要

摘要固定分成：

1. 已觀察事實；
2. 歷史證據；
3. 模型預測；
4. 限制與下一步人工調查方向。

沒有 LLM API Key 時使用 deterministic template；LLM adapter 失敗時也必須回退至相同模板。

### 4.5 調查摘要區

桌面寬度足夠時，在工作台右側顯示精簡摘要：

- 藍色：observed facts；
- 綠色：retrieved evidence；
- 橘色：model predictions；
- 紅色：limitations。

窄螢幕時摘要移至主要內容下方，不能因版面縮小而隱藏限制聲明。

## 5. 模型評估

### 5.1 使用者問題

Retrieval 與 Forecasting 在最終 Test 的表現如何，結果是否穩定，以及哪些 Alarm 類別仍不可靠？

### 5.2 相似事件 Retrieval

- Test coverage；
- Hit@5；
- Precision@5 與 random baseline；
- Precision lift；
- MRR、NDCG@5；
- Recall@5 與 Max Recall@5 的限制說明；
- 鎖定 feature、horizon 與 threshold。

### 5.3 後續 Alarm Forecasting

- Outcome coverage；
- Hit@5、Precision@5、Mean Recall@5；
- Micro-F1 與 Macro-F1；
- Validation／Test 穩定性；
- Global、Machine、Transition、Linear 與 GRU validation 比較；
- Common、Medium、Rare Test 表現；
- Rare Alarm 尚無可靠預測能力的明顯警告；
- 鎖定 model version、horizon 與 Top-K。

### 5.4 邊界

Retrieval 與 Forecasting 使用不同 relevance／target 定義，必須以頁籤分開呈現，不能把兩者的 Precision、Recall 或 Hit@5 放在同一排名中直接比較。

## 6. 跨頁狀態

第一版只保存下列 session state：

- `selected_machine_id`；
- `selected_incident_id`；
- 工作台目前頁籤；
- 選配摘要 provider。

若使用者更換設備，而目前 Incident 不屬於新設備，應清除 Incident 並選擇該設備時間上最新的可用 Episode。重新整理頁面後不保證保存狀態，第一版不建立使用者帳號或資料庫。

## 7. 90 秒展示路徑

```text
資料總覽
  → 說明資料規模與長尾限制
  → 進入事件調查工作台
  → 選擇設備與 Episode
  → 查看 Alarm 時間軸
  → 展開一筆相似歷史事件
  → 查看未來 Top-5 與 fallback scope
  → 產生具引用的 Copilot 摘要
  → 進入模型評估頁確認 Test 表現與 Rare 限制
```

## 8. 全域呈現規則

- 時間使用資料原始的 UTC-naive timestamp，不自行加上時區名稱；
- Alarm code 與 machine ID 一律視為匿名識別碼；
- Observed、Retrieved、Predicted 使用不同標題與視覺色彩；
- Prediction score 不顯示百分比符號，除非完成 probability calibration；
- 每項歷史敘述都必須能追溯至 Incident ID；
- 不顯示根因、零件、嚴重程度或維修程序；
- 資料或 artifact 缺失時顯示可執行的修復指令，不呈現空白頁；
- 基本功能在離線且沒有 LLM API Key 時仍必須可用。

## 9. 7.1 驗收條件

- 三頁各自回答明確且不重疊的使用者問題；
- 單一 Episode 的調查流程不需要跨頁重選狀態；
- 已觀察事實、歷史證據、模型預測與限制不可混淆；
- 模型評估同時呈現整體效果與長尾失效範圍；
- 使用流程符合 90 秒履歷展示目標；
- 文件足以作為 application service 與 Streamlit 實作的輸出契約。
