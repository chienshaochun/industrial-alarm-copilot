# Streamlit 應用操作指南

## 啟動方式

先啟用專案環境並安裝依賴：

```powershell
conda activate industrial-alarm-copilot
python -m pip install --editable ".[dev]"
```

確認 `data/processed/` 已有 Stage 3 至 Stage 6 artifacts，再啟動：

```powershell
python -m streamlit run app.py
```

若 artifacts 尚未建立，頁面會顯示需要執行的 CLI，而不是直接拋出未處理例外。

## 三個頁面

### 資料總覽

顯示事件數、derived episode 數、設備與 Alarm code 數量、資料期間、每月事件量、常見 Alarm、設備分布及 chronological split。所有圖表由 Parquet artifact 聚合，不使用手動寫死的數字。

### 事件調查工作台

1. 選擇設備。
2. 選擇 newest-first 的 derived episode。
3. 在四個 tab 中依序檢查已觀察事實、相似歷史事件、未來 Alarm 預測與 Copilot 摘要。
4. 以 episode ID 回查摘要所引用的歷史證據。
5. 檢查 forecast scope 與 train support，再決定是否值得進一步人工調查。

第一次進入工作台時會建立 train-fitted retrieval index，通常需要數秒；同一個 Streamlit process 的後續操作會重用 `st.cache_resource`。

### 模型評估

只讀取鎖定設定的 test artifacts，不重新挑選模型。頁面同時展示：

- Retrieval 的 Hit@5、Precision@5、Recall@5、Precision lift、MRR 與 NDCG@5。
- Forecasting 的 Hit@5、Precision@5、Recall@5、Micro F1 與 Macro F1。
- Rare、medium、common Alarm 的分組結果與長尾失敗模式。

## 90 秒展示流程

1. 在資料總覽說明 444,834 筆 Alarm 如何轉成 38,153 個 episode。
2. 到工作台選一個 upper-tail episode，解釋 Alarm 時序與三種 P95 flag。
3. 指出 Top-5 歷史 episode 的 ID、共同 Alarm 與其後觀察結果。
4. 說明 forecast 是統計排名，不是故障機率。
5. 打開 Copilot 摘要，展示事實、證據、預測與限制的分離。
6. 到模型評估頁主動說明 retrieval Recall@5 與 rare Alarm F1 的限制。
