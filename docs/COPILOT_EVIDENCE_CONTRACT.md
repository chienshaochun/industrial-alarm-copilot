# Copilot 證據契約

## 目的

Copilot 的任務是整理調查線索，不是產生新的設備事實。應用層先把資料轉成四個不可混用的區塊，再交給摘要產生器：

1. **已觀察事實**：選定 episode 內真正存在的 Alarm、時間、數量與統計旗標。
2. **歷史證據**：只包含在 query 開始前已結束的 Top-5 episode，並保留 episode ID 作為引用。
3. **統計預測**：第六階段鎖定模型輸出的 Top-5 Alarm、分數、fallback scope 與 train support。
4. **限制**：derived episode、相似性及預測各自不能代表什麼。

`InvestigationResult` 是 presentation layer 唯一可使用的調查輸入。Streamlit 不直接拼接原始 DataFrame，也不在頁面中重新計算模型參數。

## 預設摘要

`TemplateSummaryGenerator` 是預設且永遠可用的離線摘要器。它不需要 API key，輸出可重現，也能逐句追溯到 application contract。

每筆歷史描述都附帶 `incident_id`。預測描述則保留 `model_version`、`baseline_scope` 與 `train_support`，防止把模型候選誤寫成已發生的事實。

## 可選 LLM adapter

`EvidenceConstrainedLLMGenerator` 是 provider-neutral port，本階段不綁定任何雲端供應商。未來可注入任意 callable，但輸出必須是結構化資料，且受到以下檢查：

- 引用的 episode ID 必須存在於 Top-5 歷史證據。
- 引用的 Alarm code 必須存在於已觀察事件、歷史證據或 forecast 候選。
- overview 不得為空，長度不得超過 800 字。
- LLM 不能修改 deterministic 的事實、預測與限制區塊。
- API 例外、格式錯誤或未知引用都會自動退回離線模板。

這些限制不能證明自然語言中的每一句話都正確，因此 UI 仍把 LLM 視為「摘要層」，而不是根因分析器或維修決策器。

## 時間安全界線

相似 episode 必須滿足：

```text
candidate.end_time < query.start_time
```

候選 episode 的後續 6 小時 Alarm 可以作為歷史證據；query 本身的未來 Alarm 不會顯示在調查頁。Forecast 使用的是第六階段 train-fitted、validation-selected、test-locked 的 portable JSON model。
