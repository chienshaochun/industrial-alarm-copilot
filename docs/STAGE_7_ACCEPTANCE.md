# 第 7 階段驗收：AI Copilot 與 Streamlit 應用

## 驗收範圍

本階段把 Stage 3 至 Stage 6 的 artifacts 組成一個可操作的 portfolio product，但不重新訓練模型、不引入資料庫，也不要求本機部署 LLM。

## 已完成項目

- 定義三頁使用流程與 90 秒 demo 路徑。
- 建立 immutable application contracts，區分觀察、證據、預測與限制。
- 由 `incidents.parquet`、`events.parquet` 與 mapping 重建 episode 事實。
- 使用 locked retrieval setting 產生時間安全的 Top-5 歷史引用。
- 載入 `transition_frequency_v1` portable JSON 並執行 hierarchical fallback Top-5 預測。
- 建立統一 `InvestigationService`，讓 UI 不直接操作模型內部資料。
- 建立 deterministic Copilot 摘要與 provider-neutral optional LLM port。
- 對未知 episode／Alarm 引用啟用 guardrail 與 deterministic fallback。
- 完成資料總覽、事件調查工作台與模型評估三頁 Streamlit 應用。
- 缺少 generated artifacts 時提供可執行的修復指令。
- 用 `AppTest` 驗證 Streamlit entry point，並對 view model、服務與 guardrail 建立單元測試。

## 資訊邊界驗收

| 類別 | 來源 | UI 標示 | 禁止推論 |
| --- | --- | --- | --- |
| 已觀察事實 | Stage 3/4 artifacts | 已觀察事實 | 根因、維修動作 |
| 歷史證據 | time-safe Top-5 retrieval | 相似歷史事件＋episode ID | 因果關係 |
| 統計預測 | Stage 6 locked model | 未來 Alarm 預測＋scope | 故障機率 |
| 摘要 | 上述結構化資料 | Copilot 摘要＋generator | 新增未引用事實 |

## 鎖定 test 結果的誠實呈現

- Retrieval：Hit@5 77.98%、Precision@5 38.94%、Precision lift 2.45 倍，但 Recall@5 僅 0.0367%。
- Forecasting：Hit@5 86.59%、Mean Recall@5 69.97%、Micro F1 54.30%，但 Macro F1 只有 7.58%。
- Rare Alarm 的 test F1 仍為 0；應用不隱藏這個失敗模式。

## 驗證指令

```powershell
python -m pytest
python -m streamlit run app.py
```

最終結果為 **147 passed in 201.43s**。測試使用 repository 內隔離的 `--basetemp`，避免 Windows Sandbox 與登入帳號的暫存目錄 ownership 衝突。
