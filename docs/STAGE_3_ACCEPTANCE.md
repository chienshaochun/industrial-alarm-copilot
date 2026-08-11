# 第 3 階段驗收

## 1. 驗收範圍

第 3 階段完成 ALPI 資料理解、契約驗證、探索性分析、incident gap 選擇、時間序列切分，以及可重建的 processed artifact pipeline。

本階段不建立正式 incident ID、檢索器或預測模型；這些功能分別屬於後續階段。

## 2. 自動化驗收

執行：

```powershell
conda activate industrial-alarm-copilot
python -m pytest
```

驗收內容包含：

- CSV 欄位、缺值、timestamp 與識別碼驗證；
- 原始列追溯與完全重複事件標記；
- 設備內 gap 計算與跨設備隔離；
- incident gap 候選門檻統計；
- 每設備 unique-timestamp chronological split；
- 人工時間洩漏案例拒絕；
- Parquet round-trip 與 metadata JSON；
- raw CSV 到 processed artifacts 的整合流程；
- 完整 ALPI SHA-256、筆數、型別與 split 契約。

## 3. 正式 Artifact 驗收

產生命令：

```powershell
python -m industrial_alarm_copilot prepare-data
```

輸出：

| Artifact | 用途 |
| --- | --- |
| `data/processed/events.parquet` | 保存正規化事件、重複標記、gap 與 split |
| `data/processed/events.metadata.json` | 保存來源、設定、程式版本、schema 與時間邊界 |

本機驗收結果：

| 項目 | 結果 |
| --- | ---: |
| Parquet 大小 | 約 9.34 MiB |
| Shape | 444,834 × 9 |
| 重複群組涉及列數 | 166 |
| 空 gap | 20 |
| 非空 gap | 全部大於等於 0 |
| `source_row` | 全部唯一 |
| train | 311,368 |
| validation | 66,727 |
| test | 66,739 |

衍生 artifacts 已由 `.gitignore` 排除，可透過同一命令重新建立。metadata 的 `source.sha256` 必須等於：

```text
53bd4414a6fb5b6875a9535f1be622dfb2dfba69de407d071a52d0d304160d1a
```

metadata 的 `code_version` 必須與產生當下的 `git rev-parse HEAD` 相同。

## 4. 防止時間洩漏

- 每台設備分開按唯一 timestamp 切成 70% train、15% validation、15% test；
- 相同設備、相同 timestamp 不可跨 split；
- 每台設備必須滿足 train 早於 validation、validation 早於 test；
- scaler、編碼器、統計量與模型只能在 train fit；
- 相似事件候選在後續階段仍須額外套用 query-time cutoff。

每設備切分的全域 calendar-time 可能重疊，限制與第二組全域 cutoff 評估建議記錄於 `docs/SPLIT_STRATEGY.md`。

## 5. 已知限制

- ALPI 沒有警報語意、severity、clear/reset、維修紀錄與真實 incident 標籤；
- 30 分鐘 incident gap 是資料支持的工程假設，不是真實故障邊界；
- 重複事件可能是真實重複觸發、alarm flooding 或資料匯出重複，目前保留且標記；
- 資料沒有時區，不能可靠分析跨工廠的小時或星期模式；
- 第 3 階段尚未建立 incident artifact、檢索 index 或 forecasting model。

## 6. 下一階段入口

第 4 階段將讀取 `events.parquet`，使用 `configs/default.toml` 的 30 分鐘門檻建立 incident windows，並先完成 deterministic statistical baseline，再進入相似事件檢索與預測模型。
