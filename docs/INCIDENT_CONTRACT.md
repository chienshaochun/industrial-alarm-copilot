# Derived Alarm Episode 資料契約

## 1. 名稱與目的

本專案在程式與產品介面中沿用 `incident` 名稱，但其精確含義是 **derived alarm episode（衍生警報片段）**：同一設備在一段時間內密集出現的警報記錄集合。

每個 episode 必須明確標示：

```text
incident_source = "time_gap_heuristic"
is_ground_truth = false
```

它是供統計、檢索、預測與 UI 調查使用的分析容器，不是真實故障工單，也不能單獨證明多種警報具有共同根因。

## 2. 輸入資料

Episode builder 讀取第 3 階段產生的 processed events。必要欄位：

| 欄位 | 意義 |
| --- | --- |
| `timestamp` | 警報記錄時間 |
| `alarm_code` | 匿名警報代碼 |
| `machine_id` | 匿名設備代碼 |
| `source_row` | 原始 CSV 資料列位置 |
| `duplicate_group_size` | 完全相同事件群組大小 |
| `is_exact_duplicate` | 是否位於完全相同事件群組 |
| `previous_timestamp` | 同設備前一筆事件時間 |
| `gap_seconds` | 同設備前一筆事件到目前事件的秒數 |
| `split` | `train`、`validation` 或 `test` |

輸入必須依 `machine_id`、`timestamp`、`source_row` 穩定排序。Builder 可以重新排序以確保契約，但不得覆寫 `source_row`。

## 3. Episode 邊界規則

預設 `gap_minutes = 30`。一筆事件符合下列任一條件時開始新的 episode：

1. 它是該設備在目前 split 的第一筆事件；
2. `machine_id` 與上一筆事件不同；
3. `split` 與上一筆事件不同；
4. `gap_seconds > gap_minutes * 60`。

因此：

```text
gap_seconds <= 1800：保留在同一 episode
gap_seconds > 1800：開始新的 episode
```

剛好 30 分鐘仍屬於同一 episode。同一設備、同一 timestamp 的多筆事件依 `source_row` 保留順序，且 gap 為 0。

Episode 使用相鄰事件規則，而不是限制第一筆到最後一筆的總時長。例如 `10:00 → 10:25 → 10:50` 的兩個 gap 都是 25 分鐘，因此三筆屬於同一 episode，總時長可以超過 30 分鐘。

## 4. Split 邊界與時間洩漏

Episode 不得跨越 train、validation、test。即使 split 邊界兩側的 gap 小於等於 30 分鐘，也必須切開。

原因是 episode 的事件數、持續時間、警報組合與 dominant alarm 都是彙整特徵。若同一 episode 同時包含 train 與 validation 事件，模型可能透過彙整結果取得未來資訊。

每個 episode 必須恰好屬於一個 `machine_id` 與一個 `split`。

## 5. Incident 資料結構

`incidents.parquet` 每列代表一個 derived episode：

| 欄位 | 型別／規則 | 意義 |
| --- | --- | --- |
| `incident_id` | string | 由 episode 身分欄位產生的穩定 ID |
| `machine_id` | string | 設備識別碼 |
| `split` | string | train、validation 或 test |
| `incident_source` | string | 固定為 `time_gap_heuristic` |
| `is_ground_truth` | bool | 固定為 `false` |
| `gap_minutes` | float | 建立 episode 使用的門檻 |
| `start_time` | datetime | 第一筆事件時間 |
| `end_time` | datetime | 最後一筆事件時間 |
| `duration_seconds` | float | `end_time - start_time` |
| `event_count` | integer | 包含完全重複列的事件數 |
| `distinct_alarm_count` | integer | 不同警報代碼數 |
| `first_alarm_code` | string | 穩定排序後第一筆警報代碼 |
| `last_alarm_code` | string | 穩定排序後最後一筆警報代碼 |
| `dominant_alarm_code` | string | episode 中出現次數最多的代碼 |
| `duplicate_event_count` | integer | `is_exact_duplicate=true` 的列數 |

`duration_seconds` 是第一筆到最後一筆警報的時間跨度，不是故障持續、停機或維修時間。單筆事件 episode 的 duration 為 0。

`dominant_alarm_code` 若有多個代碼並列最高次數，依字串排序選最小者，使結果可重現；它不代表最嚴重或根因警報。

## 6. Event Mapping 資料結構

`incident_events.parquet` 保存事件與 episode 的關係：

| 欄位 | 意義 |
| --- | --- |
| `incident_id` | 對應 derived episode |
| `source_row` | 對應原始 CSV 資料列 |
| `event_position` | episode 內從 0 開始的事件順序 |

mapping 不複製所有事件欄位；可透過 `source_row` 與 `events.parquet` 連接，避免產生多份可能不一致的事件內容。

## 7. Stable Incident ID

`incident_id` 必須由下列內容建立：

- episode schema version；
- `incident_source`；
- `gap_minutes`；
- `machine_id`；
- `split`；
- `start_time`；
- 第一筆事件的 `source_row`。

上述內容以固定順序序列化後計算 SHA-256，使用固定長度前綴，例如：

```text
inc_7e4a2d81c930f560
```

相同輸入資料、設定與程式契約必須產生相同 ID。改變 gap、episode 邊界或第一筆事件時，ID 應跟著改變，避免把不同分組誤認為同一 episode。

## 8. 完整性規則

Builder 輸出必須滿足：

- 每一筆 processed event 恰好對應一個 `incident_id`；
- mapping 的 `source_row` 不得重複或缺漏；
- 一個 episode 不得包含多台設備或多個 split；
- `event_position` 必須由 0 開始且連續；
- episode 內事件必須依 `timestamp`、`source_row` 遞增；
- `event_count` 必須等於 mapping 列數；
- 所有 `duration_seconds` 必須大於等於 0；
- 所有 `is_ground_truth` 必須為 false；
- 所有 `incident_source` 必須為 `time_gap_heuristic`；
- incidents 的 `event_count` 加總必須等於 processed events 總列數。

## 9. 敏感度分析

正式第一版 artifact 使用 30 分鐘。分析階段另以 15、30、60 分鐘建立統計結果，比較：

- episode 總數；
- 單筆事件 episode 占比；
- event count 與 duration 分位數；
- alarm diversity；
- 後續 baseline 對門檻的敏感程度。

不同 gap 建立的 episodes 具有不同語意與 ID，不得在未標示設定時混合比較。

## 10. 解讀限制與非目標

本階段不會：

- 宣稱 episode 是真實故障；
- 產生根因、severity 或維修建議；
- 根據匿名代碼補充設備零件語意；
- 刪除完全重複事件；
- 使用 validation 或 test 調整 train baseline；
- 以 LLM 決定 episode 邊界。

若未來取得維修工單或真實 incident labels，應將本規則視為 baseline，並用 ground truth 評估其合併與切分錯誤。
