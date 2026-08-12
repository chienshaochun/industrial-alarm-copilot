# 第 4 階段驗收

## 1. 驗收範圍

第 4 階段將 processed alarm events 轉換為 derived alarm episodes，建立穩定 incident ID、事件 mapping、episode 統計摘要、gap 敏感度分析，以及只使用 train 資料擬合的 deterministic upper-tail baseline。

Derived episode 是依時間間隔規則產生的分析容器，不是真實故障工單。本階段不建立相似事件檢索器、後續警報預測模型、根因診斷或維修建議。

## 2. Episode 建立契約

- 同設備事件依 machine_id、timestamp、source_row 穩定排序。
- 同一設備與 split 內，相鄰 gap 大於 30 分鐘時開始新 episode。
- gap 剛好 30 分鐘時保留在同一 episode。
- train、validation、test 邊界一律切開，避免彙整特徵跨 split 洩漏。
- episode 不跨設備，每筆 event 恰好對應一個 incident ID。
- incident ID 由 schema、來源、gap、設備、split、起始時間與第一筆 source_row 建立穩定 SHA-256 指紋。
- incident_source 固定為 time_gap_heuristic，is_ground_truth 固定為 false。

## 3. Episode 輸出欄位

每個 episode 保存時間與 alarm 組成摘要：

- 開始與結束時間、時間跨度及事件數；
- 不同 alarm code 數；
- 第一個、最後一個及 dominant alarm code；
- 完全重複事件數；
- baseline 來源、門檻、support 與三種 upper-tail flags。

Dominant alarm 只表示出現次數最多；並列時依字串排序決定。它不代表最嚴重或根因 alarm。Duration 只表示第一筆到最後一筆警報的跨度，不是故障或停機時間。

## 4. Gap 敏感度結果

| Gap | Episode 數 | 單筆占比 | 中位事件數 | P95 事件數 | 中位時長 | P95 時長 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15 分鐘 | 68,093 | 16.38% | 4 | 21 | 約 5.0 分鐘 | 約 53 分鐘 |
| 30 分鐘 | 38,153 | 10.96% | 6 | 41 | 約 20.8 分鐘 | 約 2.47 小時 |
| 60 分鐘 | 19,039 | 6.18% | 11 | 85 | 約 74.1 分鐘 | 約 7.09 小時 |

第一版採用 30 分鐘，因為它接近相鄰 gap 分布的 P90，並位於較嚴格的 15 分鐘與較寬鬆的 60 分鐘之間。這是工程 baseline，不是經 ground truth 驗證的最佳值。

## 5. Deterministic Baseline

Baseline 只使用 train incidents 計算 P95，validation 與 test 不參與 fit。三種 flags 分別檢查 event count、duration 與 distinct alarm count 是否嚴格大於歷史 P95。

具至少 200 個 train incidents 的設備使用自己的 P95；support 不足或 train 未見過的設備使用全域 fallback。設備 19 只有 11 個 train incidents，因此其 15 個完整資料 episodes 使用全域 baseline。

全域 train P95 為：

| 指標 | 門檻 |
| --- | ---: |
| Event count | 42 |
| Duration | 9,040.763 秒 |
| Distinct alarm count | 6 |

完整資料的任一 upper-tail flag 比例為 train 7.90%、validation 7.47%、test 6.82%。這些比例只描述相對 train 歷史的統計上尾，不能解讀為故障率或異常率。

## 6. 自動化驗收

執行：

    conda activate industrial-alarm-copilot
    python -m pytest

本機驗收結果為 48 passed。測試涵蓋：

- 30 分鐘邊界、跨設備與跨 split 切分；
- 暫存 episode number 與穩定 incident ID；
- event mapping 的唯一性與 episode 內連續位置；
- episode 時間及 alarm 組成摘要；
- 15、30、60 分鐘敏感度統計；
- global 與 machine train-only P95；
- minimum support 與 global fallback；
- Parquet 與 JSON round-trip；
- prepare-incidents CLI；
- 完整 444,834 筆 ALPI events 的 incident 分析契約。

## 7. 正式 Artifacts

重建命令：

    python -m industrial_alarm_copilot prepare-data
    python -m industrial_alarm_copilot prepare-incidents

輸出：

| Artifact | 本機大小 | 用途 |
| --- | ---: | --- |
| data/processed/incidents.parquet | 約 1.39 MiB | 每列一個 episode 摘要與 baseline flags |
| data/processed/incident_events.parquet | 約 1.32 MiB | incident ID、source row 與 episode 內位置 |
| data/processed/incident_baselines.json | 約 9.37 KiB | 來源雜湊、設定、版本、門檻與 split 統計 |

Artifact 驗收結果：

| 項目 | 結果 |
| --- | ---: |
| Incidents | 38,153 |
| Mapped events | 444,834 |
| Train incidents | 25,936 |
| Validation incidents | 5,853 |
| Test incidents | 6,364 |
| Machine baseline | 38,138 episodes |
| Global fallback | 15 episodes |

所有 incident IDs 唯一，所有 source rows 唯一，mapping 與 incidents 的 ID 集合一致，event counts 完全吻合，event positions 從 0 開始且連續，duration 全部非負。

衍生 artifacts 由 .gitignore 排除。Metadata 的 source events SHA-256 用來確認輸入版本，code version 必須與產生當下的 Git HEAD 相同。

## 8. 已知限制

- ALPI 沒有真實 incident、故障、severity、維修或根因標籤。
- 30 分鐘規則可能合併無關 alarms，也可能切開同一真實事件。
- P95 flags 是透明的統計基線，不是真實異常標籤。
- Episode 完整摘要只適合 episode 結束後的離線分析；即時預測必須另外建立 observation cutoff，禁止使用尚未發生的事件。
- 不同 gap 產生的 episodes 具有不同語意與 ID，不得在未標示設定時混用。

## 9. 下一階段入口

第 5 階段將以 incidents 與 incident events 為輸入，建立可解釋的相似 episode 特徵與時間安全的歷史檢索器。查詢 episode 只能檢索在其開始時間之前已結束的歷史 episodes，並以 incident ID 引用證據。
