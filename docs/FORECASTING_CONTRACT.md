# 後續 Alarm 預測契約

## 1. 目的與範圍

第 6 階段針對一個已結束的 derived alarm episode，預測同一設備在固定未來時間窗內可能出現的 alarm codes，並輸出排序後的 Top-K 模型分數。

本階段建立的是多標籤 forecasting：未來時間窗內可以同時出現零種、一種或多種 alarms。模型不預測故障根因、嚴重程度、維修方式、精確發生時間或 alarm 出現次數。

ALPI 的 alarm codes 是匿名代碼。預測結果只能表達為「哪些代碼較可能在指定時間窗內出現」，不能補充不存在的設備零件或故障語意。

## 2. 預測時間點

第一版在完整 episode 的 `end_time` 做一次預測：

```text
episode.start_time              episode.end_time        end_time + horizon
        |------- 已觀察輸入 X ---------|------ 未來標籤 Y ------|
                                      ^
                                  預測時間點
```

因此本版本屬於 episode 結束後的離線調查與後續行為預測，不宣稱支援 episode 尚未結束時的即時預測。

若未來建立即時版本，必須另外定義 observation cutoff，且 event count、duration、alarm composition 與所有序列特徵都只能使用 cutoff 以前的事件。

## 3. 模型輸入

每筆樣本的輸入只包含 query episode 結束以前已發生的資訊。第一版允許建立：

- episode 內 alarm code 的出現與次數；
- alarm code 的穩定時間順序；
- episode 內相鄰事件的時間間隔；
- event count、duration seconds 與 distinct alarm count；
- first、last 與 dominant alarm code；
- 可選且明確版本化的 machine context。

完整重複事件仍依資料契約保留。若模型使用 machine ID，必須另行比較不使用 machine ID 的版本，避免模型只記住設備身分。

模型參數保存從 train 歷史學到的規律；目前 episode 則是每次推論時提供的狀態。模型不得將 query 的未來 alarms、future outcome、validation／test 統計或檢索 relevance label 當作輸入特徵。

## 4. 預測時間窗與標籤

第一版正式 `forecast_horizon_hours = 6`。對 incident `i`，符合下列條件的 events 形成真實未來集合：

```text
event.machine_id == incident.machine_id
incident.end_time < event.timestamp
event.timestamp <= incident.end_time + 6 hours
```

每個 alarm code 在時間窗內只要至少出現一次，其多標籤目標即為 1；未出現則為 0。重複次數與出現順序不改變第一版標籤：

```text
10:00 episode 結束
10:30 alarm 98
11:20 alarm 98
14:00 alarm 11

label set = {98, 11}
```

六小時設定與第 5 階段 outcome window 一致，使 forecast 結果可與相似歷史 episode 的 outcome 證據比較。它是可解釋的工程 baseline，不代表工業領域天然正確的時間尺度。

1／6／24 小時可用於 train／validation 敏感度分析，但不得使用 test 反覆挑選最漂亮的 horizon。若 validation 支持不同正式設定，必須在 test 解封前寫入設定檔並 commit。

## 5. Outcome 完整性

只有完整觀察到整個未來時間窗的樣本，才能用於模型訓練與主要評估。

對每台設備與每個 split，必須滿足：

```text
incident.end_time + forecast_horizon <= 該設備該 split 的最後可觀察時間
```

若時間窗跨越 train／validation／test 邊界，或超過該設備在目前 split 的資料終點，則：

```text
outcome_is_complete = false
```

不完整樣本不得因為目前已看到部分 future alarms，就被當成完整負標籤或正標籤。它們必須保留在 coverage 統計，但排除於 fit 與主要 forecasting metrics。

完整時間窗內沒有任何 future alarm 的樣本是合法的全零多標籤樣本，不得靜默刪除。評估報告必須分別記錄完整 outcome、無 future alarm 與不完整 outcome 的數量。

## 6. Label Vocabulary 與未知 Alarm

Label vocabulary 只能由 train 中完整樣本的 future alarm codes 建立。Validation 與 test 只能套用既有 vocabulary，不可重新 fit 或擴張輸出維度。

Validation／test 的 future window 若出現 train vocabulary 以外的 alarm，必須記錄：

- unknown future alarm event count；
- unknown future alarm code count；
- 含 unknown label 的 query count 與比例。

未知 alarm 不可能被現有模型正確預測，不能從評估資料中靜默刪除後宣稱已完整找回真實標籤。報告應同時區分 known-label metrics 與 unknown-label coverage limitation。

## 7. 預測輸出

每個 query episode 至少輸出：

| 欄位 | 意義 |
| --- | --- |
| `incident_id` | Query episode 的穩定 ID |
| `machine_id` | Query 設備 |
| `prediction_time` | 等於 episode `end_time` |
| `forecast_end_time` | `end_time + horizon` |
| `alarm_code` | Train vocabulary 內的候選 alarm |
| `rank` | 依模型分數排序，從 1 開始 |
| `model_score` | 頻率、decision score 或 probability，依模型版本定義 |
| `model_version` | 可追溯的模型與特徵版本 |
| `forecast_horizon_hours` | 預測時間窗 |

第一版 UI 固定顯示 Top-5。不同模型的 `model_score` 未必都是校準機率，因此介面統一稱為「模型分數」；只有通過 probability calibration 驗證的版本才可標示為機率或信心。

## 8. 模型比較順序

依序建立並比較：

1. 全域 train label frequency baseline；
2. 每台設備的 train label frequency baseline，support 不足時回退全域；
3. 依目前 alarm 狀態條件化的 transition／Markov baseline；
4. One-vs-Rest 線性多標籤模型；
5. 硬體允許時的小型 GRU 序列模型。

較複雜模型必須與 deterministic baselines 使用相同樣本、label vocabulary、horizon、Top-K 與指標。若深度模型沒有穩定勝過簡單模型，產品第一版應保留較透明且成本較低的模型。

## 9. 評估指標

主要 Top-5 指標：

- Hit@5：至少一個預測 alarm 實際出現的 query 比例；
- Precision@5：五個預測中實際出現的比例；
- Recall@5：已知實際 labels 中被 Top-5 找回的比例。

同時報告：

- micro／macro F1 與 per-label support；
- outcome coverage 與無 future alarm query 比例；
- unknown-label query／event／code 比例；
- 每台設備與各 split 的切片結果；
- common 與 rare alarms 的分組結果。

Common／rare 只能依 train 完整樣本中的 label support 定義，門檻必須在查看 test 前寫入設定並版本化。不得以 validation 或 test 的 raw event frequency 重新定義類別。

對真實 known-label 集合為空的 query，Precision@5 與 Hit@5 可記為 0；sample-level Recall@5 不具有分母，必須記為缺值並另外透過 coverage 呈現，不可任意當作 0 或 1 拉動平均。

## 10. Validation、Test 與防止洩漏

- 所有 vocabulary、scaler、encoder、feature selector、模型與 calibration 只能以 train fit。
- Validation 用於選擇 horizon、特徵、超參數、threshold 與正式模型版本。
- Test 在設定與程式 commit 凍結前不得查看。
- 不可 random shuffle 後重新切分時間序列。
- 建立一筆 query 特徵時，不得使用其 `end_time` 以後的事件。
- Retrieval candidate outcome 只能作為產品證據或獨立實驗訊號，不得直接把 query 的真實 future outcome 餵給 forecasting model。
- 若 forecasting 結果未來用於 retrieval reranking，必須使用模型預測而非 query 的真實 future labels。

正式模型選定後應先建立 Git freeze commit，再執行一次 test 評估。Test 結果只能用於最終報告，不得回頭改參數後仍稱為無偏 test。

## 11. Artifacts 與可重現性

Stage 6 artifacts 至少保存：

- forecasting label vocabulary 與 support；
- 訓練與評估樣本契約統計；
- feature schema 與模型版本；
- horizon、Top-K、common／rare 定義與模型設定；
- model／baseline parameters；
- validation 與最終 test metrics；
- events、incidents、incident mapping 的 SHA-256；
- 產生 artifacts 的 Git commit。

相同來源、設定與程式版本應產生相同的 deterministic baseline。需要隨機初始化的模型必須記錄 random seed、套件版本與硬體資訊。

## 12. 解讀限制與非目標

本階段不會：

- 將 predicted alarm 宣稱為一定會發生；
- 由匿名 alarm code 推論物理根因或維修動作；
- 預測精確發生時間、alarm 次數或事件嚴重度；
- 將 derived episode 當成真實故障工單；
- 以 test 選模型或調參；
- 因沒有 future alarm 而刪除較困難的合法樣本；
- 將未校準的模型分數標示為真實機率。

本階段的成功代表模型能在 ALPI 的時間後移資料中，比透明 baseline 更好地排序後續匿名 alarm codes；不代表它已能診斷工業故障或跨工廠直接部署。
