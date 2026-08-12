# Incident 統計基線分析

## 1. 目的

本分析為 30 分鐘 derived alarm episodes 建立可重現、可解釋的 deterministic baseline。它使用 train incidents 的歷史分布，辨識事件數、時間跨度或 alarm 種類數位於上尾的 episodes，作為後續調查排序與複雜模型的比較基準。

這些統計標記不是故障、危險或維修標籤，也不能證明 episode 內的匿名 alarm codes 具有共同根因。

## 2. 兩種門檻的差異

| 門檻 | 用途 | 是否改變 episode 邊界 |
| --- | --- | --- |
| 30 分鐘 gap | 將同設備的相鄰 alarms 切成 episodes | 是 |
| Train P95 | 衡量已建立的 episode 是否位於歷史上尾 | 否 |

30 分鐘門檻像剪刀，P95 門檻像量尺。兩者不可混為同一種設定。

## 3. Fit 與套用規則

- 只使用 split = train 的 incidents 計算門檻。
- validation 與 test 不參與 fit。
- P95 採用 higher 插值，門檻保持為實際觀測值。
- 只有數值嚴格大於門檻才產生 flag；剛好等於門檻不標記。
- 每台設備優先使用自己的 train P95。
- 設備的 train incidents 少於 200 筆時，改用全域 train P95。
- 未在 train 出現的新設備也使用全域 fallback。

200 筆 minimum support 的依據是：P95 以上約可保留 10 筆歷史觀測，降低門檻只由單一最大值決定的風險。此數值是透明的工程設定，不是真實設備規格。

## 4. 三種 Upper-tail Flags

| Flag | 判斷條件 | 精確意義 |
| --- | --- | --- |
| is_high_event_count | event_count > P95 | episode 的 alarm 記錄筆數位於歷史上尾 |
| is_high_duration_seconds | duration_seconds > P95 | 第一筆到最後一筆 alarm 的跨度位於歷史上尾 |
| is_high_distinct_alarm_count | distinct_alarm_count > P95 | episode 的匿名 alarm 種類數位於歷史上尾 |

upper_tail_flag_count 是三種條件中成立的數量；is_upper_tail 表示至少一種條件成立。Duration 不是故障、停機或維修時間，alarm diversity 也不代表嚴重程度。

## 5. 完整資料結果

30 分鐘設定共建立 38,153 個 episodes：train 25,936、validation 5,853、test 6,364。全域 train P95 為：

| 指標 | 全域門檻 |
| --- | ---: |
| event_count | 42 |
| duration_seconds | 9,040.763 秒（約 2 小時 30 分） |
| distinct_alarm_count | 6 |

全域門檻主要供 fallback 使用；具足夠 train support 的設備採用設備專屬門檻。

| Split | Episode 數 | 任一 Flag 數 | 任一 Flag 比例 | 高事件數比例 | 長時間跨度比例 | 高 alarm 多樣性比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 25,936 | 2,050 | 7.90% | 4.85% | 4.96% | 3.34% |
| validation | 5,853 | 437 | 7.47% | 4.65% | 4.56% | 3.01% |
| test | 6,364 | 434 | 6.82% | 3.91% | 4.32% | 2.78% |

三個 flags 可以在同一 episode 同時成立，因此個別比例相加不等於 is_upper_tail 比例。Train 的個別比例略低於 5%，是因為使用嚴格大於 P95 的規則，與門檻相同的 ties 不會被標記。

Validation 與 test 的任一 flag 比例沒有高於 train，未見後期 episodes 更常超過這三項歷史上尾門檻的明顯跡象。這不能解讀為設備狀態改善，只能描述統計分布相對於 train baseline 的結果。

## 6. 設備 Support 與 Fallback

20 台設備的 train episode support 範圍為 11 至 2,877，中位數為 1,279。設備 19 只有 11 個 train episodes，低於 200 筆要求；第二少的設備 4 有 217 筆，其餘設備也都符合要求。

| 設備 | Train episodes | Support 是否足夠 | 實際 baseline |
| --- | ---: | --- | --- |
| 19 | 11 | 否 | 全域 fallback |
| 其他 19 台 | 217 至 2,877 | 是 | 各設備自己的 P95 |

設備 19 自行估計的門檻仍保留供診斷，但不參與正式 flags。完整資料中使用 global fallback 的數量為 train 11、validation 2、test 2，與設備 19 的 episodes 相符。

## 7. 後續用途與限制

本 baseline 將用於：

- 在 UI 中排序值得進一步查看的統計長尾 episodes；
- 提供不依賴機器學習的可解釋分析結果；
- 作為後續檢索與預測方法的比較基準；
- 監控不同 split、設備與 gap 設定下的分布變化。

Flags 不得當作真實異常或故障標籤。若以它們作監督式學習目標，模型只會模仿 P95 規則，而不會學到真實故障。它們可以作為候選輸入特徵，但必須確保只使用預測時點已觀察到的資訊，並在 train fit 固定門檻後再套用 validation 與 test。
