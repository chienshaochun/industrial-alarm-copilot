# 時間序列切分策略

## 1. 目的

本專案的預測與檢索都模擬「使用過去資料分析未來事件」。因此禁止隨機打散警報事件，並採用每台設備內的 chronological split，避免同一設備的未來事件進入其訓練資料。

預設比例：

```text
train       70%
validation  15%
test        15%
```

設定保存在 `configs/default.toml`。

## 2. 切分單位與順序

事件先依下列欄位穩定排序：

1. `machine_id`
2. `timestamp`
3. `source_row`

切分比例依每台設備的唯一 timestamp 數量計算，而不是直接依資料列計算。同一設備、相同 timestamp 的所有事件必須放在同一 split，避免同一瞬間同時出現在過去與未來。

每台設備都必須符合：

```text
max(train.timestamp) < min(validation.timestamp)
max(validation.timestamp) < min(test.timestamp)
```

## 3. 完整 ALPI 切分結果

| Split | 事件數 | 約占比 | 設備數 | 全域最早時間 | 全域最晚時間 |
| --- | ---: | ---: | ---: | --- | --- |
| train | 311,368 | 70.00% | 20 | 2019-02-21 10:16:18.130 | 2020-06-15 17:20:13.839 |
| validation | 66,727 | 15.00% | 20 | 2019-07-23 06:36:21.926 | 2020-06-15 19:59:55.882 |
| test | 66,739 | 15.00% | 20 | 2019-07-30 01:03:11.514 | 2020-06-17 03:53:51.885 |
| **合計** | **444,834** | **100%** | **20** |  |  |

事件總數在切分前後一致，三個 split 都包含全部 20 台設備。因相同 timestamp 必須一起移動，實際比例允許有極小偏差。

## 4. 為何採每設備切分

不同設備的資料起點與終點不同。若使用全域單一日期作為切分點，較晚加入觀察的設備可能缺少 train，較早結束觀察的設備則可能缺少 validation 或 test。

每設備切分可以：

- 保證每台設備都有歷史訓練與未來評估資料；
- 評估同一設備在時間往後推進時的預測能力；
- 支援設備條件化 baseline；
- 避免同設備內最直接的未來資訊洩漏。

## 5. 全域日期重疊

三個 split 的全域日期範圍會重疊。例如某台較早開始記錄的設備可能已進入 validation，而較晚開始記錄的另一台設備仍位於 train。這不違反每台設備內的時間順序，但對跨設備共用模型存在限制：模型可能從設備 B 較晚的 calendar time 學習，再評估設備 A 較早的事件。

因此必須遵守：

- scaler、編碼器、TF-IDF 與模型只能在標示為 train 的資料上 fit；
- validation 只用於選擇設定，不可回流訓練；
- test 只用於最終報告；
- 對某事件進行相似事件檢索時，候選事件的結束時間必須早於查詢時間；
- 所有以時間變動的全域統計都必須使用查詢當下以前的資料；
- 最終報告需揭露每設備切分的跨設備 calendar-time 重疊限制。

若後續模型顯示明顯的全域時間漂移，應增加「全域日期 cutoff」作為第二組較嚴格評估，而不是用它取代目前的主要每設備評估。

## 6. 驗證方式

`validate_split_integrity()` 會自動檢查：

- 每台設備是否同時具有 train、validation、test；
- train 與 validation 的時間邊界是否嚴格遞增；
- validation 與 test 的時間邊界是否嚴格遞增。

單元測試另包含人工時間洩漏案例，確認 validator 會拒絕 validation 早於 train 的資料。

執行：

```powershell
conda activate industrial-alarm-copilot
python -m pytest
```
