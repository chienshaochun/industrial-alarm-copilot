# Forecasting Validation 模型選型分析

## 實驗契約

- 預測時間點：derived episode 的 `end_time`。
- 預測目標：同一設備在 `(end_time, end_time + 6 小時]` 內出現的 alarm codes。
- 輸出：Top-5 多標籤候選。
- 所有 vocabulary、frequency、特徵轉換與模型參數只使用完整 train outcomes 擬合。
- validation 只用於比較模型；test 在模型與選型規則鎖定後才執行一次。

## Validation 結果

| 模型 | Hit@5 | Precision@5 | Mean Recall@5 | Micro-F1 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global frequency | 0.8258 | 0.3784 | 0.5587 | 0.4412 | 0.0332 |
| Machine frequency | 0.8641 | 0.4731 | 0.7112 | 0.5515 | 0.0841 |
| Transition frequency | **0.8643** | **0.4742** | **0.7133** | **0.5528** | 0.0931 |
| OVR logistic | 0.0863 | 0.0174 | 0.0212 | 0.0203 | 0.0040 |
| OVR logistic balanced | 0.1348 | 0.0277 | 0.0485 | 0.0323 | 0.0159 |
| GRU | 0.8603 | 0.4480 | 0.6768 | 0.5223 | 0.0628 |
| GRU balanced capped | 0.8299 | 0.3835 | 0.5787 | 0.4471 | **0.0986** |

## 選型規則與結果

選型先保留 Macro-F1 距離最佳模型不超過 `0.01` 的候選，再最大化 Micro-F1；若仍同分，選擇複雜度較低的模型。這項規則同時防止整體效果掩蓋長尾警報，也避免為極小的長尾增益犧牲大量常見警報品質。

`transition_frequency_v1` 與 `gru_balanced_capped_v1` 進入 Macro-F1 容許帶，但 transition 的 Micro-F1 為 `0.5528`，明顯高於 balanced GRU 的 `0.4471`，因此鎖定 transition model。

## 解讀

Machine frequency 已經捕捉到不同設備的穩定運作習性；再加入 episode 最後一個 alarm state 後，transition baseline 取得小幅但一致的改善。GRU 能學到順序訊號，但目前資料量與匿名 alarm 語意不足以讓其全面超越統計模型。

線性模型表現很差，表示目前 episode 的 TF-IDF／shape 與未來六小時 alarm 並非穩定的線性映射。這是一個負結果，不應被刪除或包裝成成功模型。

## 長尾限制

Train vocabulary 有 139 個 alarm labels，其中 rare 1–49 筆有 73 個，但只佔 0.93% 正例量；common 500 筆以上只有 22 個，卻佔 92.34% 正例量。Class weighting 可以提高 Macro-F1，但會顯著降低整體 Precision 與 Recall，不能取代更多真實標註與資料支持。
