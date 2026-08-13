# 第 6 階段驗收

## 驗收結果

第 6 階段「後續警報預測」已完成。

- [x] 在 episode `end_time` 建立時間安全的 6 小時多標籤 outcomes。
- [x] 保留 incomplete、empty 與 unknown-label coverage。
- [x] 只使用 train 建立 139-label vocabulary。
- [x] 分析 1／6／24 小時 horizon 並鎖定 6 小時。
- [x] 分析 rare／medium／common 長尾支持度。
- [x] 建立 global、machine 與 transition frequency baselines。
- [x] 建立未加權與 balanced One-vs-Rest logistic 對照模型。
- [x] 建立 CPU 可運行的 alarm-sequence GRU 與 capped-balanced 版本。
- [x] 以 validation-only 規則鎖定 `transition_frequency_v1`。
- [x] 只執行一次 test，並分開報告支持度群組。
- [x] 輸出 model、metrics、fallback profile 與 provenance metadata。
- [x] 自動化測試覆蓋標籤、詞彙、模型、評估、選型與 pipeline。

## 物理意義

原始 alarm event 像設備控制面板上瞬間亮起的燈；episode 是一段連續的燈號波形。Forecasting 不再只描述這段波形，而是在波形結束時問：「依照這台設備過去的運作習性，以及最後亮起的燈，接下來六小時最可能亮起哪五盞燈？」

Global baseline 像使用整座工廠的熱門燈號；machine baseline 像改用該設備自己的歷史；transition baseline 再加入「目前最後停在哪一個狀態」。GRU 則像讓模型逐盞閱讀整段燈號順序。實驗顯示目前最後狀態與設備歷史已涵蓋大部分可預測訊號，複雜 GRU 沒有帶來足以抵銷成本的增益。

## 已知限制

- ALPI alarm codes 匿名，無法推論零件、根因或維修方法。
- Episode 是 30 分鐘 gap 的衍生分析單位，不是真實 incident 標註。
- 未來 alarm 是由時間窗自動建立的 proxy label，不是人工確認的故障結果。
- Rare labels 的 test Recall@5 為 0；需要更多真實樣本、設備語意或人工標註才能改善。
- Top-5 score 是排序依據，不宣稱是已校準故障機率。
