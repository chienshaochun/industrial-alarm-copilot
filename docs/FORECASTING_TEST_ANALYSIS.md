# Forecasting 最終 Test 分析

## 鎖定設定

- 模型：`transition_frequency_v1`
- 預測時間窗：6 小時
- Top-K：5
- Machine baseline 最低 train support：200
- Transition state 最低 train support：20
- 不足時依序 fallback 至 machine frequency，再 fallback 至 global frequency

設定與 validation 選型規則先於 test 固定；本結果沒有用來重新調參。

## Test 結果

| 指標 | 結果 |
| --- | ---: |
| Episode count | 6,364 |
| Complete outcomes | 6,287 |
| Outcome coverage | 98.79% |
| Hit@5 | 86.59% |
| Precision@5 | 46.59% |
| Mean Recall@5 | 69.97% |
| Micro-F1 | 0.5430 |
| Macro-F1 | 0.0758 |
| Evaluated labels | 97 |

Validation Micro-F1 為 `0.5528`，test 為 `0.5430`，下降約 `0.0098`，顯示鎖定模型在較晚時間資料上的效果穩定。

## 支持度分組

| 分組 | Labels | Test 正例 | Precision@5 | Recall@5 | Micro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rare | 73 | 315 | 0.0000 | 0.0000 | 0.0000 |
| Medium | 44 | 1,648 | 0.2500 | 0.0194 | 0.0360 |
| Common | 22 | 20,544 | 0.4668 | 0.7114 | 0.5637 |

模型主要能力來自 common alarms；rare alarms 尚不能被可靠預測。產品介面必須稱輸出為「統計預測候選」，並揭露 support／fallback，不得宣稱能保證找到罕見故障。

## Fallback 行為

- 5,617 筆完整 test queries 使用 transition state。
- 670 筆因 state support 不足，使用 machine fallback。
- 本次沒有 test query 需要 global fallback。

## 產物

- `forecast_model.json`：可攜式 transition／machine／global scores 與 support。
- `forecast_test_results.csv`：唯一一次整體 test 指標。
- `forecast_test_support_groups.csv`：長尾分組指標。
- `forecast_test_scope_profile.csv`：fallback 使用量。
- `forecast_test.metadata.json`：來源 SHA-256、設定與 Git commit。
