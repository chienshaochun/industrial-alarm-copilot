# Forecasting 標籤與長尾分析

## 1. 分析目的

本分析確認 derived episodes 能否建立足夠完整的後續 alarm 多標籤資料，並在模型訓練前固定 forecast horizon、Top-K 及 rare／medium／common support 分組。

Forecasting 標籤只回答：

> Episode 結束後固定時間窗內，同一設備實際出現過哪些匿名 alarm codes？

它是由歷史 event logs 直接觀察到的多標籤答案，不是故障、根因、嚴重度或維修結果標籤。

## 2. 標籤建立規則

正式樣本以 episode `end_time` 為預測時間點。對同一設備、同一 split，標籤時間窗為：

```text
incident.end_time < event.timestamp
event.timestamp <= incident.end_time + forecast horizon
```

時間窗不得跨越 train、validation 或 test 邊界。只有完整觀察到整個時間窗的 episode 才能進入訓練與主要評估；完整但沒有 future alarm 的 episode 保留為合法全零標籤。

Label vocabulary 只由 outcome 完整的 train 樣本建立。Validation／test 出現 train 未見 alarm 時，不擴充 vocabulary，而是另外記錄 unknown label diagnostics。

## 3. Horizon 敏感度分析

Horizon 選擇只使用 train 與 validation。比較結果如下：

| Horizon | Split | Outcome coverage | 完整全零占比 | 平均 known labels | Unknown query share |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 小時 | train | 99.90% | 50.32% | 0.745 | 0.00% |
| 1 小時 | validation | 99.52% | 49.37% | 0.749 | 0.09% |
| 6 小時 | train | 99.73% | 11.98% | 3.649 | 0.00% |
| 6 小時 | validation | 98.84% | 12.31% | 3.578 | 0.31% |
| 24 小時 | train | 99.33% | 2.12% | 7.032 | 0.00% |
| 24 小時 | validation | 97.66% | 2.12% | 6.842 | 1.15% |

1 小時約有一半完整樣本為全零，平均不足一個正標籤，訊號較稀疏。24 小時平均接近七個正標籤，超過第一版 Top-5 顯示容量，也更可能混入與目前 episode 關係較弱的日常活動。6 小時平均約 3.6 個正標籤、全零約 12%，且 validation coverage 仍達 98.84%。

因此第一版固定：

```text
forecast_horizon_hours = 6
top_k = 5
```

六小時是工程 baseline，不是工業領域天然正確的時間尺度。

## 4. 六小時 Outcome Coverage

| Split | Episodes | 完整 outcomes | Coverage | 不完整 | 完整但無 future alarm | 全零占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 25,936 | 25,867 | 99.73% | 69 | 3,099 | 11.98% |
| Validation | 5,853 | 5,785 | 98.84% | 68 | 712 | 12.31% |
| Test | 6,364 | 6,287 | 98.79% | 77 | 775 | 12.33% |

三個 split 的完整性與全零比例接近。六小時設定只排除 214 個 outcome 不完整 episodes，占全部 38,153 episodes 約 0.56%。

我們在模型選型前已查看 test 的聚合 coverage 與 vocabulary diagnostics，作為標籤管線完整性檢查；horizon 與 support 門檻的選擇只使用 train／validation，且尚未查看任何 test 模型預測指標。後續不得使用 test 聚合診斷重新調整正式設定。

## 5. Train-only Vocabulary 與未知標籤

完整 train outcomes 建立 139 種可預測 alarm codes。原始資料共有 154 種 alarm codes；沒有進入 vocabulary 的代碼可能未曾出現在完整 train future window，或只在較晚 split 出現。

| Split | 平均 known labels | 含 unknown label queries | Unknown query share | Unknown code occurrences | Unknown events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 3.649 | 0 | 0.00% | 0 | 0 |
| Validation | 3.578 | 18 | 0.31% | 18 | 54 |
| Test | 3.580 | 51 | 0.81% | 51 | 139 |

Train vocabulary 對較晚資料的 query coverage 超過 99%。未知 alarm 不可能由固定輸出層預測，因此最終評估必須把它列為 vocabulary coverage limitation，不能靜默刪除。

## 6. Label Support 分布

139 種 train labels 的 sample support 摘要：

| 統計量 | Support |
| --- | ---: |
| Minimum | 1 |
| P10 | 3 |
| P25 | 7 |
| Median | 33 |
| Mean | 679.12 |
| P75 | 181 |
| P90 | 2,131.8 |
| P95 | 3,648.6 |
| Maximum | 13,267 |

平均值遠高於中位數，顯示少數常見 alarm 主導正標籤。46 種 alarm 的 support 少於 10，73 種少於 50。

前五種 labels：

| Alarm code | Train sample support | Train sample share |
| --- | ---: | ---: |
| 98 | 13,267 | 51.29% |
| 11 | 9,749 | 37.69% |
| 137 | 9,543 | 36.89% |
| 26 | 9,501 | 36.73% |
| 29 | 6,886 | 26.62% |

Support 表示該 alarm 在多少個完整 train 六小時視窗內至少出現一次，不是 raw event count。相同時間窗重複一百次仍只提供一個正樣本。

## 7. 固定 Support 分組

第一版只依 train sample support 固定：

```text
rare：1–49
medium：50–499
common：至少 500
```

| Support group | Label 數 | Label 種類占比 | Positive samples | Positive mass 占比 |
| --- | ---: | ---: | ---: | ---: |
| Rare | 73 | 52.52% | 879 | 0.93% |
| Medium | 44 | 31.65% | 6,354 | 6.73% |
| Common | 22 | 15.83% | 87,165 | 92.34% |

只有 22 種 common labels，卻占 92.34% 的所有 train 正標籤。模型即使忽略 117 種 medium／rare alarms，仍可能得到好看的 micro 指標，因此 Stage 6 必須同時報告 macro、per-label 與 support-group metrics。

Rare／medium／common 只描述歷史訓練支援程度，不代表 alarm 的嚴重度、風險或工程重要性。

## 8. 建模含義

- 全域頻率 baseline 預期偏向少數 common alarms，作為最低比較基準。
- Machine-conditioned 與 transition baseline 可讓特定上下文提高非熱門 alarm 排名。
- 線性與序列模型需比較未加權及受限制的 class-weighted loss。
- 加權若提高 rare recall 但大幅降低 Precision@5 或 common performance，不視為全面改善。
- Support 少於 10 的 alarm 缺乏足夠案例，不宣稱能學到穩定物理規律。
- 不使用 SMOTE 直接內插匿名 alarm sequences，避免產生缺乏物理意義的合成順序。

物理上，139 種輸出像 139 盞警示燈：22 盞常見燈貢獻超過九成亮燈紀錄，73 盞 rare 燈合計不到 1%。加權可以提高系統對微弱訊號的敏感度，但不能把一筆歷史案例變成可靠經驗。

## 9. 凍結設定

設定保存於 `configs/default.toml`：

```toml
[forecasting]
top_k = 5
forecast_horizon_hours_candidates = [1, 6, 24]
selected_forecast_horizon_hours = 6
rare_max_train_support = 49
common_min_train_support = 500
```

這些設定在模型比較前固定。後續模型必須使用相同 horizon、Top-K、vocabulary 與 support 分組，確保比較公平。
