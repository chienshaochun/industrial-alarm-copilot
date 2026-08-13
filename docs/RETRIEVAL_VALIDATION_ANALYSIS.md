# 相似 Episode 檢索 Validation 分析

## 實驗範圍

本分析只使用 validation split 選擇檢索設定，沒有使用 test 指標調整模型。完整實驗包含 5,853 個 validation queries、兩個特徵版本、三個 future horizons 與三個 Jaccard relevance thresholds，共 18 組結果。

原始結果與可重現資訊寫入本機生成 artifacts：

- `data/processed/retrieval_validation_results.csv`
- `data/processed/retrieval_validation.metadata.json`

## 凍結設定

Validation 完成後選定以下設定：

| 項目 | 選定值 |
|---|---|
| Feature version | `alarm_plus_shape_v1` |
| Top-K | 5 |
| Candidate policy | `expanding_history` |
| Alarm weight | 1.0 |
| Shape weight | 1.0 |
| Future horizon | 6 小時 |
| Jaccard threshold | 0.3 |

## 選擇理由

1 小時 horizon 的 evaluation coverage 僅 50.8%，約一半 query 在時間窗內沒有後續 alarms。24 小時雖有 97.9% coverage，但可能將距離原 episode 較遠的系統反應納入 proxy。6 小時取得 87.8% coverage，在局部後續反應與資料覆蓋之間較平衡。

Jaccard threshold 0.1 在 6 小時設定下將約 60.9% 的合法候選判為 relevant，區分力不足；threshold 0.5 則使 Hit@5 降至約 48%。Threshold 0.3 的 relevant candidate share 約 18.3%，同時保留合理的命中率與排序辨識力。

在 6 小時、threshold 0.3 下，加入 shape 後相較 alarm-only 有小幅改善：

| 指標 | `alarm_tfidf_v1` | `alarm_plus_shape_v1` |
|---|---:|---:|
| Hit@5 | 0.758 | 0.775 |
| Precision@5 | 0.365 | 0.380 |
| MRR | 0.510 | 0.530 |
| NDCG@5 | 0.365 | 0.380 |

選定版本的 Precision@5 為 0.380，隨機 Precision@5 為 0.183，Precision lift 約 2.48 倍。

## Recall@5 的解讀

選定設定的 Recall@5 為 0.000467，數值很低，但 relevant 歷史候選很多，而產品固定只回傳五筆案例。其平均理論最大 Recall@5 為 0.002974，Recall efficiency 為 0.380。因此 Recall 的絕對值不能單獨解讀為檢索失敗，必須與 Precision@5、Precision lift、MRR、NDCG@5 及 Top-K 理論上限一起閱讀。

## 限制

ALPI 沒有人工標註的相似故障案例。本階段的 relevance 只表示兩個 episodes 在固定時間窗內的後續匿名 alarm code 集合具有 Jaccard 重疊，不代表真實故障原因相同。Shape 的改善幅度也尚未經統計顯著性檢定。

此設定在 test 解封前凍結。Test 只用於一次最終報告，不可再用於調整 feature version、horizon、threshold、權重或 Top-K。
