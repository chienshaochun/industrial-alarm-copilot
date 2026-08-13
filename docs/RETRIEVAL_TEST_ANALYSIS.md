# 相似 Episode 檢索 Test 分析

## 評估目的

本報告記錄第五階段相似 episode 檢索器的最終 test 驗收。特徵版本、Top-K、候選策略、future horizon、relevance threshold 與特徵權重都在查看 test 指標前，依 validation 結果完成凍結。

- 選型設定 commit：`c6c3cea`
- Test 驗收程式 commit：`801600e`
- Test artifact 完整 code version：`801600eeba5556955ec5aa8053493def36fb8255`

Test 結果不得再用來調整上述設定。後續若加入 reranker、改變 K 或修改 relevance 定義，必須視為新的模型版本並重新設計評估。

## 凍結設定

| 項目 | 設定 |
| --- | --- |
| Feature version | `alarm_plus_shape_v1` |
| Alarm weight | 1.0 |
| Shape weight | 1.0 |
| Top-K | 5 |
| Candidate policy | `expanding_history` |
| Future horizon | 6 小時 |
| Jaccard relevance threshold | 0.3 |

TF-IDF vocabulary、IDF 與 shape scaler 只使用 train episodes 擬合。對每個 query，候選 episode 必須先於 query 開始時間結束，而且候選的 6 小時 outcome 也必須在 query 開始前完整可得。

## Validation 與 Test 結果

| 指標 | Validation | Test |
| --- | ---: | ---: |
| Evaluation coverage | 0.878 | 0.8661 |
| Relevant query share | 0.878 | 0.8660 |
| Relevant candidate share | 0.183 | 0.1868 |
| Hit@5 | 0.775 | 0.7798 |
| Precision@5 | 0.380 | 0.3894 |
| Random Precision@5 | 0.183 | 0.1868 |
| Mean Precision Lift@5 | 2.48 | 2.4532 |
| Recall@5 | 0.000467 | 0.000367 |
| Maximum Recall@5 | 0.002974 | 0.003059 |
| Recall Efficiency@5 | 0.380 | 0.3895 |
| MRR | 0.530 | 0.5472 |
| NDCG@5 | 0.380 | 0.3921 |

Test 的 coverage、候選 relevance 密度與 validation 接近；Hit@5、Precision@5、MRR 與 NDCG@5 沒有下降。這支持檢索器在 ALPI 較晚時間區段內具有穩定表現，但不構成跨工廠或跨資料集泛化證據。

Mean Precision Lift@5 是先對每個可評估 query 計算 precision lift，再取平均，因此不必等於整體平均 Precision@5 除以整體平均 Random Precision@5。

## Recall@5 解讀

Test Recall@5 為 0.000367。這個數字很低，主要原因是 relevance proxy 可能為單一 query 標出大量歷史候選，而介面固定只取五筆。即使排序完美，平均理論最大 Recall@5 也只有 0.003059。

因此本產品的主要問題是「從大量可能 relevant 的案例中，把少數較有用的案例放到前面」，而不是列出所有 relevant 案例。第五階段以 Hit@5、Precision@5、precision lift、MRR、NDCG@5 與 recall efficiency 共同評估，不用放大 K 來美化 Recall。

後續可在不改變 v1 結論的前提下探索：

- UI 預設 Top-5，另提供「查看更多」模式；
- 先取較大的候選集合，再利用 alarm sequence 或預測結果 rerank；
- 使用第六階段預測的 future alarm distribution，比對歷史案例的已知 outcome；
- 導入人工標註的相似故障或處置結果，取代目前的 proxy relevance。

## 結論與限制

選定檢索器在 test 上達到 77.98% Hit@5 與 38.94% Precision@5，平均排序品質約為隨機選取的 2.45 倍。Validation 與 test 結果相近，沒有觀察到明顯過度擬合。

ALPI 沒有故障原因、維修紀錄或人工相似案例標籤。Relevance 只表示兩個 episodes 的後續匿名 alarm code 集合具有至少 0.3 的 Jaccard 重疊，不代表相同故障、根因、風險或處置。上述結論只適用於本資料契約與評估定義。

## 正式 Artifacts

執行命令：

```powershell
python -m industrial_alarm_copilot test-retrieval
```

輸出：

- `data/processed/retrieval_test_results.csv`
- `data/processed/retrieval_test.metadata.json`

Metadata 保存來源 artifact SHA-256、凍結設定、產生時間與完整 Git code version。Processed artifacts 由 `.gitignore` 排除，正式指標則保存在本報告中。
