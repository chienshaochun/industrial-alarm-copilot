# 第 5 階段驗收

## 1. 驗收範圍

第 5 階段以 Stage 4 的 derived alarm episodes 為查詢單位，建立時間安全、可解釋、可重現的相似歷史 episode 檢索器，並使用 validation 選型及 test 最終驗收。

本階段完成 retrieval，不產生根因、維修建議或自然語言摘要。ALPI 的匿名 alarm code 不被解讀為具體故障語意。

## 2. 檢索契約與時間安全

- Query 是一個已完成的 derived episode。
- 候選必須滿足 `candidate.end_time < query.start_time`。
- 用於評估候選的 future outcome 也必須在 query 開始前完整可得。
- `expanding_history` 允許較晚的 query 使用當時已完整發生的 train、validation 或較早 test 歷史。
- TF-IDF vocabulary、IDF 與 shape scaler 只使用 train episodes 擬合。
- Machine ID 不進入相似度特徵，但輸出保留 `same_machine` 供解讀。
- 相同分數依候選結束時間與 incident ID 穩定排序。

## 3. 特徵與 Top-K

本階段比較兩個 deterministic 特徵版本：

1. `alarm_tfidf_v1`：將 episode 內的匿名 alarm code 序列轉成 train-fitted TF-IDF。
2. `alarm_plus_shape_v1`：在 alarm TF-IDF 外加入 log-transformed event count、duration 與 distinct alarm count，使用 train-fitted StandardScaler，兩個 block 各自正規化後以 1:1 權重結合。

使用 exact cosine similarity 搜尋合法歷史候選，固定回傳 Top-5。輸出包含 rank、similarity、incident ID、時間、設備是否相同與共同 alarm codes，可作為後續 Copilot 的引用證據。

## 4. Outcome Proxy 與評估

每個 episode 的 outcome 定義為同設備在 episode 結束後固定時間窗內出現的 alarm code 集合。Query 與候選 outcome 的 Jaccard similarity 達到 threshold 時，候選被視為 relevant。

此 relevance 是可重現的 proxy，不是真實相似故障標籤。評估同時報告：

- evaluation coverage 與 relevant candidate density；
- Hit@5、Precision@5、Recall@5；
- Maximum Recall@5 與 Recall Efficiency@5；
- Random Precision@5 與 Precision Lift@5；
- MRR 與 NDCG@5。

## 5. Validation 選型與凍結

Validation 使用 5,853 個 queries，比較兩個 feature versions、1／6／24 小時 horizons 與 0.1／0.3／0.5 thresholds，共 18 組設定。

在查看 test 前凍結：

| 項目 | 選定值 |
| --- | --- |
| Feature | `alarm_plus_shape_v1` |
| Top-K | 5 |
| Candidate policy | `expanding_history` |
| Alarm／Shape weight | 1.0／1.0 |
| Future horizon | 6 小時 |
| Jaccard threshold | 0.3 |

選型設定 commit 為 `c6c3cea`。Test CLI 不提供臨時覆寫上述設定的參數。

## 6. 最終 Test 結果

Test 共包含 6,364 個 episodes，正式程式版本為 `801600eeba5556955ec5aa8053493def36fb8255`。

| 指標 | Test |
| --- | ---: |
| Evaluation coverage | 86.61% |
| Hit@5 | 77.98% |
| Precision@5 | 38.94% |
| Random Precision@5 | 18.68% |
| Mean Precision Lift@5 | 2.45 倍 |
| Recall@5 | 0.000367 |
| Maximum Recall@5 | 0.003059 |
| Recall Efficiency@5 | 38.95% |
| MRR | 0.547 |
| NDCG@5 | 0.392 |

結果與 validation 接近，沒有觀察到明顯的時間外推崩落。低 Recall@5 主要由大量 proxy-relevant 候選與固定五筆輸出造成，不以放大 K 回頭調整 v1。

## 7. 自動化與效能驗收

完整本機測試結果為 85 passed，涵蓋：

- document 建立與重複 alarm 保留；
- train-only TF-IDF、shape scaling 與特徵對齊；
- 時間安全候選與 outcome availability gate；
- exact deterministic Top-K 搜尋；
- outcome Jaccard 與 ranking metrics；
- validation grid、診斷指標與結果 artifacts；
- 凍結設定解析與 test-only evaluator；
- Parquet orchestration、metadata 與 CLI。

經向量化 outcome、document 建立及重用 search index 後，100-query validation smoke run 由約 60.3 秒降至約 26.9 秒；完整 validation 已成功執行 5,853 queries 與 18 組設定。

## 8. 正式 Artifacts

Validation：

- `data/processed/retrieval_validation_results.csv`
- `data/processed/retrieval_validation.metadata.json`

Test：

- `data/processed/retrieval_test_results.csv`
- `data/processed/retrieval_test.metadata.json`

Artifacts 不納入 Git；metadata 保存來源雜湊、設定、範圍、產生時間與 code version。Git 追蹤的中文分析文件保存選型理由與正式結果。

## 9. 已知限制

- Derived episodes 與 relevance 都是 heuristic／proxy，不是真實 incident 或故障標籤。
- Bag-of-alarms TF-IDF 不完整表達 alarm 順序與長距離依賴。
- Shape 改善幅度未做統計顯著性檢定。
- Test 只驗證 ALPI 內較晚時間區段，不能證明跨工廠泛化。
- 已查看 test 後不得以相同結果繼續挑選 K、權重、特徵或 threshold。
- 未來 reranker 或 UI「查看更多」功能必須標示為新版本或產品層選項。

## 10. 下一階段入口

第 6 階段將建立後續 alarm forecasting。先以透明的頻率、轉移或 Markov baseline 建立可比較基準，再評估適合目前硬體的輕量序列模型。預測結果未來可作為 retrieval reranker 的額外訊號，但不得回寫或改動本階段已封存的 v1 test 結論。
