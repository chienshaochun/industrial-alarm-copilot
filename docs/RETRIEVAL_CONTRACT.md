# 相似 Alarm Episode 檢索契約

## 1. 目的與範圍

第 5 階段針對一個已完成的 derived alarm episode，從當時可取得的歷史 episodes 中找出 Top-K 相似案例，並回傳可追溯的 incident ID、時間範圍、相似度與比較證據。

本階段是 evidence-grounded RAG 的 retrieval layer，尚不讓 LLM 生成答案，也不推論匿名 alarm 的設備零件語意、故障根因或維修方法。

相似只表示 episode 在所選統計特徵空間中接近，不代表兩者是同一種真實故障。

## 2. 輸入 Artifacts

檢索流程讀取：

| Artifact | 用途 |
| --- | --- |
| events.parquet | 取得每筆 alarm code 與原始事件證據 |
| incidents.parquet | 取得 episode 身分、時間範圍、split 與統計摘要 |
| incident_events.parquet | 透過 source_row 連接 episode 與 events |

三張表必須滿足第 4 階段契約：incident ID 集合一致、每個 source row 只對應一個 episode，且 mapping 的 event position 從 0 開始連續。

## 3. Query 定義

第一版 query 是一個已結束的完整 episode。它可以使用該 episode 內全部已觀察到的 alarm codes、event count、duration 與 distinct alarm count。

因此第一版支援離線案件調查，不宣稱支援 episode 尚未結束時的即時檢索。若未來加入即時模式，必須建立 observation cutoff，且特徵只能使用 cutoff 以前的事件。

## 4. 時間安全候選規則

候選 episode 必須滿足：

    candidate.end_time < query.start_time

規則採嚴格小於。候選若在 query 開始時仍未結束，或與 query 有任何時間重疊，就不能作為歷史證據。

流程必須先建立合法候選 mask，再在合法子集合計算相似度。禁止先從包含未來資料的完整 index 取固定 Top-K，之後才刪除不合法結果，因為這可能遺漏真正的歷史 Top-K。

第一版主要採 expanding-history policy：validation 或 test query 可以使用當時已經結束的較早 episodes，不限候選 split。這模擬系統隨時間累積歷史案件。所有可學習的特徵轉換參數仍只能以 train fit。

另保留 train-only candidate policy 作為診斷比較；兩種 policy 的評估結果不得混合報告。

## 5. 設備範圍

第一版候選池允許跨設備檢索，且 machine_id 不直接放入相似度向量。每筆結果另輸出 same_machine，讓使用者與評估報告比較同設備及跨設備案例。

原因是設備代碼本身只代表身分，不代表 alarm 行為相似。若直接 one-hot 編碼 machine_id，模型可能只學會優先返回同一設備，而不是比較 episode 內容。

未來可以加入 same-machine-only filter，但必須視為明確的 retrieval policy。

## 6. 特徵版本

第一版依序建立兩個可比較版本：

### 6.1 Alarm-only Baseline

將 episode 內的 alarm codes 視為一份 bag-of-codes 文件，保留重複出現次數，使用 train episodes fit TF-IDF。單一字元與純數字代碼必須被保留，不能沿用會忽略單字元 token 的預設文字 tokenizer。

TF-IDF vocabulary 與 IDF 只能以 train fit；validation、test 與查詢當下新增的歷史 episodes 只能 transform。Train vocabulary 以外的 alarm code 不可偷偷重新 fit，會被視為 out-of-vocabulary 並在診斷資料中記錄。

### 6.2 Alarm + Shape

在 alarm TF-IDF 之外，加入：

- log1p(event_count)
- log1p(duration_seconds)
- log1p(distinct_alarm_count)

數值轉換與縮放參數只使用 train fit。Alarm composition 與 shape 必須保留明確的 feature version 及權重，評估報告要分開比較 alarm-only 與 alarm-plus-shape，不能只保留表現較好的結果。

第 4 階段的 P95 flags、baseline scope 與 machine ID 第一版不作為相似度特徵。它們可供結果解釋與切片分析，但加入向量前必須另做實驗，避免由人工規則或設備身分主導相似度。

## 7. 相似度與排序

第一版使用 cosine similarity。流程為：

1. 依 query start time 過濾合法候選；
2. 將 query 與候選轉成同一 feature version；
3. 在合法候選集合計算 exact cosine similarity；
4. 依 similarity 由高到低排序；
5. 相同 similarity 時，較近期的 candidate end time 優先；
6. 若仍相同，依 candidate incident ID 字串排序，確保結果可重現。

若合法候選少於 K，回傳實際可用數量。Query 自身不得出現在結果中。第一版優先確保 filter-first 的正確性，再以完整資料 benchmark 決定是否需要近似向量 index。

## 8. Retrieval Result Schema

每筆結果至少包含：

| 欄位 | 意義 |
| --- | --- |
| query_incident_id | 查詢 episode |
| rank | 從 1 開始的名次 |
| candidate_incident_id | 歷史候選 episode |
| similarity_score | cosine similarity |
| feature_version | 使用的特徵版本 |
| candidate_policy | expanding_history 或 train_only |
| candidate_machine_id | 候選設備 |
| candidate_start_time | 候選開始時間 |
| candidate_end_time | 候選結束時間 |
| same_machine | query 與候選是否同設備 |
| shared_alarm_codes | 兩者共有的匿名 alarm codes |

使用者介面呈現歷史證據時，必須引用 candidate incident ID 與時間範圍，不能只顯示相似度分數。

## 9. Relevance Proxy 與評估

ALPI 沒有人工標註的相似 incident pairs，因此不能直接量測真實故障相似性。第一版使用後續 alarm 集合是否相似，作為「這個歷史案例對預測後續行為是否有用」的 proxy relevance。

對每個 episode，在同設備內建立未來時間窗：

    episode.end_time < event.timestamp <= episode.end_time + horizon

Query 與 candidate 的未來 alarm code 集合以 Jaccard overlap 比較，達到設定門檻時視為 relevant。future horizon 與 Jaccard 門檻必須放在設定檔，使用 train／validation 決定後固定；test 只用於最終報告。

若評估或產品畫面使用 candidate 的後續 outcome，還必須滿足：

    candidate.end_time + horizon < query.start_time

確保 candidate 的完整未來結果在 query 發生前已經可觀察。

評估輸出包含：

- Hit@K
- Precision@K
- Recall@K
- MRR
- NDCG@K
- 可評估 query coverage
- 合法候選數量分布
- same-machine 與 cross-machine 結果切片

沒有未來 alarms 的 query、沒有合法候選或沒有 relevant 候選時，必須分開記錄 coverage，不能靜默刪除後只報告較容易的 queries。

這套 relevance 是下游效用 proxy，不是真實故障類型標籤。高檢索指標只能表示相似特徵可能連結到相似後續 alarm 集合。

## 10. 防止資料洩漏

- TF-IDF vocabulary、IDF、數值縮放與 feature weights 只能以 train fit。
- Validation 用於選擇 feature version、權重、future horizon 與 relevance 門檻。
- Test 在設計固定前不可查看或用來調參。
- 每個 query 都要重新套用時間 cutoff。
- Retrieval ranking 不可使用 query 的未來 alarm 集合；未來集合只用於離線評估。
- 若 candidate outcome 被顯示或用於 relevance，其完整 outcome window 必須早於 query。
- Episode 的完整特徵只適用於 episode 結束後的調查模式。

## 11. 可重現性與完整性

Retrieval artifact 必須保存：

- feature schema version；
- source incidents、mapping 與 events 的 SHA-256；
- train-fit vocabulary 與轉換參數；
- alarm／shape feature weights；
- candidate policy、K、future horizon 與 relevance 門檻；
- 程式 Git commit；
- validation／test 指標與 query coverage。

相同輸入、設定與程式版本必須產生相同排序。每筆 result 的 candidate 必須存在於 incidents artifact，且滿足時間安全規則。

## 12. 非目標與限制

本階段不會：

- 將相似 episode 宣稱為相同故障；
- 依匿名 alarm code 虛構零件或工程語意；
- 使用 LLM 決定相似度或重新排序；
- 使用 query 未來資料進行 ranking；
- 在沒有評估前直接導入近似向量資料庫；
- 產生維修或安全操作建議。

若未來取得維修工單或人工 similarity labels，應以真實標註重新評估目前的 proxy relevance。
