# Deployment artifact snapshot

此目錄保存 Streamlit Community Cloud 所需的固定展示快照。它與 `data/processed/` 的角色不同：

- `data/processed/` 是本機 pipeline 可重新產製的輸出，因此維持 Git ignore。
- `data/deployment/` 是經完整測試驗收後才更新的唯讀版本，讓乾淨 clone 能直接啟動 portfolio app。

快照由 ALPI V1 原始資料依 Stage 3 至 Stage 6 pipeline 產製。來源、授權與衍生資料界線請參閱 `NOTICE.md`、`data/README.md` 與各階段 acceptance 文件。

## Manifest

| 檔案 | bytes | SHA-256 |
| --- | ---: | --- |
| `events.parquet` | 9,797,148 | `d6393aae5b781176f9d02ca75a78c0f6c1a3303b502ac24ad20725ee3fb01161` |
| `incidents.parquet` | 1,455,161 | `6861a919fe257bef1f7caa3e71e91ab9c3958bfb911a2c45c33c36ee49949393` |
| `incident_events.parquet` | 1,379,779 | `d40b79d6ddd52030aa44269934d1288a3d2cb9cc3dc996dcb3861a995f5959db` |
| `forecast_model.json` | 487,547 | `5675227676c24abbe3cb733f1812398f14c32f0350e0c18f93c0ca49f6f57431` |
| `retrieval_test_results.csv` | 894 | `e30093a73e29c258227e91c0cf1c489a2635545d1bafd46d881ac4bac0f7fc3f` |
| `forecast_test_results.csv` | 481 | `f618337aff377f9242821c2351b76ea0dff76746faa3d9dfc994443d00d3fec0` |
| `forecast_test_support_groups.csv` | 420 | `3aa361b1009eff28147c088ab37f7851c211d7de8767c85f592f76d43f3d2de3` |

更新快照時必須重新計算 hash、執行完整測試，並使用獨立 Git commit 記錄原因。
