# Streamlit Community Cloud 部署

## 部署架構

Community Cloud 從 GitHub Repository root 建立 Python 3.12 環境，依 `requirements.txt` 安裝專案，並以 `app.py` 作為入口。

應用的 artifact 來源順序如下：

1. `INDUSTRIAL_ALARM_ARTIFACT_DIR` 指定的目錄。
2. 本機完整的 `data/processed/`。
3. Git 追蹤的 `data/deployment/` 展示快照。

`app.py` 會設定 `INDUSTRIAL_ALARM_PROJECT_ROOT`，因此專案安裝成 wheel 後仍能正確定位 Repository 內的 config 與 artifacts。

## 部署前驗證

### 本機環境

```powershell
python -m pytest
python scripts/validate_deployment.py
```

### 乾淨環境驗收結果

本階段另外使用 `git archive HEAD` 建立只含 Git 追蹤檔案的副本，並在全新 venv 中執行：

1. `python -m pip install -r requirements.txt`
2. 三頁 Streamlit AppTest。
3. 真正的 headless Streamlit server。
4. `/_stcore/health` HTTP health check。

結果：三頁無 `st.error`、指標卡完整產生，HTTP health endpoint 回傳 `200 ok`。

加入 deployment snapshot、root resolver 與 CI 後的完整本機驗收結果為 **153 passed in 158.07s**，隨後的 deployment smoke 亦為三頁通過。

GitHub Actions 會在 Ubuntu／Python 3.12 重複執行完整 pytest 與 deployment smoke，避免只在 Windows 本機成立。CI 額外安裝 CPU PyTorch 2.11 以覆蓋 Stage 6 GRU 測試；Community Cloud runtime 不需要 PyTorch，因此 `requirements.txt` 不包含這個大型 optional dependency。

## Community Cloud 設定

完成 deployment branch 合併後：

1. 前往 <https://share.streamlit.io/> 並以 GitHub 帳號登入。
2. 選擇 **Create app** → **Yup, I have an app**。
3. Repository：`chienshaochun/industrial-alarm-copilot`。
4. Branch：`main`。
5. Main file path：`app.py`。
6. App URL：選擇容易放進履歷的名稱，例如 `industrial-alarm-copilot`。
7. Advanced settings 的 Python version 選擇 `3.12`。
8. 本版本不需要 secrets；不要填入 API key。
9. 按下 **Deploy**，等待 build 與 health check 完成。

公開 Repository 部署完成後，應用預設為公開網址。取得 URL 後，應回填 README 並重新執行 smoke test。

## Snapshot 更新規則

`data/deployment/` 不是開發中的任意輸出。只有在以下條件都滿足時才能更新：

- Stage 3 至 Stage 6 pipeline 已重新驗收。
- 重新計算 README manifest 中的 SHA-256。
- `test_deployment_snapshot.py` 通過。
- 完整 pytest 與 deployment smoke 通過。
- 使用獨立 Git commit 說明 artifact 版本變更。
