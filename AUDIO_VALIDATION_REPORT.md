# 音訊前處理：遠端 GPU 實際驗證結果

本報告記錄本次真實執行結果，不是預期輸出。程式版本：`9eea7aa`。
程式逐段解釋見 [AUDIO_PROCESSING_GUIDE.md](AUDIO_PROCESSING_GUIDE.md)。

## 執行環境

- 遠端工作目錄：`/workspace/HW1_code`
- GPU：NVIDIA GeForce RTX 3060，12 GB VRAM
- Python：3.12.14，執行檔 `/venv/main/bin/python`
- 既有 PyTorch：2.14.0+cu130；保留原 CUDA build，沒有降版
- NumPy：2.5.3；補裝 SciPy 1.18.1、Matplotlib 3.11.2
- 頻譜及統計計算指定 `--device cuda`；檔案讀取與 WAV 解碼由 CPU 執行

## 資料上傳與完整性

兩份本機官方資料已上傳至遠端專案，沒有上傳到 GitHub。
全部 2,292 個 WAV 都重新計算 SHA-256，逐筆比對官方 manifest，失敗數為 0。

| Dataset | train | validation | test | WAV 總數 | 雜湊不符 |
|---|---:|---:|---:|---:|---:|
| A | 1026 | 132 | 132 | 1290 | 0 |
| B | 798 | 102 | 102 | 1002 | 0 |

遠端重新執行資料檢查：沒有重複 ID、切分間 ID 重疊或缺少音訊。test 標籤全空。
公開資料不含歌手資訊，不能自行核對歌手是否重疊；保留官方切分。

## 真實 train 統計量

使用全部 train，每首均勻取九段 3.69 秒，計算 128 mel 頻帶、346 時間欄的固定參考 log-power 頻譜；將所有段落、頻帶、時間格共同合併為一組全域統計。

| Dataset | 實際讀取 train 筆數 | mean | std（母體標準差） |
|---|---:|---:|---:|
| A | 1026 | -13.433779015295153 | 15.901908863374409 |
| B | 798 | -13.61382625843696 | 15.64405667890439 |

保存位置：`audio_stats/dataset_A.json`、`audio_stats/dataset_B.json`。
遠端與本機各保存一份；這兩份檔案在 Git 忽略規則中，未隨程式提交。
後續交付已訓練模型時，需要把相對應的統計 JSON 一起交付。

validation/test 音訊未參與 mean/std 計算。其資料只用於檔案完整性、切分檢查及統計量固定後的流程測試。

## 測試與整合結果

遠端執行 18 項本次新增測試，全部通過：

- 八項標準函式庫測試：PCM 換算、立體聲平均、截斷／空 WAV、固定裁切頭尾與覆蓋、短音訊起點、串流統計、統計保存，以及 train 清單讀取。
- 十項 PyTorch 測試：重採樣、隨機種子、靜音／補零、log-mel batch 一致性、train-only 統計、DataLoader 形狀、來源不符拒絕、logits 合併、完整 WAV 推論與 GPU／CPU 一致性。
- 其中 GPU／CPU 比較確實在 RTX 3060 執行，沒有跳過。

兩份資料都額外使用真實 WAV 在 GPU 檢查，結果一致：

| 階段 | 結果 |
|---|---|
| 30 秒 mono 波形 | `[720000]` |
| 隨機 train 裁切 | `[1,88560]` |
| 固定 eval 裁切 | `[9,88560]` |
| 九段 log-mel | `[9,128,346]`，全部為有限值 |
| train batch=2 | `[2,1,128,346]` |
| validation/test batch=2 | `[2,9,1,128,346]` |
| 隱藏 test 標籤 | Dataset 輸出中沒有 `target` |
| 完整 test WAV 推論 | 自動得到六個分數、三個不重複標籤，多次呼叫一致 |

**推論整合使用 DiagnosticModel 測試模型，沒有學到分類能力；這不是 CNN 訓練結果或可提交預測。**

## 頻譜圖與機器可讀結果

產物同時保存在遠端和本機的 `reports/`（不隨 Git 上傳）：

- `transfer_integrity.json`
- `dataset_A_pipeline.json`、`dataset_B_pipeline.json`
- `dataset_A_spectrogram.png`、`dataset_B_spectrogram.png`

兩張頻譜圖各顯示一筆 train 範例的第一個固定片段，上方為 log-mel，下方為標準化結果。兩張面板使用各自的色階單位：dB 與 z score。圖案相同是預期結果，因為全域標準化改變的是共同數值尺度，沒有重新排列時間與頻帶。

## 範圍與保留事項

- 未修改官方切分、原始音訊或隱藏 test 標籤。
- 尚未訓練 CNN，也未使用驗證集調整本版超參數。
- 遠端原先未提交的 `audio_processing.py`、`requirements.txt`、`tests/test_audio_processing.py` 均保留，沒有覆寫。
- 本次使用的模組是 `audio_common.py`、`audio_pipeline.py`，執行入口是 `fit_audio_stats.py` 與 `check_audio_pipeline.py`。
