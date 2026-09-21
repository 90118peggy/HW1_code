# HW1 資料夾架構與執行方式

本機與遠端均以 `HW1_code` 為根目錄。這次整理只調整檔案位置、import 與執行入口；官方資料、裁切策略、頻譜設定與標準化統計保持原樣。

## 實際架構

```text
HW1_code/
├── README.md
├── .gitignore
├── requirements-audio.txt
├── data_pipeline/
│   ├── __init__.py
│   ├── inspect_dataset.py
│   ├── verify_audio_hashes.py
│   ├── audio_common.py
│   └── audio_pipeline.py
├── models/
│   └── __init__.py
├── inference/
│   ├── __init__.py
│   └── predictor.py
├── scripts/
│   ├── __init__.py
│   ├── fit_audio_stats.py
│   └── check_audio_pipeline.py
├── tests/
│   ├── test_inspect_dataset.py
│   ├── test_audio_common.py
│   └── test_audio_pipeline.py
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── AUDIO_PROCESSING_GUIDE.md
│   └── AUDIO_VALIDATION_REPORT.md
├── dataset_A/                      # 原始資料
├── dataset_B/                      # 原始資料
├── audio_stats/                    # 已計算的 train 統計
├── reports/                        # 檢查報告與圖片
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
└── inspect_dataset.py              # 保留舊程式相容入口
```

`models/` 與 `outputs/` 已建立，但沒有 CNN 或訓練產物。後續才會建立 `models/short_chunk_cnn.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/predict.py`。

遠端原先未提交的 `audio_processing.py`、`requirements.txt`、`tests/test_audio_processing.py` 留在原處，不屬於本版流程。根目錄的 `inspect_dataset.py` 只轉接正式模組，讓舊程式的 import 仍可使用；檢查邏輯沒有複製兩份。

## 功能分工

| 位置 | 作用 |
|---|---|
| `data_pipeline/inspect_dataset.py` | 檢查官方 split、標籤、ID 與音訊路徑 |
| `data_pipeline/verify_audio_hashes.py` | 逐個 WAV 核對官方 SHA-256 |
| `data_pipeline/audio_common.py` | AudioConfig、PCM16 讀取、清單及統計保存 |
| `data_pipeline/audio_pipeline.py` | 波形裁切、LogMel、AudioFrontend、train 統計與 Dataset |
| `inference/predictor.py` | recording_logits 與 RecordingPredictor；多段分數合併、完整 WAV 推論 |
| `scripts/` | 解析命令列參數並串接上述模組 |
| `tests/` | 用模擬資料驗證行為，不修改官方音訊 |
| `docs/` | 教學、操作與實測結果 |

`__init__.py` 表示該資料夾是 Python 套件，方便用完整模組路徑 import。資料夾叫 `models`，避免與單一模型物件 `model` 混淆。

## 在遠端執行

以下指令均從根目錄執行，不要先切換到 `scripts/`。

```bash
cd /workspace/HW1_code
source /venv/main/bin/activate

# 官方切分檢查；預設資料位置仍是專案根目錄。
python -m data_pipeline.inspect_dataset

# 完整性檢查：讀取所有 WAV，耗時比切分檢查長。
python -m data_pipeline.verify_audio_hashes

# 使用既有統計檢查前處理與完整 WAV 推論，不訓練模型。
python -m scripts.check_audio_pipeline --dataset-dir dataset_A --stats audio_stats/dataset_A.json --device cuda
python -m scripts.check_audio_pipeline --dataset-dir dataset_B --stats audio_stats/dataset_B.json --device cuda
```

只有在尚無統計檔或前處理設定改變時，才另外計算；請選擇未存在的輸出目錄：

```bash
python -m scripts.fit_audio_stats --dataset both --device cuda --output-dir audio_stats_new
```

測試本版三個測試檔：

```bash
python -m unittest discover -s tests -p test_inspect_dataset.py -v
python -m unittest discover -s tests -p test_audio_common.py -v
python -m unittest discover -s tests -p test_audio_pipeline.py -v
```

執行入口採 `python -m 套件.模組`，例如 `python -m scripts.check_audio_pipeline`，不要再執行 `python check_audio_pipeline.py` 或 `python scripts/check_audio_pipeline.py`。

## 新的 import

```python
from data_pipeline.audio_common import AudioConfig, NormalizationStats
from data_pipeline.audio_pipeline import HW1AudioDataset, AudioFrontend
from inference.predictor import RecordingPredictor, recording_logits
```

模型訓練時，Dataset 先產生頻譜；推論時，RecordingPredictor 接收完整 WAV，內部呼叫相同前處理，再呼叫模型。前處理不需要 import 推論模組，因此不會形成循環依賴。

資料集、`audio_stats/`、`reports/` 與實際 `outputs/` 產物不隨這次提交上傳 GitHub。`outputs/` 僅以 `.gitkeep` 保留空資料夾結構。

## 整理後的驗證

- 本機 13 項標準函式庫測試通過，新的資料檢查入口可從專案根目錄找到 A、B。
- 遠端 RTX 3060 執行本版 23 項測試全部通過，包含 GPU／CPU 頻譜一致性。
- A、B 各以真實 WAV 執行新的 `scripts.check_audio_pipeline` 入口：頻譜形狀、固定裁切位置、train／validation／test batch 形狀，以及完整 WAV 推論介面皆通過。
- 沿用既有 A、B 統計檔，不需要重算；推論檢查仍使用 DiagnosticModel，沒有訓練 CNN。
