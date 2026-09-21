# HW1 音訊處理：程式做了什麼，為什麼這樣寫

這一步把整首 WAV 變成模型可以使用的頻譜，並提供完整 WAV 的自動推論入口。官方的 train／validation／test 分配不變；不會產生新的裁切音訊檔，也不需要人工切 test。尚未訓練 CNN。

## 1. 標準化頻譜常見嗎？

常見。以 Audio Spectrogram Transformer（AST）為例，官方模型文件也強調輸入頻譜的 normalization，並提供 mean/std 設定。不過「常見」不代表每個模型必須用同一組數字：AST 的特定尺度與預訓練設定，不能直接拿來當作本作業的最佳值。[AST 官方文件](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)

我們現在準備的是從頭訓練 CNN 的 log-mel 輸入，採用原實作計畫指定的 **訓練集全域平均值與標準差**。A 和 B 各算一組：

```text
z = (log_mel - train_mean) / train_std
```

把它想成先給所有聲音能量一把共同的尺。例如 mean=-30、std=10 時，-40 變成 -1、-30 變成 0、-20 變成 1。這些是假設數字，真實值由程式計算。

這種轉換能讓輸入尺度更穩定，減少訓練對任意數值尺度的敏感程度。但準確率是否提高，仍須用驗證集比較；不能保證它自動解決過擬合。

### 為什麼只用 train？

平均值和標準差也是從資料學到的參數。如果先把 validation/test 算進去，就在訓練前用了保留資料的分布資訊。我們的 `fit_normalization()` 沒有讓你選 validation/test 的參數，固定只讀 train 音訊。

### 為什麼不是每首歌各自算？

每首歌各自減去自己的平均、除以自己的標準差，會抹去部分歌曲之間的整體能量與變化幅度差異。年代／市場分類可能用得到其中一些線索，所以第一版先讓所有歌共用 train 的一組尺度。這是設計理由，不是已證明能提升本資料集表現的結論。

全域標準化也不同於「每個 mel 頻帶分別算 mean/std」。兩種都有人使用，但不能把一種的統計檔拿去配另一種前處理。

## 2. 一首歌經過哪些步驟？

```text
train：
完整 WAV → 讀取／平均聲道／必要時重採樣
         → 隨機取 3.69 秒 → log-mel → train 全域標準化 → CNN

validation / test：
完整 WAV → 相同音訊讀取
         → 固定 9 段 → 各段 log-mel 與標準化
         → CNN 各段 logits → 同一首歌平均 logits → softmax → Top-3
```

logits 是模型尚未轉成機率的六個分數。本版依原計畫先平均 logits，再做 softmax；平均各段機率是另一種方法，結果通常不同，本版未採用。

## 3. 設定為什麼這樣選？

| 設定 | 本版值 | 白話說明 |
|---|---:|---|
| sample_rate | 24,000 Hz | 和官方 WAV 一致，一秒有 24,000 個樣本點 |
| crop_seconds | 3.69 秒 | 沿用規劃；每段 88,560 點 |
| eval_chunks | 9 | 固定多段，30 秒資料可涵蓋頭尾與中間且無空隙 |
| n_fft / win_length | 1,024 | 每次分析約 42.7 毫秒的局部聲音 |
| hop_length | 256 | 約每 10.7 毫秒產生下一個時間欄 |
| n_mels | 128 | 將頻率能量整理成 128 個符合聽覺刻度的頻帶 |
| f_min / f_max | 0 / 12,000 Hz | 24 kHz 取樣能表示的頻率範圍 |
| mel 尺度與濾波器 | Slaney、面積正規化 | 低頻較密、高頻較疏，避免寬頻帶純因寬度取得更大權重 |
| log | `10 * log10(power)` | 壓縮功率差距；參考功率固定為 1 |
| log_floor | 1e-10 | 避免靜音產生 `log(0)` 與無限值 |
| STFT 邊界 | center=True、reflect | 視窗以時間點為中心，音訊兩端使用鏡射補值 |

這些是可重現的初始設定，不是作業強制值，也不是已調到最佳的超參數。Slaney 濾波器面積正規化與 train mean/std 是兩個不同階段，不要混淆。[Mel 頻譜參數說明](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html)

30 秒音訊的九個固定起點（秒）為：

```text
0, 3.28875, 6.5775, 9.86625, 13.155,
16.44375, 19.7325, 23.02125, 26.31
```

最後一段在 30 秒結束。相鄰段略有重疊，不會改變標籤。短於 3.69 秒時，在波形右側補零；評估的九段相同，不會偷偷丟掉短檔。長於 30 秒的輸入仍會在全長均勻取九段，但不保證九段能覆蓋所有聲音。

## 4. 每個檔案負責什麼？

### `audio_common.py`：設定、清單、WAV 與統計

- `AudioConfig`：集中保存所有前處理設定，避免訓練和推論各寫一份而不同步。
- `read_pcm16_wav()`：使用 Python 的 `wave` 讀取官方 signed PCM16 WAV，檢查空音訊與截斷檔。數值除以 32768，平均聲道成 mono。這是固定的 PCM 換算，沒有每首歌的音量正規化。
- `fixed_crop_starts()`：使用整數樣本點計算起點，頭尾不因秒數四捨五入而少取。
- `read_records()`：從官方 `split` 欄位挑選資料，同時檢查 ID、標籤與路徑。不更動 manifest。
- `RunningMoments`：每讀一首就合併數量、平均與平方離差和，最後得到全域母體標準差；不需要把所有頻譜一次放進 RAM 或 GPU。
- `NormalizationStats`：保存 mean/std、完整音訊設定、類別順序、train 筆數與清單指紋。統計檔不允許覆寫既有檔案。

### `audio_pipeline.py`：裁切、頻譜、資料集與推論

- `load_waveform()`：將解碼結果轉為 float32 tensor。取樣率不同時，用 SciPy `resample_poly` 先低通濾波再重採樣，不直接改取樣率標籤或粗暴丟點。官方 24 kHz 資料不需要這一步。[SciPy 官方說明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
- `crop_waveform()`：train 每次讀取時重新抽起點，所以不同 epoch 可能看到同一首歌的不同位置；eval 永遠使用固定起點。不要在每次裁切函式裡重設 seed，否則會一直抽到相同段落。
- `LogMel`：Hann 視窗 → 短時傅立葉轉換（STFT）→ 複數振幅的平方得到功率 → 三角 mel 濾波器 → 固定參考的分貝。以原生 PyTorch 實作，可直接使用遠端既有 CUDA build，不依賴 torchaudio。
- `AudioFrontend`：用已保存的全域 mean/std 做標準化，增加 CNN 需要的單聲道維度。mean/std 存為 buffer，不會被梯度更新。
- `fit_normalization()`：固定讀全部 train，每首用相同的九段流程估計頻譜統計。以 float64 累積，降低數值誤差；GPU 模式在 GPU 算 STFT、mel 與批次統計，CPU 負責檔案讀取與最後的小量合併。
- `HW1AudioDataset`：方便交給 PyTorch DataLoader。train 回傳一段、validation/test 回傳九段。test 的字典沒有 `target` 欄位，避免把未知標籤當成真實訓練標籤。
- `recording_logits()`：將 `[B,K,1,M,T]` 攤平送進片段模型，再還原成每首歌並平均 logits。
- `RecordingPredictor.predict_wav()`：完整 WAV 的正式流程入口，自動完成讀取、裁切、頻譜、標準化、模型評估與 Top-3。推論時關閉 Dropout／BatchNorm 的訓練行為。

為什麼不用每段的最大能量當 dB 參考？這會讓一段頻譜的尺度依周圍片段改變。我們使用固定功率參考、不做每段 `top_db` 截斷，讓同一段單獨處理和放入 batch 處理得到一致輸入。[片段相關 dB 計算的注意事項](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.AmplitudeToDB.html)

### 兩個可以直接執行的入口

- `fit_audio_stats.py`：在遠端以全部 train 音訊建立 A、B 各一份統計 JSON。
- `check_audio_pipeline.py`：用真實音訊列印形狀、畫頻譜，提供統計檔後再檢查 DataLoader 與完整 WAV 推論。它的 `DiagnosticModel` 只檢查接線，沒有分類能力，不能用來提交作業。

## 5. 遠端如何執行？

這次主機的專案是 `/workspace/HW1_code`，Python 環境是 `/venv/main`。在 VS Code 遠端終端機：

```bash
cd /workspace/HW1_code
source /venv/main/bin/activate

# 確認 GPU 可被 PyTorch 使用
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 先執行本版測試，包含 GPU 與 CPU 一致性
python -m unittest discover -s tests -p test_audio_common.py -v
python -m unittest discover -s tests -p test_audio_pipeline.py -v

# 尚未標準化前，先檢查一首音訊
python check_audio_pipeline.py --dataset-dir dataset_A --device cuda

# 分別從 A、B 全部 train 算 mean/std；不會讀 validation/test 音訊
python fit_audio_stats.py --data-root . --dataset both --device cuda --output-dir audio_stats

# 使用已保存的統計量檢查完整流程，並畫圖
python check_audio_pipeline.py --dataset-dir dataset_A --stats audio_stats/dataset_A.json --device cuda --plot reports/dataset_A_spectrogram.png --report reports/dataset_A_pipeline.json
python check_audio_pipeline.py --dataset-dir dataset_B --stats audio_stats/dataset_B.json --device cuda --plot reports/dataset_B_spectrogram.png --report reports/dataset_B_pipeline.json
```

若統計檔或報告已存在，不必重算，直接使用；要另做實驗就換輸出目錄／檔名。
GPU 計算不代表全部步驟都會在 GPU：硬碟讀取與 WAV 解碼仍是 CPU 工作，也可能成為速度瓶頸。

`requirements-audio.txt` 列出本版所需套件。已有可用 CUDA PyTorch 時，保留它，只補缺少的依賴，不要安裝 CPU wheel 去蓋掉 GPU build。

## 6. 輸入模型的形狀

| 階段 | 單筆形狀 | batch=2 時 |
|---|---|---|
| 原始 30 秒波形 | `[720000]` | 本版在 Dataset 內逐筆讀取 |
| train 標準化頻譜 | `[1,128,346]` | `[2,1,128,346]` |
| validation/test 頻譜 | `[9,1,128,346]` | `[2,9,1,128,346]` |
| 每首歌的分類分數 | `[6]` | `[2,6]` |

128 是 mel 頻帶數；346 是本設定的時間欄數，不是 346 秒。`1` 是 CNN 的輸入聲道數；`9` 是同一首的片段數，不能把它們當成九首獨立歌曲計算評估指標。

## 7. 後續 CNN 怎麼接？

下列程式片段示意介面，`cnn` 將在下一階段建立：

```python
from audio_common import NormalizationStats
from audio_pipeline import HW1AudioDataset, RecordingPredictor, recording_logits
from torch.utils.data import DataLoader

stats = NormalizationStats.load("audio_stats/dataset_A.json")
train_data = HW1AudioDataset("dataset_A", "train", stats)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)

# cnn 必須接收 [N,1,128,346] 並回傳 [N,6]
# 訓練：cnn(batch["features"].to(device))
# 驗證：cnn.eval() 後用 recording_logits(cnn, batch["features"].to(device))
# 推論完整檔案：
predictor = RecordingPredictor(cnn, stats, device="cuda")
result = predictor.predict_wav("dataset_A/audio/某個測試ID.wav")
print(result["top3_labels"])
```

先用 `num_workers=0` 方便除錯；之後可依 CPU 與磁碟效能增加 worker。這只影響讀取效率，不應改動官方切分。

## 8. 必須一起保存的東西

未來訓練 CNN 時，除了模型權重，還要保存相應的 `audio_stats/dataset_A.json` 或 `dataset_B.json`。其中包含前處理設定和類別順序。模型本身另存權重，不能拿一份不相配的統計檔案來推論。

全域統計是針對 train 的固定九段集合計算，不是所有可能隨機片段的精確分布。因此這個集合標準化後應接近 mean=0、std=1；單首歌、隨機裁切集合、validation/test **不必**各自等於 0 和 1。

此流程僅適用於本版 CNN 的 log-mel 輸入。之後的 MERT 或音訊語言模型需要遵守各自的波形前處理要求，不應直接套用這份頻譜統計。
