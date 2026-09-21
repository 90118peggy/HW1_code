# 音樂分析 HW1

目前只完成第一步：檢查資料與官方切分。使用 Python 3 標準函式庫，不需安裝第三方套件或使用 GPU。

## 資料擺放

請自行將官方資料放在下列位置；Git 儲存庫不提供資料集。

```text
HW1_code/
├── inspect_dataset.py
├── README.md
├── .gitignore
├── tests/test_inspect_dataset.py
├── dataset_A/                  # 不上傳
│   ├── README.md               # 官方說明
│   ├── manifest.csv            # 官方樣本清單，內含 split 與 label
│   └── audio/                  # 1,290 個 WAV
└── dataset_B/                  # 不上傳
    ├── README.md
    ├── manifest.csv
    └── audio/                  # 1,002 個 WAV
```

## 執行檢查

在這個專案資料夾的終端機執行（若環境只提供 `python3`，以它替換 `python`）：

```bash
python -X utf8 inspect_dataset.py
```

資料不在程式旁邊時，用 `--data-root` 指定「包含 dataset_A 和 dataset_B 的父資料夾」：

```bash
python -X utf8 inspect_dataset.py --data-root /path/to/hw1_data
```

選擇另存完整檢查結果：

```bash
python -X utf8 inspect_dataset.py --output reports/dataset_inspection.json
```

報告已存在時，請換一個輸出檔名；程式不覆寫既有檔案，也不允許把報告寫進官方資料夾。
結束碼 `0` 表示本次檢查通過，`1` 表示資料有問題，`2` 表示命令參數或報告輸出設定有問題。

## 官方切分如何表示

依兩份官方 README，`manifest.csv` 每列代表一筆錄音，不需要另外製作隨機切分。

| 欄位 | 意義 |
|---|---|
| sample_id | 匿名樣本 ID |
| split | 官方指定的 train、validation 或 test |
| label | 年代或市場；test 的空字串是刻意隱藏的標籤 |
| audio_path | 相對於該 dataset 資料夾的 WAV 路徑 |
| duration_seconds | 官方清單記載的長度 |
| sample_rate | 官方清單記載的取樣率 |
| sha256 | 官方提供的檔案雜湊值；本步驟尚未重新計算驗證 |

少量實際範例（只顯示理解切分所需欄位）：

| sample_id | split | label | audio_path |
|---|---|---|---|
| A_005851d35e74 | train | 2010s | audio/A_005851d35e74.wav |
| A_007e1d02fd77 | validation | 1960s | audio/A_007e1d02fd77.wav |
| A_0072adbc81aa | test | 空字串 | audio/A_0072adbc81aa.wav |

例如 A 的音訊完整路徑是 `資料根目錄 / dataset_A / audio_path`。
不要用 `train_test_split` 重新分配，不要把 test 的空標籤改成任何類別。

## 本機實際檢查結果

下表來自本次提供的資料。換到遠端後，應再次執行程式，確認資料已完整複製。

| Dataset | train | validation | test | 合計 |
|---|---:|---:|---:|---:|
| A | 1026 | 132 | 132 | 1290 |
| B | 798 | 102 | 102 | 1002 |

| Dataset | 類別 | 每類 train 筆數 | 每類 validation 筆數 |
|---|---|---:|---:|
| A | 1960s、1970s、1980s、1990s、2000s、2010s | 171 | 22 |
| B | Brazil、Germany、Italy、Spain、UK、US | 133 | 17 |

- A、B 各自的重複 ID、各 split 配對的 ID 交集、缺少音訊、重複音訊路徑均為 0。
- train 與 validation 標籤完整；test 共 234 筆標籤全空，符合官方說明。
- 官方清單記載每筆 30 秒、24,000 Hz；此處不是對實際 WAV 的解碼驗證。
- 公開資料沒有歌手資訊，無法自行驗證歌手是否重疊；保留官方已提供的切分。
- 此步只檢查音訊檔存在，尚未檢查檔案內容、WAV 標頭、雜湊或不同 ID 是否包含相同音訊。

## 程式怎麼讀

1. `csv.DictReader` 把每一列讀成字典，例如 `row["split"]` 就是該筆的官方分組。
2. `by_split` 依現有欄位分組，不會重新抽樣、搬動檔案或修改 manifest。
3. `Counter` 計算 ID 和標籤各出現幾次；同一 ID 出現超過一次就是重複。
4. `set` 的 `&` 取得兩組 ID 的交集；交集非空就表示切分之間有重疊。
5. `Path.is_file()` 確認每列指向的音訊存在。
6. `errors` 收集問題；test 標籤為空是預期情況，train／validation 標籤為空才是錯誤。

## 檢查程式的測試

```bash
python -X utf8 -m unittest discover -s tests -v
```

測試只在暫存資料夾建立模擬清單與空的佔位檔，不使用或修改官方音訊。

本步沒有進行訓練、資料增強或標準化。下一步先理解清單讀取，再建立音訊載入流程。
