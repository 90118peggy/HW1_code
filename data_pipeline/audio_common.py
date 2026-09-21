"""音訊設定、官方清單與統計工具；本檔只使用 Python 標準函式庫。"""

import csv
import hashlib
import json
import math
import sys
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

from data_pipeline.inspect_dataset import LABELS, SPLITS


@dataclass(frozen=True)
class AudioConfig:
    """音訊設定。"""
    sample_rate: int = 24000
    crop_seconds: float = 3.69
    eval_chunks: int = 9  # 用於評估的音訊片段數量
    n_fft: int = 1024 # 快速傅立葉變換的點數
    hop_length: int = 256 # 每個音訊片段的跳躍長度
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 12000.0
    log_floor: float = 1e-10
    mel_scale: str = "slaney"
    mel_norm: str = "slaney"
    center: bool = True
    pad_mode: str = "reflect"

    def __post_init__(self):
        if self.sample_rate <= 0 or self.crop_seconds <= 0 or not math.isfinite(self.crop_seconds):
            raise ValueError("取樣率與裁切秒數必須為正數")
        if self.eval_chunks < 2:
            raise ValueError("eval_chunks 至少為 2，才能包含音訊頭尾")
        if not 0 < self.hop_length <= self.n_fft < self.crop_samples:
            raise ValueError("須滿足 0 < hop_length <= n_fft < 裁切樣本數")
        if not 0 < self.n_mels <= self.n_fft // 2 + 1:
            raise ValueError("n_mels 超出有效範圍")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError("頻率範圍必須位於 0 至 Nyquist 頻率之間")
        if not math.isfinite(self.log_floor) or self.log_floor <= 0:
            raise ValueError("log_floor 必須是有限正數")
        if self.mel_scale != "slaney" or self.mel_norm != "slaney":
            raise ValueError("本版固定使用 Slaney mel 尺度與濾波器面積正規化")
        if not self.center or self.pad_mode != "reflect":
            raise ValueError("本版固定使用 center=True、reflect 邊界補值")

    @property
    def crop_samples(self):
        return round(self.sample_rate * self.crop_seconds)


def fixed_crop_starts(length, crop_samples, count):
    """回傳樣本索引，整數等距分布，精確包含第一段與最後一段。"""
    if length <= 0 or crop_samples <= 0 or count < 2:
        raise ValueError("音訊不可為空，裁切長度須為正，多段裁切至少為 2 段")
    end = max(0, length - crop_samples)
    return [i * end // (count - 1) for i in range(count)]


def read_pcm16_wav(path):
    """解碼官方 PCM16 WAV，回傳單聲道 float32 array 與原取樣率。

    /32768 是固定 PCM 數值換算，不是依每首歌音量重新標準化。
    """
    with wave.open(str(path), "rb") as handle:
        channels, width, rate, frames, compression, _ = handle.getparams()
        if width != 2 or compression != "NONE":
            raise ValueError(f"{path}: 本作業讀取器要求未壓縮 PCM signed 16-bit WAV")
        if channels < 1 or frames == 0 or rate <= 0:
            raise ValueError(f"{path}: 音訊為空或 WAV 標頭無效")
        raw = handle.readframes(frames)
    if len(raw) != frames * channels * width:
        raise ValueError(f"{path}: WAV 內容被截斷，實際資料長度不足")
    pcm = array("h")
    pcm.frombytes(raw)
    if sys.byteorder != "little":
        pcm.byteswap()
    if channels == 1:
        mono = array("f", (value / 32768.0 for value in pcm))
    else:
        mono = array("f", (sum(pcm[i:i + channels]) / (channels * 32768.0)
                           for i in range(0, len(pcm), channels)))
    return mono, rate


def read_records(dataset_dir, split):
    """只回傳指定官方 split；test 不需要 label，也不會讀取它的音訊來擬合。"""
    dataset_dir = Path(dataset_dir).resolve()
    if dataset_dir.name not in LABELS or split not in SPLITS:
        raise ValueError("請指定 dataset_A／dataset_B，以及 train／validation／test")
    with (dataset_dir / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "split", "label", "audio_path", "sha256"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("manifest 缺少必要欄位")
        rows = list(reader)
    ids = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("manifest 欄數不符")
        if not row["sample_id"] or row["sample_id"] in ids:
            raise ValueError("manifest 有空 ID 或重複 ID")
        ids.add(row["sample_id"])
        if row["split"] not in SPLITS:
            raise ValueError(f"未知 split: {row['split']}")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"{dataset_dir.name} 的 {split} 沒有資料")
    for row in selected:
        if split != "test" and row["label"] not in LABELS[dataset_dir.name]:
            raise ValueError(f"{row['sample_id']}: 訓練／驗證標籤缺少或不合法")
        if split == "test" and row["label"]:
            raise ValueError(f"{row['sample_id']}: 官方 test 標籤應為空")
        relative = Path(row["audio_path"])
        if not row["audio_path"] or relative.is_absolute() or dataset_dir not in (dataset_dir / relative).resolve().parents:
            raise ValueError(f"{row['sample_id']}: 音訊路徑必須在 dataset 內")
    return selected


def training_fingerprint(rows):
    """以 train 的 ID、標籤、路徑及官方音訊 hash 識別統計來源，與絕對路徑無關。"""
    keys = ("sample_id", "label", "audio_path", "sha256")
    content = [{key: row[key] for key in keys} for row in sorted(rows, key=lambda r: r["sample_id"])]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, count, mean, m2):
        """合併每批的數量、平均與平方離差和，不把所有頻譜留在記憶體。"""
        if count <= 0 or not all(math.isfinite(x) for x in (mean, m2)) or m2 < 0:
            raise ValueError("統計批次必須非空，且所有統計量有效")
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    @property
    def std(self):
        if not self.count:
            raise ValueError("尚未累積任何訓練頻譜")
        return math.sqrt(self.m2 / self.count)  # 母體標準差；不是 sample std。


@dataclass(frozen=True)
class NormalizationStats:
    dataset: str
    config: dict
    mean: float
    std: float
    value_count: int
    recording_count: int
    training_fingerprint: str
    class_names: list
    source_split: str = "train"
    format_version: int = 1

    def __post_init__(self):
        AudioConfig(**self.config)
        if self.dataset not in LABELS or self.source_split != "train" or self.format_version != 1:
            raise ValueError("統計量必須來自本版支援的資料集 train split")
        if self.class_names != sorted(LABELS[self.dataset]):
            raise ValueError("類別順序與資料集不符")
        if not math.isfinite(self.mean) or not math.isfinite(self.std) or self.std < 1e-6:
            raise ValueError("mean/std 無效或 std 太小，不能標準化")
        if self.value_count <= 0 or self.recording_count <= 0:
            raise ValueError("統計量必須來自非空資料")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
