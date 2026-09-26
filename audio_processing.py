"""WAV → 模型內裁切 → log-mel → train 統計標準化；保留官方 split。"""

import argparse
import csv
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from inspect_dataset import LABELS, SPLITS


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 24000
    crop_seconds: float = 5.0
    eval_crops: int = 5
    n_fft: int = 1024
    hop_length: int = 240
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float = 12000.0
    log_floor: float = 1e-10
    std_floor: float = 1e-5

    def __post_init__(self):
        for name in ("sample_rate", "eval_crops", "n_fft", "hop_length", "n_mels"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必須是正整數")
        if not math.isfinite(self.crop_seconds) or self.crop_samples < self.n_fft:
            raise ValueError("裁切長度必須有限，且至少包含 n_fft 個樣本")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError("頻率範圍必須在 0 到 Nyquist 頻率之間")
        if not all(math.isfinite(x) and x > 0 for x in (self.log_floor, self.std_floor)):
            raise ValueError("log_floor 與 std_floor 必須為有限正數")

    @property
    def crop_samples(self):
        return round(self.sample_rate * self.crop_seconds)


def read_wav(path, sample_rate=24000):
    """讀取未壓縮 PCM 8/16/24/32-bit WAV，縮放到 float32 並平均聲道。

    不做 peak normalization；不相符的取樣率直接報錯，不偷偷改變播放速度。
    IEEE float / 壓縮 WAV 不在 Python wave 的支援範圍。
    """
    try:
        with wave.open(str(path), "rb") as handle:
            rate, channels, width = (handle.getframerate(), handle.getnchannels(),
                                     handle.getsampwidth())
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path}: 請提供未壓縮 PCM WAV ({exc})") from exc
    if rate != sample_rate:
        raise ValueError(f"{path}: 取樣率 {rate} Hz，預期 {sample_rate} Hz；請先重取樣")
    if frames == 0 or channels < 1 or width not in (1, 2, 3, 4):
        raise ValueError(f"{path}: 空音訊或不支援的 PCM 格式")
    if len(raw) != frames * channels * width:
        raise ValueError(f"{path}: WAV 資料截斷")
    if width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = ((values ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    else:
        values = np.frombuffer(raw, dtype=f"<i{width}").astype(np.float32)
        values /= 2 ** (8 * width - 1)
    return torch.from_numpy(values.reshape(-1, channels).mean(axis=1).copy())


class ManifestAudioDataset(Dataset):
    """回傳整首 waveform；隱藏 test 標籤使用 -1，禁止用於 supervised loss。"""

    def __init__(self, dataset_dir, split, sample_rate=24000):
        self.root = Path(dataset_dir).resolve()
        if self.root.name not in LABELS or split not in SPLITS:
            raise ValueError("請指定 dataset_A 或 dataset_B，split 為 train/validation/test")
        self.split, self.sample_rate = split, sample_rate
        self.class_names = sorted(LABELS[self.root.name])
        with (self.root / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"{self.root}: {split} 沒有資料")
        for row in self.rows:
            path = (self.root / row["audio_path"]).resolve()
            if Path(row["audio_path"]).is_absolute() or self.root not in path.parents:
                raise ValueError("audio_path 必須位於資料集內")
            if split != "test" and row["label"] not in self.class_names:
                raise ValueError(f"未知或缺少標籤: {row['sample_id']}")
            if split == "test" and row["label"]:
                raise ValueError("test 預期為隱藏標籤")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"sample_id": row["sample_id"],
                "waveform": read_wav(self.root / row["audio_path"], self.sample_rate),
                "target": -1 if self.split == "test" else self.class_names.index(row["label"])}


def collate_audio(items):
    """保留不同長度的整首音訊；不要先補成 batch 最長，否則裁切位置會失真。"""
    return {"sample_id": [item["sample_id"] for item in items],
            "waveforms": [item["waveform"] for item in items],
            "targets": torch.tensor([item["target"] for item in items], dtype=torch.long)}


class AudioFrontend(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or AudioConfig()
        c = self.config
        self.register_buffer("window", torch.hann_window(c.n_fft))
        # HTK mel 刻度，三角濾波器、不做面積正規化。
        mel_min = 2595 * math.log10(1 + c.f_min / 700)
        mel_max = 2595 * math.log10(1 + c.f_max / 700)
        edges = 700 * (10 ** (torch.linspace(mel_min, mel_max, c.n_mels + 2) / 2595) - 1)
        freq = torch.fft.rfftfreq(c.n_fft, d=1 / c.sample_rate)
        rising = (freq[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
        falling = (edges[2:, None] - freq[None, :]) / (edges[2:] - edges[1:-1])[:, None]
        filters = torch.minimum(rising, falling).clamp_min(0)
        if (filters.sum(dim=1) == 0).any():
            raise ValueError("有空的 mel 頻帶，請增加 n_fft 或減少 n_mels")
        self.register_buffer("mel_filters", filters)
        self.register_buffer("mean", torch.zeros(1, c.n_mels, 1))
        self.register_buffer("std", torch.ones(1, c.n_mels, 1))
        self.register_buffer("stats_ready", torch.tensor(False))

    def get_extra_state(self):
        return asdict(self.config)

    def set_extra_state(self, state):
        if state != asdict(self.config):
            raise ValueError("checkpoint 音訊設定不符，請使用儲存時的 AudioConfig")

    def crop_waveform(self, waveform, *, training):
        if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
            raise ValueError("waveform 必須是非空、有限值的 mono 一維 tensor")
        waveform = waveform.to(device=self.window.device, dtype=torch.float32)
        size = self.config.crop_samples
        waveform = F.pad(waveform, (0, max(0, size - waveform.numel())))
        last = waveform.numel() - size
        if training:
            starts = [torch.randint(last + 1, ()).item()]
        elif self.config.eval_crops == 1:
            starts = [last // 2]
        else:
            starts = torch.linspace(0, last, self.config.eval_crops, dtype=torch.float64).round().long().tolist()
        return torch.stack([waveform[start:start + size] for start in starts])

    def log_mel(self, crops):
        """[segments, samples] → [segments, mel, time]；自然對數功率頻譜。"""
        # 前處理維持 float32，即使外層訓練使用混合精度。
        with torch.autocast(device_type=crops.device.type, enabled=False):
            spectrum = torch.stft(crops.float(), n_fft=self.config.n_fft,
                                  hop_length=self.config.hop_length, window=self.window,
                                  center=True, pad_mode="constant", return_complex=True)
            power = spectrum.abs().square()
            return (self.mel_filters @ power).clamp_min(self.config.log_floor).log()

    @torch.no_grad()
    def fit_statistics(self, dataset):
        """只接受 train，固定多段取樣；逐筆合併每個 mel 頻帶的母體 mean/std。"""
        if not isinstance(dataset, ManifestAudioDataset) or dataset.split != "train":
            raise ValueError("標準化統計只能從 ManifestAudioDataset 的 train 計算")
        if dataset.sample_rate != self.config.sample_rate:
            raise ValueError("dataset 與 frontend 取樣率不符")
        count, mean, m2 = 0, 0, 0
        for item in dataset:
            features = self.log_mel(self.crop_waveform(item["waveform"], training=False))
            values = features.transpose(0, 1).reshape(self.config.n_mels, -1).double()
            n = values.shape[1]
            batch_mean = values.mean(dim=1)
            batch_m2 = ((values - batch_mean[:, None]) ** 2).sum(dim=1)
            delta = batch_mean - mean
            m2 = m2 + batch_m2 + delta.square() * (count * n / (count + n))
            mean = mean + delta * (n / (count + n))
            count += n
        if not count:
            raise ValueError("train 沒有可計算的資料")
        self.mean.copy_(mean.float()[None, :, None])
        self.std.copy_((m2 / count).sqrt().clamp_min(self.config.std_floor).float()[None, :, None])
        self.stats_ready.fill_(True)

    def forward(self, waveforms):
        """[完整音訊, ...] → [batch, crops, 1, mel, time]。"""
        if not self.stats_ready.item():
            raise RuntimeError("請先以 train fit_statistics，或載入已計算統計的 checkpoint")
        if len(waveforms) == 0:
            raise ValueError("batch 不可為空")
        crops = torch.stack([self.crop_waveform(w, training=self.training) for w in waveforms])
        b, k, samples = crops.shape
        features = self.log_mel(crops.reshape(b * k, samples))
        features = (features - self.mean) / self.std
        return features.reshape(b, k, 1, *features.shape[-2:])


class AudioClassifier(nn.Module):
    """包住接收 [N, 1, mel, time]、輸出 [N, classes] logits 的分類器。

    train 回傳單段 logits；eval 回傳多段平均機率的 log，兩者均可交給
    CrossEntropyLoss，softmax 後皆為每首歌的預測機率。
    """

    def __init__(self, classifier, frontend):
        super().__init__()
        self.frontend = frontend
        self.classifier = classifier

    def forward(self, waveforms):
        features = self.frontend(waveforms)
        b, k = features.shape[:2]
        logits = self.classifier(features.flatten(0, 1))
        if logits.ndim != 2 or logits.shape[0] != b * k:
            raise ValueError("classifier 必須輸出 [batch * crops, classes] logits")
        logits = logits.reshape(b, k, -1)
        if self.training:
            return logits[:, 0]
        return torch.logsumexp(logits.float().log_softmax(dim=-1), dim=1) - math.log(k)

    @torch.inference_mode()
    def predict_files(self, paths):
        """輸入完整 WAV 路徑，逐首預測以限制記憶體；回傳 CPU [files, classes] 機率。"""
        modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            predictions = [self([read_wav(path, self.frontend.config.sample_rate)]).softmax(-1).cpu()
                           for path in paths]
            if not predictions:
                raise ValueError("paths 不可為空")
            return torch.cat(predictions)
        finally:
            for module, training in modes.items():
                module.training = training


def load_frontend(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    frontend = AudioFrontend(AudioConfig(**payload["config"]))
    frontend.load_state_dict(payload["state_dict"])
    return frontend


def main():
    parser = argparse.ArgumentParser(description="只讀取 train，計算並保存 log-mel 標準化統計")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-seconds", type=float, default=5.0)
    parser.add_argument("--eval-crops", type=int, default=5)
    args = parser.parse_args()
    root, output = args.dataset_dir.resolve(), args.output.resolve()
    if any(parent.name in LABELS for parent in (output, *output.parents)):
        parser.error("統計檔必須存放在官方資料夾之外")
    if output.exists():
        parser.error("輸出已存在，請使用新的檔名")
    config = AudioConfig(crop_seconds=args.crop_seconds, eval_crops=args.eval_crops)
    dataset = ManifestAudioDataset(root, "train", config.sample_rate)
    frontend = AudioFrontend(config)
    print(f"計算 {root.name} 的 {len(dataset)} 筆 train 統計…", flush=True)
    frontend.fit_statistics(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save({"config": asdict(config), "state_dict": frontend.state_dict(),
                    "dataset": root.name, "class_names": dataset.class_names}, handle)
    print(f"已儲存: {output}")


if __name__ == "__main__":
    main()
