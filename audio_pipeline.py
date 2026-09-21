"""完整 WAV → 裁切 → log-mel → 固定訓練統計標準化 → 模型分數合併。"""

from dataclasses import asdict
from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from audio_common import (
    AudioConfig, NormalizationStats, RunningMoments, fixed_crop_starts,
    read_pcm16_wav, read_records, training_fingerprint,
)
from inspect_dataset import LABELS


def load_waveform(path, config):
    samples, rate = read_pcm16_wav(path)
    waveform = torch.frombuffer(samples, dtype=torch.float32).clone()
    if rate != config.sample_rate:
        # 帶低通濾波的重採樣，避免直接丟樣本造成混疊。
        from scipy.signal import resample_poly
        divisor = math.gcd(rate, config.sample_rate)
        waveform = torch.from_numpy(resample_poly(
            waveform.numpy(), config.sample_rate // divisor, rate // divisor,
        ).copy()).float()
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"{path}: 音訊為空或含非有限值")
    return waveform


def crop_waveform(waveform, config, training=False, generator=None):
    """回傳 [片段數, 樣本數] 與起點；短音訊只在右側補零。"""
    if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("裁切輸入必須是非空、有限值的單聲道波形")
    size = config.crop_samples
    last_start = max(0, waveform.numel() - size)
    if training:
        # 包含最後合法起點；不在每次呼叫時重新設定 seed。
        starts = [int(torch.randint(last_start + 1, (1,), generator=generator))]
    else:
        starts = fixed_crop_starts(waveform.numel(), size, config.eval_chunks)
    padded = F.pad(waveform, (0, max(0, size - waveform.numel())))
    return torch.stack([padded[start:start + size] for start in starts]), starts


class LogMel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Slaney 尺度：1 kHz 以下線性，以上對數。三角頻帶再按寬度正規化。
        # 直接使用 PyTorch，與遠端既有 CUDA 版相容，不依賴 torchaudio 的二進位版本。
        def hz_to_mel(hz):
            return hz * 3 / 200 if hz < 1000 else 15 + math.log(hz / 1000) * 27 / math.log(6.4)

        positions = torch.linspace(hz_to_mel(config.f_min), hz_to_mel(config.f_max), config.n_mels + 2)
        edges = torch.where(positions < 15, positions * 200 / 3,
                            1000 * torch.exp((positions - 15) * math.log(6.4) / 27))
        frequencies = torch.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1)
        ascending = (frequencies[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
        descending = (edges[2:, None] - frequencies[None]) / (edges[2:] - edges[1:-1])[:, None]
        bank = torch.minimum(ascending, descending).clamp_min(0)
        bank *= (2 / (edges[2:] - edges[:-2]))[:, None]
        self.register_buffer("mel_filters", bank)
        self.register_buffer("window", torch.hann_window(config.n_fft))
        if torch.any(bank.sum(dim=1) == 0):
            raise ValueError("有 mel 濾波器全零；請減少 n_mels 或增加 n_fft")

    def forward(self, chunks):
        # 固定 power=1 的 dB 參考，不減去各片段最大值，也不使用片段相關 top_db。
        # 因而同一片段獨立計算或放在 batch 裡，數值一致。
        # 即使未來 CNN 使用混合精度，聲音前處理仍保持 float32。
        with torch.autocast(device_type=chunks.device.type, enabled=False):
            spectrum = torch.stft(
                chunks.float(), n_fft=self.config.n_fft, hop_length=self.config.hop_length,
                win_length=self.config.n_fft, window=self.window,
                center=self.config.center, pad_mode=self.config.pad_mode, return_complex=True,
            )
            power = self.mel_filters @ spectrum.abs().square()
            return 10.0 * torch.log10(power.clamp_min(self.config.log_floor))


class AudioFrontend(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.config = AudioConfig(**stats.config)
        self.log_mel = LogMel(self.config)
        # buffer 不接受梯度更新，也會隨 .to(device) 移動和存入 state_dict。
        self.register_buffer("mean", torch.tensor(stats.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(stats.std, dtype=torch.float32))

    def forward(self, chunks):
        normalized = (self.log_mel(chunks) - self.mean) / self.std
        return normalized.unsqueeze(-3)  # [K, mel, time] → [K, 1, mel, time]


@torch.inference_mode()
def fit_normalization(dataset_dir, config=AudioConfig(), device="cpu", progress=None):
    """只讀 train 音訊，以固定九段估計全域統計；不接受 validation/test 參數。"""
    dataset_dir = Path(dataset_dir).resolve()
    records = read_records(dataset_dir, "train")
    feature = LogMel(config).to(device)
    moments = RunningMoments()
    for index, row in enumerate(records, start=1):
        waveform = load_waveform(dataset_dir / row["audio_path"], config)
        chunks, _ = crop_waveform(waveform, config, training=False)
        values = feature(chunks.to(device)).double()
        mean = values.mean()
        moments.update(values.numel(), mean.item(), ((values - mean) ** 2).sum().item())
        if progress and (index % 50 == 0 or index == len(records)):
            progress(index, len(records))
    return NormalizationStats(
        dataset=dataset_dir.name, config=asdict(config), mean=moments.mean, std=moments.std,
        value_count=moments.count, recording_count=len(records),
        training_fingerprint=training_fingerprint(records),
        class_names=sorted(LABELS[dataset_dir.name]),
    )


class HW1AudioDataset(Dataset):
    """training item=[1,M,T]；validation/test item=[K,1,M,T]。"""
    def __init__(self, dataset_dir, split, stats):
        self.dataset_dir = Path(dataset_dir).resolve()
        if stats.dataset != self.dataset_dir.name:
            raise ValueError("A 與 B 不可混用標準化統計量")
        train = read_records(self.dataset_dir, "train")
        if len(train) != stats.recording_count or training_fingerprint(train) != stats.training_fingerprint:
            raise ValueError("訓練清單與統計量來源不符，請重新計算統計量")
        self.records = read_records(self.dataset_dir, split)
        self.split = split
        self.frontend = AudioFrontend(stats)
        self.class_to_index = {label: index for index, label in enumerate(stats.class_names)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        waveform = load_waveform(self.dataset_dir / row["audio_path"], self.frontend.config)
        chunks, starts = crop_waveform(waveform, self.frontend.config, training=self.split == "train")
        features = self.frontend(chunks)
        item = {
            "sample_id": row["sample_id"],
            "features": features[0] if self.split == "train" else features,
            "crop_starts": torch.tensor(starts),
        }
        if self.split != "test":
            item["target"] = self.class_to_index[row["label"]]
        return item


def recording_logits(chunk_model, features):
    """驗證用：[B,K,1,M,T] → [B,類別數]，每首歌先平均片段 logits。"""
    if features.ndim != 5 or features.shape[2] != 1:
        raise ValueError("輸入必須是 [batch, chunks, 1, mel, time]")
    batch, chunks = features.shape[:2]
    logits = chunk_model(features.flatten(0, 1))
    if logits.ndim != 2 or logits.shape != (batch * chunks, 6) or not torch.isfinite(logits).all():
        raise ValueError("片段模型必須回傳有限的 [batch*chunks, 6] logits")
    return logits.reshape(batch, chunks, 6).mean(dim=1)


class RecordingPredictor:
    """未來 CNN 的完整 WAV 推論入口；呼叫者不用先人工裁切或轉頻譜。"""
    def __init__(self, chunk_model, stats, device="cpu"):
        self.device = torch.device(device)
        self.model = chunk_model.to(self.device)
        self.frontend = AudioFrontend(stats).to(self.device)
        self.class_names = stats.class_names

    @torch.inference_mode()
    def predict_wav(self, path):
        waveform = load_waveform(path, self.frontend.config)
        chunks, starts = crop_waveform(waveform, self.frontend.config, training=False)
        # 關閉 Dropout 和 BatchNorm 的訓練行為；結束後恢復模型原狀態。
        was_training = self.model.training
        self.model.eval()
        try:
            features = self.frontend(chunks.to(self.device))
            logits = recording_logits(self.model, features.unsqueeze(0))[0]
            probabilities = logits.softmax(dim=-1)
            order = torch.argsort(probabilities, descending=True, stable=True)[:3]
            return {
                "top3_labels": [self.class_names[index] for index in order.tolist()],
                "probabilities": probabilities.cpu().tolist(),
                "class_names": list(self.class_names),
                "logits": logits.cpu().tolist(),
                "crop_start_seconds": [start / self.frontend.config.sample_rate for start in starts],
            }
        finally:
            self.model.train(was_training)
