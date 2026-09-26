"""提供 MERT 使用的波形，沿用官方切分與既有裁切方式。"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from data_pipeline.audio_common import AudioConfig, read_records
from data_pipeline.audio_pipeline import load_waveform, crop_waveform
from data_pipeline.inspect_dataset import LABELS


class HW1WaveformDataset(Dataset):
    """
    單筆資料的波形形狀：
        train:           [S]
        validation/test: [K, S]

    S：每段音訊的取樣點數。
    K：固定裁切的片段數。
    """

    def __init__(self, dataset_dir, split, config=None):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.split = split
        self.config = config if config is not None else AudioConfig()

        # 本次使用的 MERT-v1-95M 要求 24 kHz 音訊。
        if self.config.sample_rate != 24000:
            raise ValueError("MERT-v1-95M 的輸入取樣率必須是 24000 Hz")

        # 直接使用官方 manifest 的 split，不重新切分資料。
        self.records = read_records(self.dataset_dir, split)

        # 和既有 CNN 使用相同的類別順序。
        self.class_names = sorted(LABELS[self.dataset_dir.name])
        self.class_to_index = {
            name: index
            for index, name in enumerate(self.class_names)
        }

    def __len__(self):
        """此 split 中的錄音數量。"""
        return len(self.records)

    def __getitem__(self, index):
        """讀取一首錄音，並在取出資料時執行裁切。"""
        row = self.records[index]
        audio_path = self.dataset_dir / row["audio_path"]

        # 回傳單聲道、指定取樣率的 CPU float32 波形。
        waveform = load_waveform(audio_path, self.config)

        chunks, starts = crop_waveform(
            waveform,
            self.config,
            training=self.split == "train",
        )

        # train 只有一段，移除片段數量為 1 的維度。
        # validation/test 保留完整的 K 段。
        waveforms = chunks[0] if self.split == "train" else chunks

        item = {
            "sample_id": row["sample_id"],
            "waveforms": waveforms,
            "crop_starts": torch.tensor(starts, dtype=torch.long),
        }

        # 官方 test 沒有標籤，因此不建立 target。
        if self.split != "test":
            item["target"] = self.class_to_index[row["label"]]

        return item