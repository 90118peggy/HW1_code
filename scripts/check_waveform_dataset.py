"""檢查 A/B 的波形 Dataset、官方切分與 batch 形狀。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig
from data_pipeline.waveform_dataset import HW1WaveformDataset


def main():
    # 只在程式開始時設定一次，不在每次裁切時重設。
    torch.manual_seed(42)
    torch.set_num_threads(2)

    root = Path(__file__).resolve().parents[1]
    config = AudioConfig(
        sample_rate=24000,
        crop_seconds=3.69,
        eval_chunks=9,
    )

    for dataset_name in ("dataset_A", "dataset_B"):
        print(f"\nDataset: {dataset_name}")

        for split in ("train", "validation", "test"):
            dataset = HW1WaveformDataset(
                dataset_dir=root / dataset_name,
                split=split,
                config=config,
            )

            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=split == "train",
                num_workers=0,
            )

            batch = next(iter(loader))
            waveforms = batch["waveforms"]

            expected_shape = (
                (2, config.crop_samples)
                if split == "train"
                else (2, config.eval_chunks, config.crop_samples)
            )

            assert tuple(waveforms.shape) == expected_shape
            assert waveforms.dtype == torch.float32
            assert torch.isfinite(waveforms).all()

            if split == "test":
                assert "target" not in batch
            else:
                targets = batch["target"]
                assert targets.shape == (2,)
                assert targets.dtype == torch.long
                assert ((targets >= 0) & (targets < 6)).all()

            # 同一筆評估資料重讀兩次，裁切起點與波形應完全一致。
            if split != "train":
                first = dataset[0]
                second = dataset[0]
                assert torch.equal(
                    first["crop_starts"], second["crop_starts"]
                )
                assert torch.equal(
                    first["waveforms"], second["waveforms"]
                )

            print(
                f"  {split}: recordings={len(dataset)}, "
                f"batch_shape={tuple(waveforms.shape)}, "
                f"has_target={'target' in batch}"
            )
            print("    第一首裁切起點：", batch["crop_starts"][0].tolist())

        print("  類別順序：", dataset.class_names)

    print("\nA/B 波形 Dataset 檢查通過")


if __name__ == "__main__":
    main()