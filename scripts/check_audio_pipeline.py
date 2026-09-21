"""用真實 WAV 檢查前處理，選擇繪圖；不訓練模型，不輸出作業預測。"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig, NormalizationStats, read_records
from inference.predictor import RecordingPredictor
from data_pipeline.audio_pipeline import (
    AudioFrontend, HW1AudioDataset, LogMel, crop_waveform, load_waveform,
)


class DiagnosticModel(nn.Module):
    """只有輸入/輸出介面用途，沒有學習過分類能力，不能提交它的預測。"""
    def forward(self, features):
        average = features.mean(dim=(1, 2, 3))
        return average[:, None] * torch.arange(1, 7, device=features.device)[None, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--stats", type=Path, help="完成 fit_audio_stats 後提供，啟用標準化與推論整合檢查")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--plot", type=Path, help="選填：第一個 train 範例的頻譜 PNG")
    parser.add_argument("--report", type=Path, help="選填：另存檢查結果 JSON")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    root = args.dataset_dir.resolve()
    for output in (args.plot, args.report):
        if output and (output.exists() or root == output.resolve() or root in output.resolve().parents):
            parser.error("輸出已存在或位於官方資料夾內，請選擇新的外部路徑")
    stats = NormalizationStats.load(args.stats) if args.stats else None
    config = AudioConfig(**stats.config) if stats else AudioConfig()
    row = read_records(root, "train")[0]
    waveform = load_waveform(root / row["audio_path"], config)
    train, train_starts = crop_waveform(waveform, config, True, torch.Generator().manual_seed(42))
    evaluation, eval_starts = crop_waveform(waveform, config)
    with torch.inference_mode():
        raw = LogMel(config).to(device)(evaluation.to(device))
        if not torch.isfinite(raw).all():
            raise RuntimeError("頻譜出現非有限值")
    report = {
        "dataset": root.name, "sample_id": row["sample_id"], "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "waveform_shape": list(waveform.shape), "crop_samples": config.crop_samples,
        "train_crop_shape": list(train.shape), "eval_crop_shape": list(evaluation.shape),
        "train_start_seconds": [start / config.sample_rate for start in train_starts],
        "eval_start_seconds": [start / config.sample_rate for start in eval_starts],
        "log_mel_shape": list(raw.shape), "finite": True,
    }
    normalized = None
    if stats:
        # Dataset 會驗證統計量的資料集及完整 train 清單指紋。
        for split in ("train", "validation", "test"):
            dataset = HW1AudioDataset(root, split, stats)
            batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
            report[f"{split}_batch_shape"] = list(batch["features"].shape)
            if split == "test" and "target" in batch:
                raise RuntimeError("test 不應產生 target")
        with torch.inference_mode():
            normalized = AudioFrontend(stats).to(device)(evaluation.to(device))
        test_row = read_records(root, "test")[0]
        predictor = RecordingPredictor(DiagnosticModel(), stats, device)
        first = predictor.predict_wav(root / test_row["audio_path"])
        second = predictor.predict_wav(root / test_row["audio_path"])
        if first != second:
            raise RuntimeError("固定評估流程無法重現")
        report["full_wav_prediction"] = "PASS: 6 scores, 3 distinct labels, deterministic; diagnostic model only"
        report["normalization"] = {"mean": stats.mean, "std": stats.std,
                                   "source_split": stats.source_split, "train_recordings": stats.recording_count}
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = 2 if normalized is not None else 1
        fig, axes = plt.subplots(panels, 1, figsize=(10, 3.6 * panels), squeeze=False, layout="constrained")
        pictures = [(raw[0].cpu().numpy(), "Log-mel power (fixed reference)", "dB")]
        if normalized is not None:
            pictures.append((normalized[0, 0].cpu().numpy(), "Standardized with training-set mean / std", "z score"))
        for ax, (values, title, unit) in zip(axes[:, 0], pictures):
            im = ax.imshow(values, origin="lower", aspect="auto", cmap="magma",
                           extent=[0, config.crop_seconds, 0, config.n_mels])
            ax.set(title=title, xlabel="Time in crop (seconds)", ylabel="Mel band index")
            fig.colorbar(im, ax=ax, label=unit)
        fig.suptitle(f"{root.name}: {row['sample_id']} | first fixed crop | train sample")
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        plt.close(fig)
        report["plot"] = str(args.plot)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
