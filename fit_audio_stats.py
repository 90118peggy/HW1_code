"""分別以 A、B 全部 train 音訊，建立可保存及重用的頻譜統計量。"""

import argparse
from pathlib import Path

import torch

from audio_common import AudioConfig
from audio_pipeline import fit_normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", choices=("A", "B", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("audio_stats"))
    parser.add_argument("--threads", type=int, default=2, help="CPU 計算執行緒數，避免過度占用")
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 cuda:0；GPU 請使用已分配的運算節點")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必須為正整數")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用；請檢查 GPU 配額、驅動與 PyTorch CUDA build，不會自動改用 CPU")
    names = ("A", "B") if args.dataset == "both" else (args.dataset,)
    paths = [args.output_dir.resolve() / f"dataset_{name}.json" for name in names]
    for path in paths:
        if path.exists():
            parser.error(f"{path} 已存在，請換輸出目錄以保留原統計量")
        for dataset in (args.data_root.resolve() / "dataset_A", args.data_root.resolve() / "dataset_B"):
            if path == dataset or dataset in path.parents:
                parser.error("統計量必須存於官方資料夾以外")
    for name, path in zip(names, paths):
        print(f"計算 dataset_{name}，只使用完整 train split…", flush=True)
        stats = fit_normalization(
            args.data_root / f"dataset_{name}", AudioConfig(), device=device,
            progress=lambda current, total: print(f"  {current}/{total}", flush=True),
        )
        stats.save(path)
        print(f"mean={stats.mean:.6f}, std={stats.std:.6f}, recordings={stats.recording_count}")
        print(f"已保存 {path}")


if __name__ == "__main__":
    main()
