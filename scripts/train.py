import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import NormalizationStats
from data_pipeline.audio_pipeline import HW1AudioDataset
from inference.predictor import recording_logits
from models.short_chunk_cnn import ShortChunkCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """走過一次完整 train，每首錄音提供一個隨機片段。"""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for step, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Training loss 出現 NaN 或無限值")

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.detach().argmax(dim=1) == targets
        ).sum().item()
        total_samples += batch_size

        if step == 1 or step % 20 == 0 or step == len(loader):
            print(
                f"  Train batch {step}/{len(loader)} | "
                f"loss={total_loss / total_samples:.4f}",
                flush=True,
            )

    return {
        "loss": total_loss / total_samples,
        "top1": total_correct / total_samples,
    }

@torch.no_grad()
def validate(model, loader, criterion, device):
    """固定九段評估；每首錄音合併成一組 logits。"""
    model.eval()

    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_samples = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        logits = recording_logits(model, features)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Validation loss 出現 NaN 或無限值")

        top3_indices = logits.topk(k=3, dim=1).indices
        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size

        total_top1 += (
            top3_indices[:, 0] == targets
        ).sum().item()

        total_top3 += (
            top3_indices == targets[:, None]
        ).any(dim=1).sum().item()

        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top3": total_top3 / total_samples,
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", choices=["A", "B"], required=True
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs 與 patience 必須至少為 1")
    if args.batch_size < 2 or args.val_batch_size < 1:
        parser.error("train batch 至少為 2；validation batch 至少為 1")
    if args.lr <= 0 or args.weight_decay < 0:
        parser.error("lr 必須為正，weight-decay 不可為負")

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]
    dataset_name = f"dataset_{args.dataset}"

    stats = NormalizationStats.load(
        root / "audio_stats" / f"{dataset_name}.json"
    )

    train_data = HW1AudioDataset(
        root / dataset_name, "train", stats
    )
    val_data = HW1AudioDataset(
        root / dataset_name, "validation", stats
    )

    # 分類器有 BatchNorm1d，訓練 batch 不可只剩一筆。
    if len(train_data) % args.batch_size == 1:
        parser.error(
            "此 batch-size 會讓最後一批只剩一筆，請改用其他大小，例如 16"
        )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_config = {
        "num_classes": 6,
        "base_channels": 128,
    }

    model = ShortChunkCNN(**model_config).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = args.output_dir.resolve()

    # 保護官方資料，並避免覆寫既有實驗。
    for name in ("dataset_A", "dataset_B"):
        official_dir = (root / name).resolve()
        if output_dir == official_dir or official_dir in output_dir.parents:
            parser.error("輸出目錄不可放在官方資料夾內")

    if output_dir.exists():
        parser.error("輸出目錄已存在，請為這次實驗換一個新名稱")

    output_dir.mkdir(parents=True)

    run_config = {
        **vars(args),
        "output_dir": str(output_dir),
        "model": model_config,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device),
    }

    (output_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history = []
    best_top1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(
        f"{dataset_name} | "
        f"train={len(train_data)} | "
        f"validation={len(val_data)}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_metrics = validate(
            model, val_loader, criterion, device
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "seconds": time.perf_counter() - start_time,
        }
        history.append(record)

        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Top-1={train_metrics['top1']:.1%}\n"
            f"Val loss={val_metrics['loss']:.4f} | "
            f"Top-1={val_metrics['top1']:.1%} | "
            f"Top-3={val_metrics['top3']:.1%} | "
            f"time={record['seconds']:.1f}s",
            flush=True,
        )

        top1_improved = val_metrics["top1"] > best_top1

        is_best = top1_improved or (
            val_metrics["top1"] == best_top1
            and val_metrics["loss"] < best_loss
        )

        if top1_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if is_best:
            best_top1 = val_metrics["top1"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "normalization_stats": asdict(stats),
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "validation": val_metrics,
                    "run_config": run_config,
                },
                output_dir / "best.pt",
            )

            print("已保存新的最佳模型", flush=True)

        if epochs_without_improvement >= args.patience:
            print("Validation Top-1 持續未提高，提前停止", flush=True)
            break

    print(
        f"\n完成：最佳 epoch={best_epoch}，"
        f"validation Top-1={best_top1:.1%}"
    )
    print("最佳模型：", output_dir / "best.pt")


if __name__ == "__main__":
    main()