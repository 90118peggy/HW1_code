"""E1：Frozen MERT + 線性分類器的訓練與驗證。"""

import argparse
import json
import math
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import (
    AudioConfig,
    training_fingerprint,
)
from data_pipeline.waveform_dataset import HW1WaveformDataset
from inference.mert_predictor import mert_recording_logits
from models.mert_classifier import (
    FrozenMERTClassifier,
    MODEL_NAME,
    MODEL_REVISION,
)


def run_epoch(
    model,
    loader,
    device,
    chunk_batch_size,
    optimizer=None,
):
    """
    有 optimizer：執行 train，更新分類器。
    沒有 optimizer：執行 validation，不更新參數。
    """
    training = optimizer is not None

    # FrozenMERTClassifier 的 train() 會讓 encoder 持續保持 eval。
    model.train(training)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_samples = 0
    phase = "Train" if training else "Validation"

    for step, batch in enumerate(loader, start=1):
        # 波形保留在 CPU，交給模型內的 processor 處理。
        waveforms = batch["waveforms"]
        targets = batch["target"].to(device)

        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)

                # train: [B, S] → [B, 6]
                logits = model(waveforms)
            else:
                # validation: [B, K, S] → [B, 6]
                logits = mert_recording_logits(
                    model,
                    waveforms,
                    chunk_batch_size=chunk_batch_size,
                )

            loss = criterion(logits, targets)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"{phase} loss 出現 NaN 或無限值"
                )

            if training:
                loss.backward()
                optimizer.step()

        # 指標以歌曲數量計算。
        batch_size = targets.size(0)
        top3_indices = logits.detach().topk(k=3, dim=1).indices

        total_loss += loss.item() * batch_size
        total_top1 += (
            top3_indices[:, 0] == targets
        ).sum().item()
        total_top3 += (
            top3_indices == targets[:, None]
        ).any(dim=1).sum().item()
        total_samples += batch_size

        if step == 1 or step % 20 == 0 or step == len(loader):
            print(
                f"  {phase} batch {step}/{len(loader)} | "
                f"loss={total_loss / total_samples:.4f}",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError(f"{phase} 沒有讀取到任何樣本")

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top3": total_top3 / total_samples,
        "samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["A", "B"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--chunk-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    positive_counts = (
        args.epochs,
        args.batch_size,
        args.val_batch_size,
        args.chunk_batch_size,
        args.patience,
    )
    if min(positive_counts) < 1:
        parser.error("epochs、各 batch size 與 patience 必須至少為 1")

    if (
        not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error("lr 必須為有限正數，weight-decay 必須為有限非負數")

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]
    dataset_name = f"dataset_{args.dataset}"
    output_dir = args.output_dir.resolve()

    # 避免把實驗輸出寫進官方資料夾。
    for name in ("dataset_A", "dataset_B"):
        official_dir = (root / name).resolve()
        if output_dir == official_dir or official_dir in output_dir.parents:
            parser.error("輸出目錄不可放在官方資料夾內")

    if output_dir.exists():
        parser.error("輸出目錄已存在，請為這次實驗使用新名稱")

    audio_config = AudioConfig(
        sample_rate=24000,
        crop_seconds=3.69,
        eval_chunks=9,
    )

    train_data = HW1WaveformDataset(
        root / dataset_name,
        "train",
        config=audio_config,
    )
    val_data = HW1WaveformDataset(
        root / dataset_name,
        "validation",
        config=audio_config,
    )

    if train_data.class_names != val_data.class_names:
        raise RuntimeError("train 與 validation 類別順序不一致")

    # 歌曲打散使用自己的隨機數產生器。
    shuffle_generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model_config = {
        "num_classes": len(train_data.class_names),
        "model_name": MODEL_NAME,
        "revision": MODEL_REVISION,
    }
    model = FrozenMERTClassifier(**model_config).to(device)

    # E1 只更新分類器。
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_config = {
        **vars(args),
        "output_dir": str(output_dir),
        "experiment": "E1_frozen_MERT_linear",
        "model_config": model_config,
        "audio_config": asdict(audio_config),
        "class_names": train_data.class_names,
        "training_fingerprint": training_fingerprint(train_data.records),
        "pooling": "last_hidden_state_time_mean",
        "recording_aggregation": "mean_logits",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "transformers_version": version("transformers"),
        "huggingface_hub_version": version("huggingface-hub"),
    }

    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history = []
    best_top1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    print(
        f"{dataset_name} | train={len(train_data)} | "
        f"validation={len(val_data)}",
        flush=True,
    )
    print("Class names:", train_data.class_names, flush=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.chunk_batch_size,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.chunk_batch_size,
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

        # 和 CNN 相同：Top-1 優先，同分時比較 loss。
        top1_improved = val_metrics["top1"] > best_top1
        is_best = top1_improved or (
            val_metrics["top1"] == best_top1
            and val_metrics["loss"] < best_loss
        )

        # Early stopping 只看 Top-1 是否提高。
        stale_epochs = 0 if top1_improved else stale_epochs + 1

        if is_best:
            best_top1 = val_metrics["top1"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "format_version": 1,
                    "experiment": "E1_frozen_MERT_linear",
                    "classifier_state_dict": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.classifier.state_dict().items()
                    },
                    "model_config": model_config,
                    "audio_config": asdict(audio_config),
                    "processor_config": model.processor.to_dict(),
                    "class_names": train_data.class_names,
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "validation": val_metrics,
                    "run_config": run_config,
                },
                output_dir / "best.pt",
            )
            print("已保存新的最佳分類器", flush=True)

        if stale_epochs >= args.patience:
            print("Validation Top-1 持續未提高，提前停止", flush=True)
            break

    print(
        f"\n訓練完成：最佳 epoch={best_epoch}，"
        f"validation Top-1={best_top1:.1%}",
        flush=True,
    )

    # 重新建立模型，確認 checkpoint 能重現最佳驗證結果。
    del optimizer, model
    torch.cuda.empty_cache()

    checkpoint = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    reloaded_model = FrozenMERTClassifier(
        **checkpoint["model_config"]
    ).to(device)
    reloaded_model.classifier.load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True,
    )

    print("\n重新載入最佳 checkpoint，驗證結果……", flush=True)
    reloaded_metrics = run_epoch(
        reloaded_model,
        val_loader,
        device,
        args.chunk_batch_size,
    )

    saved_metrics = checkpoint["validation"]

    for key in ("top1", "top3", "samples"):
        if reloaded_metrics[key] != saved_metrics[key]:
            raise RuntimeError(f"重新載入後的 {key} 與存檔不一致")

    if not math.isclose(
        reloaded_metrics["loss"],
        saved_metrics["loss"],
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError("重新載入後的 loss 與存檔不一致")

    (output_dir / "reload_validation.json").write_text(
        json.dumps(reloaded_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Checkpoint 重新載入驗證通過", flush=True)
    print("最佳分類器：", output_dir / "best.pt", flush=True)


if __name__ == "__main__":
    main()