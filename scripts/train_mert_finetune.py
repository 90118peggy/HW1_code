import argparse
import gc
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig, training_fingerprint
from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.mert_partial_finetune import MERTPartialFinetune
from scripts.train_mert import run_epoch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def check_metrics(actual, expected):
    for key in ("top1", "top3", "samples"):
        if actual[key] != expected[key]:
            raise RuntimeError(
                f"{key} 不一致："
                f"重新評估={actual[key]}，預期={expected[key]}"
            )

    if not math.isclose(
        actual["loss"],
        expected["loss"],
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            f"Loss 不一致："
            f"重新評估={actual['loss']}，預期={expected['loss']}"
        )


def train_epoch(model, loader, device, optimizer):
    model.train()

    # 本版刻意關閉 MERT 的隨機訓練行為，但保留梯度。
    if model.encoder.training:
        raise RuntimeError("本版 E4 的 encoder 應維持 eval 模式")

    criterion = nn.CrossEntropyLoss()
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    total_loss = 0.0
    correct1 = 0
    correct3 = 0
    total = 0

    for step, batch in enumerate(loader, start=1):
        waveforms = batch["waveforms"]
        targets = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(waveforms)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss 含有 NaN 或 Inf")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        count = targets.numel()
        top3 = logits.detach().topk(3, dim=1).indices

        total_loss += loss.item() * count
        correct1 += (top3[:, 0] == targets).sum().item()
        correct3 += (
            top3 == targets[:, None]
        ).any(dim=1).sum().item()
        total += count

        if step == 1 or step % 20 == 0 or step == len(loader):
            print(
                f"  Train batch {step}/{len(loader)}"
                f" | loss={total_loss / total:.4f}",
                flush=True,
            )

    if total == 0:
        raise RuntimeError("Train loader 沒有樣本")

    return {
        "loss": total_loss / total,
        "top1": correct1 / total,
        "top3": correct3 / total,
        "samples": total,
    }


def save_checkpoint(path, model, metadata, epoch, validation):
    import os
    import shutil
    from pathlib import Path

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")

    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    # 預留模型大小、序列化額外空間，以及 512 MiB 緩衝。
    tensor_bytes = sum(
        value.numel() * value.element_size()
        for value in state.values()
    )
    required = int(tensor_bytes * 1.15) + 513 * 1024**2
    free = shutil.disk_usage(path.parent).free

    if free < required:
        raise OSError(
            f"Checkpoint 儲存空間不足："
            f"可用 {free / 1024**3:.2f} GiB，"
            f"預估至少需要 {required / 1024**3:.2f} GiB。"
            "原有 checkpoint 保留。"
        )

    checkpoint = {
        **metadata,
        "epoch": epoch,
        "validation": dict(validation),
        "model_state_dict": state,
    }

    created = False
    try:
        # 若已有同名暫存檔，停止並保留它，不直接覆寫。
        with temporary.open("xb") as file:
            created = True
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())

        # 完整寫入成功後，才替換正式檔案。
        temporary.replace(path)

    except BaseException:
        if created:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["A", "B"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--chunk-batch-size", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for name in (
        "epochs", "batch_size", "val_batch_size",
        "chunk_batch_size", "patience",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} 必須大於 0")

    for name in ("backbone_lr", "classifier_lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必須是有限正數")

    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay 必須是有限非負數")

    if not 0 <= args.seed < 2**32:
        raise ValueError("seed 必須介於 0～2**32-1")

    if not torch.cuda.is_available():
        raise RuntimeError("未偵測到 CUDA")

    output_dir = args.output_dir.resolve()

    for name in ("dataset_A", "dataset_B"):
        root = (PROJECT_ROOT / name).resolve()
        if output_dir == root or root in output_dir.parents:
            raise ValueError("不能將輸出寫入官方資料集")

    if output_dir.exists():
        raise FileExistsError(
            f"輸出資料夾已存在：{output_dir}\n"
            "請指定新的資料夾，保留原本的實驗。"
        )

    seed_everything(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda")

    feature_layer = {"A": 6, "B": 5}[args.dataset]
    dataset_name = f"dataset_{args.dataset}"
    source_path = (
        PROJECT_ROOT
        / "outputs/checkpoints"
        / f"{args.dataset}_E2_seed42"
        / f"layer_{feature_layer:02d}_best.pt"
    )

    source = torch.load(
        source_path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        source.get("format_version") != 1
        or source.get("experiment") != "E2_mert_layer_probes"
        or source["dataset"] != dataset_name
        or source["layer"] != feature_layer
    ):
        raise RuntimeError("E2 checkpoint 格式或設定不一致")

    config = AudioConfig(**source["audio_config"])
    train_data = HW1WaveformDataset(
        PROJECT_ROOT / dataset_name,
        split="train",
        config=config,
    )
    val_data = HW1WaveformDataset(
        PROJECT_ROOT / dataset_name,
        split="validation",
        config=config,
    )

    if not (
        train_data.class_names
        == val_data.class_names
        == source["class_names"]
    ):
        raise RuntimeError("類別順序不一致")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
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
        **source["model_config"],
        "feature_layer": feature_layer,
        "unfreeze_last_n": 2,
    }

    model = MERTPartialFinetune(**model_config).to(device)
    model.classifier.load_state_dict(
        source["classifier_state_dict"],
        strict=True,
    )

    if model.processor.to_dict() != source["processor_config"]:
        raise RuntimeError("Processor 設定與 E2 不一致")

    print(
        f"{dataset_name} | train={len(train_data)}"
        f" | validation={len(val_data)}"
    )
    print("Feature layer:", feature_layer)
    print("Unfrozen layers:", model.unfrozen_layers)

    print("\n微調前：完整重現 E2 validation……")
    initial_validation = run_epoch(
        model,
        val_loader,
        device,
        args.chunk_batch_size,
    )
    check_metrics(initial_validation, source["validation"])
    print(
        "E2 起點重現通過"
        f" | Top-1={initial_validation['top1']:.2%}"
        f" | Top-3={initial_validation['top3']:.2%}"
    )

    run_config = {
        **vars(args),
        "output_dir": str(output_dir),
        "model_config": model_config,
        "audio_config": asdict(config),
        "training_fingerprint": training_fingerprint(train_data.records),
        "source_e2_checkpoint": str(source_path),
        "source_e2_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "source_e2_epoch": source["epoch"],
        "encoder_mode": "eval_with_grad",
        "gradient_clip_norm": 1.0,
        "recording_aggregation": "mean_logits",
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    metadata = {
        "format_version": 1,
        "experiment": "E4_mert_partial_finetune",
        "dataset": dataset_name,
        "model_config": model_config,
        "audio_config": asdict(config),
        "processor_config": model.processor.to_dict(),
        "class_names": train_data.class_names,
        "initial_validation": initial_validation,
        "run_config": run_config,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "run_config.json", run_config)
    save_json(
        output_dir / "initial_validation.json",
        initial_validation,
    )

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    # Epoch 0 是尚未微調的 E2 起點。
    best_metrics = dict(initial_validation)
    best_epoch = 0
    save_checkpoint(
        best_path, model, metadata, 0, initial_validation
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    p for p in model.encoder.parameters()
                    if p.requires_grad
                ],
                "lr": args.backbone_lr,
            },
            {
                "params": model.classifier.parameters(),
                "lr": args.classifier_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    # 將訓練的隨機序列與前置 validation 分開。
    seed_everything(args.seed)
    history = []
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        train_metrics = train_epoch(
            model, train_loader, device, optimizer
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.chunk_batch_size,
        )

        top1_improved = val_metrics["top1"] > best_metrics["top1"]
        tie_with_lower_loss = (
            val_metrics["top1"] == best_metrics["top1"]
            and val_metrics["loss"] < best_metrics["loss"]
        )

        if top1_improved or tie_with_lower_loss:
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            save_checkpoint(
                best_path, model, metadata, epoch, val_metrics
            )
            print("已儲存新的最佳 E4 checkpoint")

        # 與先前實驗一致：只有 Top-1 提高才重設 patience。
        stale_epochs = 0 if top1_improved else stale_epochs + 1

        save_checkpoint(
            last_path, model, metadata, epoch, val_metrics
        )

        seconds = time.perf_counter() - start
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "seconds": seconds,
            "best_epoch": best_epoch,
        })
        save_json(output_dir / "history.json", history)

        print(
            f"Train loss={train_metrics['loss']:.4f}"
            f" | Top-1={train_metrics['top1']:.2%}"
        )
        print(
            f"Val loss={val_metrics['loss']:.4f}"
            f" | Top-1={val_metrics['top1']:.2%}"
            f" | Top-3={val_metrics['top3']:.2%}"
            f" | time={seconds:.1f}s",
            flush=True,
        )

        if stale_epochs >= args.patience:
            print("Validation Top-1 持續未提高，提前停止")
            break

    # 即使最佳模型是 epoch 0，也驗證訓練後的最後一輪
    # 是否完整保存了 MERT 和分類器。
    saved_last = torch.load(
        last_path,
        map_location="cpu",
        weights_only=True,
    )
    current_state = model.state_dict()
    saved_state = saved_last["model_state_dict"]

    if set(current_state) != set(saved_state):
        raise RuntimeError("最後一輪 checkpoint 的參數鍵不一致")

    for name, value in current_state.items():
        if not torch.equal(value.detach().cpu(), saved_state[name]):
            raise RuntimeError(f"最後一輪權重儲存不一致：{name}")

    print("\n最後一輪完整模型權重儲存檢查通過")

    del saved_state, saved_last, current_state
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    print("重新載入最佳 E4 checkpoint，驗證結果……")
    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True,
    )
    reloaded = MERTPartialFinetune(
        **best["model_config"]
    ).to(device)

    # 必須載入完整 state_dict，不能只還原 classifier。
    reloaded.load_state_dict(
        best["model_state_dict"],
        strict=True,
    )

    if reloaded.processor.to_dict() != best["processor_config"]:
        raise RuntimeError("重新載入後的 processor 設定不一致")

    for name, value in reloaded.state_dict().items():
        if not torch.equal(
            value.detach().cpu(),
            best["model_state_dict"][name],
        ):
            raise RuntimeError(f"重新載入權重不一致：{name}")

    reload_metrics = run_epoch(
        reloaded,
        val_loader,
        device,
        args.chunk_batch_size,
    )
    check_metrics(reload_metrics, best["validation"])

    save_json(
        output_dir / "reload_validation.json",
        reload_metrics,
    )

    print("\nE4 完整模型重新載入驗證通過")
    print(
        f"最佳 epoch={best['epoch']}"
        f" | Top-1={reload_metrics['top1']:.2%}"
        f" | Top-3={reload_metrics['top3']:.2%}"
    )

    if best["epoch"] == 0:
        print("最佳結果仍是 E2 起點；本次微調尚未超越起點。")

    print("最佳模型：", best_path)


if __name__ == "__main__":
    main()