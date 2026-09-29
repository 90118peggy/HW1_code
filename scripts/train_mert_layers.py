"""E2：各層獨立線性分類器，共用凍結的 MERT。"""

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
from inference.mert_layer_predictor import mert_layer_recording_logits
from models.mert_classifier import MODEL_NAME, MODEL_REVISION
from models.mert_layer_probes import MERTLayerProbes


def write_json(path, content):
    path.write_text(
        json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def run_epoch(
    model,
    loader,
    device,
    chunk_batch_size,
    optimizer=None,
    active=None,
):
    """回傳每層各自的 loss、Top-1、Top-3。"""
    training = optimizer is not None

    if training and (active is None or not any(active)):
        raise ValueError("訓練時至少需要一個仍在學習的分類器")

    model.train(training)
    assert not model.encoder.training

    num_layers = model.num_layers
    criterion = nn.CrossEntropyLoss()

    total_loss = torch.zeros(num_layers, dtype=torch.float64)
    total_top1 = torch.zeros(num_layers, dtype=torch.long)
    total_top3 = torch.zeros(num_layers, dtype=torch.long)
    total_samples = 0

    phase = "Train" if training else "Validation"

    for step, batch in enumerate(loader, start=1):
        waveforms = batch["waveforms"]
        targets = batch["target"].to(device)

        with torch.set_grad_enabled(training):
            if training:
                optimizer.zero_grad(set_to_none=True)
                logits = model(waveforms)
            else:
                logits = mert_layer_recording_logits(
                    model,
                    waveforms,
                    chunk_batch_size=chunk_batch_size,
                )

            # logits: [B, L, 6]
            # 各層分別計算 batch 平均 loss。
            losses = torch.stack(
                [
                    criterion(logits[:, index, :], targets)
                    for index in range(num_layers)
                ]
            )

            if not torch.isfinite(losses).all():
                raise RuntimeError(f"{phase} loss 出現 NaN 或無限值")

            if training:
                active_indices = [
                    index
                    for index, enabled in enumerate(active)
                    if enabled
                ]

                # 只更新尚未提前停止的分類器。
                objective = losses[active_indices].sum()
                objective.backward()
                optimizer.step()

        batch_size = targets.size(0)

        # [B, L, 3]
        top3 = logits.detach().topk(k=3, dim=-1).indices

        # targets: [B] → [B, 1, 1]
        matches = top3 == targets[:, None, None]

        total_loss += losses.detach().cpu().double() * batch_size
        total_top1 += matches[:, :, 0].sum(dim=0).cpu()
        total_top3 += matches.any(dim=2).sum(dim=0).cpu()
        total_samples += batch_size

        if step == 1 or step % 20 == 0 or step == len(loader):
            mean_layer_loss = (total_loss / total_samples).mean().item()
            print(
                f"  {phase} batch {step}/{len(loader)} | "
                f"mean layer loss={mean_layer_loss:.4f}",
                flush=True,
            )

    if total_samples != len(loader.dataset):
        raise RuntimeError("實際處理的樣本數與 Dataset 長度不一致")

    return [
        {
            "loss": total_loss[index].item() / total_samples,
            "top1": total_top1[index].item() / total_samples,
            "top3": total_top3[index].item() / total_samples,
            "samples": total_samples,
        }
        for index in range(num_layers)
    ]


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

    if min(
        args.epochs,
        args.batch_size,
        args.val_batch_size,
        args.chunk_batch_size,
        args.patience,
    ) < 1:
        parser.error("epochs、batch size、patience 必須至少為 1")

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

    if output_dir.exists():
        parser.error("輸出目錄已存在，請使用新的實驗名稱")

    for name in ("dataset_A", "dataset_B"):
        official_dir = (root / name).resolve()
        if output_dir == official_dir or official_dir in output_dir.parents:
            parser.error("輸出目錄不可放在官方資料夾內")

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
        "num_classes": len(train_data.class_names),
        "model_name": MODEL_NAME,
        "revision": MODEL_REVISION,
    }
    model = MERTLayerProbes(**model_config).to(device)

    optimizer = torch.optim.Adam(
        model.classifiers.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_layers = model.num_layers

    run_config = {
        **vars(args),
        "output_dir": str(output_dir),
        "experiment": "E2_mert_layer_probes",
        "model_config": model_config,
        "audio_config": asdict(audio_config),
        "class_names": train_data.class_names,
        "training_fingerprint": training_fingerprint(train_data.records),
        "layers": list(range(1, num_layers + 1)),
        "pooling": "per_layer_time_mean",
        "recording_aggregation": "per_layer_mean_logits",
        "head_initialization": "identical_values_independent_parameters",
        "early_stopping": "per_layer_validation_top1",
        "torch_version": str(torch.__version__),
        "transformers_version": version("transformers"),
        "gpu": torch.cuda.get_device_name(device),
    }

    output_dir.mkdir(parents=True)
    write_json(output_dir / "config.json", run_config)

    # 每一層都有自己的最佳結果、等待次數與啟用狀態。
    best = [None] * num_layers
    stale_epochs = [0] * num_layers
    active = [True] * num_layers
    history = []

    print(
        f"{dataset_name} | train={len(train_data)} | "
        f"validation={len(val_data)} | layers={num_layers}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        active_before_epoch = active.copy()

        print(
            f"\nEpoch {epoch}/{args.epochs} | "
            f"active layers={sum(active)}",
            flush=True,
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.chunk_batch_size,
            optimizer=optimizer,
            active=active,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            args.chunk_batch_size,
        )

        for index in range(num_layers):
            if not active[index]:
                continue

            current = val_metrics[index]
            previous = best[index]

            top1_improved = (
                previous is None
                or current["top1"] > previous["top1"]
            )
            is_best = top1_improved or (
                current["top1"] == previous["top1"]
                and current["loss"] < previous["loss"]
            )

            stale_epochs[index] = (
                0 if top1_improved else stale_epochs[index] + 1
            )

            if is_best:
                filename = f"layer_{index + 1:02d}_best.pt"

                best[index] = {
                    "layer": index + 1,
                    "epoch": epoch,
                    "checkpoint": filename,
                    **current,
                }

                torch.save(
                    {
                        "format_version": 1,
                        "experiment": "E2_mert_layer_probes",
                        "layer": index + 1,
                        "classifier_state_dict": {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in
                            model.classifiers[index].state_dict().items()
                        },
                        "model_config": model_config,
                        "audio_config": asdict(audio_config),
                        "processor_config": model.processor.to_dict(),
                        "class_names": train_data.class_names,
                        "dataset": dataset_name,
                        "epoch": epoch,
                        "validation": current,
                        "run_config": run_config,
                    },
                    output_dir / filename,
                )

            print(
                f"Layer {index + 1:02d} | "
                f"train loss={train_metrics[index]['loss']:.4f} | "
                f"val loss={current['loss']:.4f} | "
                f"Top-1={current['top1']:.1%} | "
                f"Top-3={current['top3']:.1%}"
                + (" | saved best" if is_best else ""),
                flush=True,
            )

            if stale_epochs[index] >= args.patience:
                active[index] = False

                # 真正停用該分類器的梯度。
                # 避免它繼續被 optimizer 更新。
                model.classifiers[index].requires_grad_(False)
                for parameter in model.classifiers[index].parameters():
                    parameter.grad = None

                print(
                    f"Layer {index + 1:02d} 提前停止",
                    flush=True,
                )

        record = {
            "epoch": epoch,
            "active_layers": [
                index + 1
                for index, enabled in enumerate(active_before_epoch)
                if enabled
            ],
            "train": train_metrics,
            "validation": val_metrics,
            "seconds": time.perf_counter() - start_time,
        }
        history.append(record)

        write_json(output_dir / "history.json", history)
        write_json(output_dir / "layer_summary.json", best)

        print(f"Epoch time: {record['seconds']:.1f}s", flush=True)

        if not any(active):
            print("所有層都已提前停止", flush=True)
            break

    # 重新建立一個 MERT，載回各層各自最佳的分類器。
    del optimizer, model
    torch.cuda.empty_cache()

    reloaded_model = MERTLayerProbes(**model_config).to(device)
    saved_metrics = []

    for index, result in enumerate(best):
        checkpoint = torch.load(
            output_dir / result["checkpoint"],
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["layer"] != index + 1:
            raise RuntimeError("Checkpoint 層數不一致")

        reloaded_model.classifiers[index].load_state_dict(
            checkpoint["classifier_state_dict"],
            strict=True,
        )
        saved_metrics.append(checkpoint["validation"])

    print("\n重新載入各層最佳分類器，驗證結果……", flush=True)

    reloaded_metrics = run_epoch(
        reloaded_model,
        val_loader,
        device,
        args.chunk_batch_size,
    )

    for index, (actual, expected) in enumerate(
        zip(reloaded_metrics, saved_metrics)
    ):
        for key in ("samples", "top1", "top3"):
            if actual[key] != expected[key]:
                raise RuntimeError(
                    f"Layer {index + 1}: {key} 無法重現"
                )

        if not math.isclose(
            actual["loss"],
            expected["loss"],
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"Layer {index + 1}: loss 無法重現"
            )

    write_json(
        output_dir / "reload_validation.json",
        [
            {"layer": index + 1, **metrics}
            for index, metrics in enumerate(reloaded_metrics)
        ],
    )

    # 只依 validation 選層：
    # Top-1 越高越好；同分時 loss 越低越好。
    # 若兩者完全相同，以較小層號作固定的最後排序規則。
    selected = min(
        best,
        key=lambda result: (
            -result["top1"],
            result["loss"],
            result["layer"],
        ),
    )
    write_json(output_dir / "selected_layer.json", selected)

    print("\n各層最佳結果：", flush=True)
    for result in best:
        print(
            f"Layer {result['layer']:02d} | "
            f"epoch={result['epoch']} | "
            f"Top-1={result['top1']:.2%} | "
            f"Top-3={result['top3']:.2%}",
            flush=True,
        )

    print("十二層 checkpoint 重新載入驗證通過", flush=True)
    print(
        f"Validation 選出的最佳層：{selected['layer']} | "
        f"Top-1={selected['top1']:.2%}",
        flush=True,
    )
    print("Output:", output_dir, flush=True)


if __name__ == "__main__":
    main()