"""E3：凍結 MERT，訓練層權重與線性分類器。"""

import argparse
import json
import math
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline.audio_common import (
    AudioConfig,
    training_fingerprint,
)
from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.mert_classifier import MODEL_NAME, MODEL_REVISION
from models.mert_weighted_classifier import MERTWeightedClassifier
from scripts.train_mert import run_epoch


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
        parser.error("epochs、batch size 與 patience 必須至少為 1")

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
    model = MERTWeightedClassifier(**model_config).to(device)

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if set(trainable) != {
        "layer_logits",
        "classifier.weight",
        "classifier.bias",
    }:
        raise RuntimeError("E3 的可訓練參數與預期不一致")

    # 同時更新層的原始分數與分類器。
    optimizer = torch.optim.Adam(
        list(trainable.values()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_config = {
        **vars(args),
        "output_dir": str(output_dir),
        "experiment": "E3_mert_weighted",
        "model_config": model_config,
        "audio_config": asdict(audio_config),
        "class_names": train_data.class_names,
        "training_fingerprint": training_fingerprint(train_data.records),
        "layers": list(range(1, model.num_layers + 1)),
        "layer_weighting": "global_softmax_uniform_initialization",
        "pooling": "per_layer_time_mean",
        "recording_aggregation": "mean_logits",
        "torch_version": str(torch.__version__),
        "transformers_version": version("transformers"),
        "gpu": torch.cuda.get_device_name(device),
    }

    output_dir.mkdir(parents=True)
    write_json(output_dir / "config.json", run_config)

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
    print("Trainable parameters:", list(trainable), flush=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        # 沿用 E1 的訓練與九段評估流程。
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

        layer_weights = (
            model.get_layer_weights().detach().cpu().tolist()
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "layer_weights": layer_weights,
            "seconds": time.perf_counter() - start_time,
        }
        history.append(record)
        write_json(output_dir / "history.json", history)

        print(
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Top-1={train_metrics['top1']:.1%}\n"
            f"Val loss={val_metrics['loss']:.4f} | "
            f"Top-1={val_metrics['top1']:.1%} | "
            f"Top-3={val_metrics['top3']:.1%} | "
            f"time={record['seconds']:.1f}s",
            flush=True,
        )
        print(
            "Layer weights:",
            " ".join(
                f"L{index + 1}={weight:.4f}"
                for index, weight in enumerate(layer_weights)
            ),
            flush=True,
        )

        top1_improved = val_metrics["top1"] > best_top1
        is_best = top1_improved or (
            val_metrics["top1"] == best_top1
            and val_metrics["loss"] < best_loss
        )

        # 與 E1 相同，early stopping 只看 Top-1 是否提高。
        stale_epochs = 0 if top1_improved else stale_epochs + 1

        if is_best:
            best_top1 = val_metrics["top1"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "format_version": 1,
                    "experiment": "E3_mert_weighted",
                    "model_config": model_config,
                    "audio_config": asdict(audio_config),
                    "processor_config": model.processor.to_dict(),
                    "class_names": train_data.class_names,
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "validation": val_metrics,
                    "run_config": run_config,

                    # 真正需要恢复的可訓練參數。
                    "layer_logits": model.layer_logits.detach().cpu().clone(),
                    "classifier_state_dict": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor
                        in model.classifier.state_dict().items()
                    },

                    # 方便人閱讀的 softmax 權重。
                    "layer_weights": layer_weights,
                },
                output_dir / "best.pt",
            )
            print("已保存新的最佳 E3 模型", flush=True)

        if stale_epochs >= args.patience:
            print("Validation Top-1 持續未提高，提前停止", flush=True)
            break

    print(
        f"\n訓練完成：最佳 epoch={best_epoch}，"
        f"validation Top-1={best_top1:.2%}",
        flush=True,
    )

    # 釋放原模型與 optimizer 的參考。
    del trainable, optimizer, model
    torch.cuda.empty_cache()

    checkpoint = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    reloaded_model = MERTWeightedClassifier(
        **checkpoint["model_config"]
    ).to(device)

    reloaded_model.classifier.load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True,
    )

    saved_layer_logits = checkpoint["layer_logits"]

    if saved_layer_logits.shape != reloaded_model.layer_logits.shape:
        raise RuntimeError("Checkpoint 的層權重形狀不一致")

    # 複製數值到既有 Parameter，保留它的參數身分。
    with torch.no_grad():
        reloaded_model.layer_logits.copy_(
            saved_layer_logits.to(device)
        )

    torch.testing.assert_close(
        reloaded_model.layer_logits.detach().cpu(),
        saved_layer_logits,
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        reloaded_model.get_layer_weights().detach().cpu(),
        torch.tensor(
            checkpoint["layer_weights"],
            dtype=torch.float32,
        ),
        rtol=1e-6,
        atol=1e-7,
    )

    if reloaded_model.processor.to_dict() != checkpoint["processor_config"]:
        raise RuntimeError("重新載入的波形處理器設定不一致")

    print("\n重新載入最佳 E3 checkpoint，驗證結果……", flush=True)

    reloaded_metrics = run_epoch(
        reloaded_model,
        val_loader,
        device,
        args.chunk_batch_size,
    )

    expected = checkpoint["validation"]

    for key in ("samples", "top1", "top3"):
        if reloaded_metrics[key] != expected[key]:
            raise RuntimeError(f"重新載入後的 {key} 無法重現")

    if not math.isclose(
        reloaded_metrics["loss"],
        expected["loss"],
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError("重新載入後的 loss 無法重現")

    write_json(
        output_dir / "reload_validation.json",
        reloaded_metrics,
    )
    write_json(
        output_dir / "best_layer_weights.json",
        [
            {"layer": index + 1, "weight": weight}
            for index, weight in enumerate(checkpoint["layer_weights"])
        ],
    )

    print("E3 層權重與分類器重新載入驗證通過", flush=True)
    print("最佳模型：", output_dir / "best.pt", flush=True)


if __name__ == "__main__":
    main()