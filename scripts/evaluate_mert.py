"""獨立評估 E1 checkpoint，輸出 validation 預測與混淆矩陣。"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

# 遠端主機不需要開啟圖形視窗，直接輸出圖片。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig
from data_pipeline.waveform_dataset import HW1WaveformDataset
from inference.mert_predictor import mert_recording_logits
from models.mert_classifier import FrozenMERTClassifier


@torch.no_grad()
def evaluate(model, loader, class_names, device, chunk_batch_size):
    model.eval()

    criterion = nn.CrossEntropyLoss()
    num_classes = len(class_names)

    # 列是真實類別，欄是預測類別。
    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )

    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_samples = 0
    rows = []

    for step, batch in enumerate(loader, start=1):
        targets = batch["target"].to(device)

        logits = mert_recording_logits(
            model,
            batch["waveforms"],
            chunk_batch_size=chunk_batch_size,
        )

        loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Validation loss 出現 NaN 或無限值")

        # 和訓練程式採用相同的排序方式。
        top3 = logits.topk(k=3, dim=1).indices
        probabilities = logits.softmax(dim=1)

        top1_correct = top3[:, 0] == targets
        top3_correct = (top3 == targets[:, None]).any(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += top1_correct.sum().item()
        total_top3 += top3_correct.sum().item()
        total_samples += batch_size

        targets_cpu = targets.cpu()
        top3_cpu = top3.cpu()
        probabilities_cpu = probabilities.cpu()

        for index, sample_id in enumerate(batch["sample_id"]):
            true_index = int(targets_cpu[index])
            predicted_indices = top3_cpu[index].tolist()
            predicted_index = predicted_indices[0]

            confusion[true_index, predicted_index] += 1

            row = {
                "sample_id": sample_id,
                "true_label": class_names[true_index],
                "top1_label": class_names[predicted_indices[0]],
                "top2_label": class_names[predicted_indices[1]],
                "top3_label": class_names[predicted_indices[2]],
                "correct_top1": int(predicted_index == true_index),
                "correct_top3": int(true_index in predicted_indices),
            }

            for class_index, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(
                    probabilities_cpu[index, class_index]
                )

            rows.append(row)

        if step == 1 or step % 20 == 0 or step == len(loader):
            print(
                f"Validation batch {step}/{len(loader)}",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError("Validation 沒有任何樣本")

    metrics = {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top3": total_top3 / total_samples,
        "samples": total_samples,
    }

    return metrics, rows, confusion


def save_confusion_plot(confusion, class_names, title, path):
    matrix = confusion.numpy()

    fig, ax = plt.subplots(figsize=(9, 7))
    plot = ax.imshow(matrix, cmap="Blues", vmin=0)

    ticks = list(range(len(class_names)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # 深色格子用白字，淺色格子用黑字。
    threshold = matrix.max() / 2

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            count = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
            )

    fig.colorbar(plot, ax=ax, label="Number of recordings")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.batch_size < 1 or args.chunk_batch_size < 1:
        parser.error("batch size 必須至少為 1")

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    torch.set_num_threads(2)
    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        parser.error("輸出目錄已存在，請使用新的資料夾名稱")

    for name in ("dataset_A", "dataset_B"):
        official_dir = (root / name).resolve()
        if output_dir == official_dir or official_dir in output_dir.parents:
            parser.error("輸出目錄不可放在官方資料夾內")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("experiment") != "E1_frozen_MERT_linear"
    ):
        raise ValueError("這支程式只接受目前 E1 格式的 checkpoint")

    dataset_name = checkpoint["dataset"]
    if dataset_name not in ("dataset_A", "dataset_B"):
        raise ValueError("Checkpoint 中的資料集名稱不合法")

    class_names = checkpoint["class_names"]
    audio_config = AudioConfig(**checkpoint["audio_config"])

    dataset = HW1WaveformDataset(
        root / dataset_name,
        "validation",
        config=audio_config,
    )

    if dataset.class_names != class_names:
        raise ValueError("Checkpoint 類別順序與資料集不一致")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = FrozenMERTClassifier(
        **checkpoint["model_config"]
    ).to(device)

    model.classifier.load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True,
    )

    if model.processor.to_dict() != checkpoint["processor_config"]:
        raise ValueError("目前的波形處理器設定與 checkpoint 不一致")

    print("Dataset:", dataset_name)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation samples:", len(dataset))
    print("Classes:", class_names)

    metrics, rows, confusion = evaluate(
        model,
        loader,
        class_names,
        device,
        args.chunk_batch_size,
    )

    # 確認每首 validation 錄音剛好評估一次。
    actual_ids = [row["sample_id"] for row in rows]
    expected_ids = {row["sample_id"] for row in dataset.records}

    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("輸出的預測包含重複 ID")

    if set(actual_ids) != expected_ids:
        raise RuntimeError("輸出的預測 ID 與官方 validation 不一致")

    if int(confusion.sum()) != len(dataset):
        raise RuntimeError("混淆矩陣的總數與 validation 樣本數不一致")

    # 獨立評估結果必須能重現訓練時保存的指標。
    saved = checkpoint["validation"]

    for key in ("samples", "top1", "top3"):
        if metrics[key] != saved[key]:
            raise RuntimeError(f"{key} 無法重現 checkpoint 的結果")

    if not math.isclose(
        metrics["loss"],
        saved["loss"],
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError("Loss 無法重現 checkpoint 的結果")

    # 通過上述檢查後才建立輸出資料夾。
    output_dir.mkdir(parents=True)

    report = {
        "dataset": dataset_name,
        "split": "validation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "class_names": class_names,
        **metrics,
        "confusion_matrix": confusion.tolist(),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label / predicted_label", *class_names])
        for label, counts in zip(class_names, confusion.tolist()):
            writer.writerow([label, *counts])

    save_confusion_plot(
        confusion,
        class_names,
        title=(
            f"E1 {dataset_name} validation | "
            f"checkpoint epoch {checkpoint['epoch']}"
        ),
        path=output_dir / "confusion_matrix.png",
    )

    print("\n獨立評估與 checkpoint 指標一致")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Top-1: {metrics['top1']:.2%}")
    print(f"Top-3: {metrics['top3']:.2%}")
    print("Reports:", output_dir)


if __name__ == "__main__":
    main()