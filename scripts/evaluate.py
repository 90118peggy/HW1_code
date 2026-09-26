import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_pipeline.audio_common import NormalizationStats
from data_pipeline.audio_pipeline import HW1AudioDataset
from inference.predictor import recording_logits
from models.short_chunk_cnn import ShortChunkCNN


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("batch-size 必須至少為 1")

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    torch.set_num_threads(2)

    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()

    # 保護官方資料，並避免覆寫舊評估結果。
    for name in ("dataset_A", "dataset_B"):
        official_dir = (root / name).resolve()
        if output_dir == official_dir or official_dir in output_dir.parents:
            parser.error("輸出目錄不可放在官方資料夾內")

    if output_dir.exists():
        parser.error("輸出目錄已存在，請換一個新名稱")

    # 1. 先將 checkpoint 載入 CPU。
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    # 使用 checkpoint 內保存的前處理設定與統計。
    stats = NormalizationStats(
        **checkpoint["normalization_stats"]
    )

    dataset_name = checkpoint["dataset"]
    class_names = stats.class_names
    num_classes = len(class_names)

    if dataset_name != stats.dataset:
        raise ValueError("checkpoint 的資料集與標準化統計不一致")

    # 2. 建立相同架構，再載入訓練好的權重。
    model = ShortChunkCNN(
        **checkpoint["model_config"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    # 3. 只評估官方 validation。
    dataset = HW1AudioDataset(
        dataset_dir=root / dataset_name,
        split="validation",
        stats=stats,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_top1 = 0
    total_top3 = 0
    total_samples = 0

    confusion = torch.zeros(
        num_classes, num_classes, dtype=torch.int64
    )

    predictions = []

    print("Dataset:", dataset_name)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation samples:", len(dataset))
    print("Classes:", class_names)

    # 4. 每首錄音合併九段 logits，再計算指標。
    for step, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        logits = recording_logits(model, features)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError("評估 loss 出現 NaN 或無限值")

        top3 = logits.topk(k=3, dim=1).indices
        top1 = top3[:, 0]

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_top1 += (top1 == targets).sum().item()
        total_top3 += (
            top3 == targets[:, None]
        ).any(dim=1).sum().item()
        total_samples += batch_size

        # 搬回 CPU，累積混淆矩陣與每首錄音的預測。
        targets_cpu = targets.cpu()
        top3_cpu = top3.cpu()

        for sample_id, target, ranked in zip(
            batch["sample_id"],
            targets_cpu.tolist(),
            top3_cpu.tolist(),
        ):
            predicted = ranked[0]
            confusion[target, predicted] += 1

            predictions.append({
                "sample_id": sample_id,
                "true_label": class_names[target],
                "top1_label": class_names[ranked[0]],
                "top2_label": class_names[ranked[1]],
                "top3_label": class_names[ranked[2]],
            })

        if step == 1 or step % 10 == 0 or step == len(loader):
            print(
                f"Validation batch {step}/{len(loader)}",
                flush=True,
            )

    # 5. 整理整個 validation 的結果。
    metrics = {
        "dataset": dataset_name,
        "split": "validation",
        "checkpoint_epoch": checkpoint["epoch"],
        "samples": total_samples,
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top3": total_top3 / total_samples,
        "class_names": class_names,
        "confusion_matrix": confusion.tolist(),
    }

    # 確認每首錄音只被計算一次。
    assert total_samples == len(dataset)
    assert confusion.sum().item() == total_samples

    output_dir.mkdir(parents=True)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 這是 validation 分析檔，不是作業的 test 提交檔。
    with (output_dir / "validation_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "true_label",
                "top1_label",
                "top2_label",
                "top3_label",
            ],
        )
        writer.writeheader()
        writer.writerows(predictions)

    # 6. 畫混淆矩陣：列為正確類別，欄為預測類別。
    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(
        confusion.numpy(),
        cmap="Blues",
        vmin=0,
    )

    ax.set_xticks(range(num_classes), labels=class_names)
    ax.set_yticks(range(num_classes), labels=class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"{dataset_name} validation | "
        f"checkpoint epoch {checkpoint['epoch']}"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    threshold = confusion.max().item() / 2

    for row in range(num_classes):
        for column in range(num_classes):
            count = confusion[row, column].item()
            ax.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                color="white" if count > threshold else "black",
            )

    fig.colorbar(image, ax=ax, label="Number of recordings")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    # 7. 與訓練時保存的同一個 checkpoint 結果比較。
    print("\n重新載入後的 validation 結果：")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Top-1: {metrics['top1']:.2%}")
    print(f"Top-3: {metrics['top3']:.2%}")

    saved = checkpoint["validation"]

    print("\n訓練時保存在 checkpoint 的結果：")
    print(f"Loss: {saved['loss']:.4f}")
    print(f"Top-1: {saved['top1']:.2%}")
    print(f"Top-3: {saved['top3']:.2%}")

    print("\n輸出目錄：", output_dir)


if __name__ == "__main__":
    main()