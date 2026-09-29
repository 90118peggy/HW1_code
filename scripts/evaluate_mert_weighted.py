import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig
from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.mert_weighted_classifier import MERTWeightedClassifier
from scripts.evaluate_mert import evaluate, save_confusion_plot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def check_metrics(actual, expected):
    """確認重新評估的結果與訓練時儲存的結果一致。"""
    for key in ("top1", "top3", "samples"):
        if actual[key] != expected[key]:
            raise RuntimeError(
                f"{key} 不一致："
                f"重新評估={actual[key]}，checkpoint={expected[key]}"
            )

    if not math.isclose(
        actual["loss"],
        expected["loss"],
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            f"Loss 不一致："
            f"重新評估={actual['loss']}，"
            f"checkpoint={expected['loss']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--chunk-batch-size", type=int, default=2)
    args = parser.parse_args()

    if args.val_batch_size <= 0 or args.chunk_batch_size <= 0:
        raise ValueError("Batch size 必須大於 0")

    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # 防止覆寫既有報告或寫入官方資料集。
    for name in ("dataset_A", "dataset_B"):
        dataset_root = (PROJECT_ROOT / name).resolve()
        if output_dir == dataset_root or dataset_root in output_dir.parents:
            raise ValueError("輸出資料夾不能放在官方資料集內")

    if output_dir.exists():
        raise FileExistsError(
            f"輸出資料夾已存在：{output_dir}\n"
            "請指定新的 --output-dir，保留原本的報告。"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("未偵測到 CUDA，請確認已啟用 MERT 環境")

    torch.set_num_threads(2)
    device = torch.device("cuda")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("experiment") != "E3_mert_weighted"
    ):
        raise ValueError("這個檔案不是目前支援的 E3 checkpoint")

    dataset_name = checkpoint["dataset"]
    if dataset_name not in ("dataset_A", "dataset_B"):
        raise ValueError(f"不支援的資料集：{dataset_name}")

    config = AudioConfig(**checkpoint["audio_config"])
    dataset = HW1WaveformDataset(
        PROJECT_ROOT / dataset_name,
        split="validation",
        config=config,
    )

    class_names = checkpoint["class_names"]
    if dataset.class_names != class_names:
        raise RuntimeError("資料集與 checkpoint 的類別順序不一致")

    loader = DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = MERTWeightedClassifier(
        **checkpoint["model_config"]
    ).to(device)

    # 還原線性分類器。
    model.classifier.load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True,
    )

    # 還原 softmax 之前的層權重參數。
    saved_logits = checkpoint["layer_logits"]
    if saved_logits.shape != model.layer_logits.shape:
        raise RuntimeError("Checkpoint 的層權重參數形狀不一致")

    if not torch.isfinite(saved_logits).all():
        raise RuntimeError("Checkpoint 的層權重參數含有 NaN 或 Inf")

    with torch.no_grad():
        model.layer_logits.copy_(saved_logits.to(device))

    torch.testing.assert_close(
        model.layer_logits.detach().cpu(),
        saved_logits,
        rtol=0,
        atol=0,
    )

    if model.processor.to_dict() != checkpoint["processor_config"]:
        raise RuntimeError("音訊前處理設定與 checkpoint 不一致")

    model.eval()
    if model.encoder.training:
        raise RuntimeError("MERT encoder 應處於 eval 模式")

    layer_weights = model.get_layer_weights().detach().cpu()
    saved_weights = torch.tensor(
        checkpoint["layer_weights"],
        dtype=layer_weights.dtype,
    )

    torch.testing.assert_close(
        layer_weights,
        saved_weights,
        rtol=1e-6,
        atol=1e-7,
    )

    if not torch.isfinite(layer_weights).all():
        raise RuntimeError("層權重含有 NaN 或 Inf")

    if not math.isclose(
        layer_weights.sum().item(), 1.0, abs_tol=1e-6
    ):
        raise RuntimeError("層權重總和不等於 1")

    print(f"Dataset: {dataset_name}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation samples: {len(dataset)}")
    print(f"Classes: {class_names}")
    print("分類器與層權重載入檢查通過\n")

    metrics, rows, confusion = evaluate(
        model=model,
        loader=loader,
        class_names=class_names,
        device=device,
        chunk_batch_size=args.chunk_batch_size,
    )

    # 確保官方 validation 的每首歌恰好評估一次。
    expected_ids = [row["sample_id"] for row in dataset.records]
    actual_ids = [row["sample_id"] for row in rows]

    if len(actual_ids) != len(expected_ids):
        raise RuntimeError("預測筆數與 validation 筆數不一致")

    if len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError("預測結果有重複的 sample_id")

    if set(actual_ids) != set(expected_ids):
        raise RuntimeError("預測結果與官方 validation 的 ID 不一致")

    if metrics["samples"] != len(dataset):
        raise RuntimeError("Metrics 的樣本數不正確")

    if int(confusion.sum().item()) != len(dataset):
        raise RuntimeError("混淆矩陣的總數不正確")

    if not math.isclose(
        confusion.diag().sum().item() / len(dataset),
        metrics["top1"],
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("混淆矩陣與 Top-1 不一致")

    check_metrics(metrics, checkpoint["validation"])
    print("\n最佳 checkpoint 指標重現檢查通過")

    # 所有檢查通過後才輸出報告。
    output_dir.mkdir(parents=True, exist_ok=False)

    weights_report = [
        {"layer": index, "weight": float(weight)}
        for index, weight in enumerate(layer_weights.tolist(), start=1)
    ]

    report = {
        "experiment": "E3_mert_weighted",
        "dataset": dataset_name,
        "split": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "class_names": class_names,
        **metrics,
        "layer_weights": weights_report,
        "confusion_matrix": confusion.tolist(),
    }
    save_json(output_dir / "metrics.json", report)
    save_json(output_dir / "layer_weights.json", weights_report)

    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true_label/predicted_label", *class_names])
        for label, counts in zip(class_names, confusion.tolist()):
            writer.writerow([label, *counts])

    save_confusion_plot(
        confusion,
        class_names,
        (
            f"E3 {dataset_name} validation | "
            f"checkpoint epoch {checkpoint['epoch']}"
        ),
        output_dir / "confusion_matrix.png",
    )

    print(
        f"\nLoss={metrics['loss']:.4f} | "
        f"Top-1={metrics['top1']:.2%} | "
        f"Top-3={metrics['top3']:.2%}"
    )
    print(f"報告已儲存：{output_dir}")


if __name__ == "__main__":
    main()