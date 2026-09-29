"""評估 E2 的指定層 checkpoint，輸出預測與混淆矩陣。"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig
from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.mert_layer_probes import MERTLayerProbes
from scripts.evaluate_mert import evaluate, save_confusion_plot


class SelectedLayerClassifier(nn.Module):
    """把 E2 指定層包成輸出 [B, 6] 的模型。"""

    def __init__(self, probes, layer_index):
        super().__init__()
        self.probes = probes
        self.layer_index = layer_index

    def forward(self, waveforms):
        # [B, 12, 768]
        features = self.probes.extract_features(waveforms)

        # 選出指定層：[B, 768]
        selected_features = features[:, self.layer_index, :]

        # 使用該層已訓練的分類器：[B, 6]
        classifier = self.probes.classifiers[self.layer_index]
        return classifier(selected_features)


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
        or checkpoint.get("experiment") != "E2_mert_layer_probes"
    ):
        raise ValueError("這支程式只接受目前 E2 格式的 checkpoint")

    dataset_name = checkpoint["dataset"]
    if dataset_name not in ("dataset_A", "dataset_B"):
        raise ValueError("Checkpoint 資料集名稱不合法")

    class_names = checkpoint["class_names"]
    audio_config = AudioConfig(**checkpoint["audio_config"])

    dataset = HW1WaveformDataset(
        root / dataset_name,
        "validation",
        config=audio_config,
    )

    if dataset.class_names != class_names:
        raise ValueError("Checkpoint 與資料集的類別順序不一致")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    probes = MERTLayerProbes(**checkpoint["model_config"])

    # Checkpoint 使用 1～12；Python 索引使用 0～11。
    layer_number = checkpoint["layer"]
    if not isinstance(layer_number, int) or not 1 <= layer_number <= probes.num_layers:
        raise ValueError("Checkpoint 的層數不合法")

    layer_index = layer_number - 1

    probes.classifiers[layer_index].load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True,
    )

    if probes.processor.to_dict() != checkpoint["processor_config"]:
        raise ValueError("波形處理器設定與 checkpoint 不一致")

    model = SelectedLayerClassifier(
        probes,
        layer_index,
    ).to(device)
    model.eval()

    print("Dataset:", dataset_name)
    print("Selected layer:", layer_number)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation samples:", len(dataset))

    # 重用 E1 的歌曲層級評估與 CSV 資料整理。
    metrics, rows, confusion = evaluate(
        model,
        loader,
        class_names,
        device,
        args.chunk_batch_size,
    )

    actual_ids = [row["sample_id"] for row in rows]
    expected_ids = {row["sample_id"] for row in dataset.records}

    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("預測包含重複 ID")

    if set(actual_ids) != expected_ids:
        raise RuntimeError("預測 ID 與官方 validation 不一致")

    if int(confusion.sum()) != len(dataset):
        raise RuntimeError("混淆矩陣總數與 validation 樣本數不一致")

    saved = checkpoint["validation"]

    for key in ("samples", "top1", "top3"):
        if metrics[key] != saved[key]:
            raise RuntimeError(f"{key} 無法重現 checkpoint 結果")

    if not math.isclose(
        metrics["loss"],
        saved["loss"],
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError("Loss 無法重現 checkpoint 結果")

    output_dir.mkdir(parents=True)

    report = {
        "dataset": dataset_name,
        "split": "validation",
        "layer": layer_number,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint": str(args.checkpoint.resolve()),
        "class_names": class_names,
        **metrics,
        "confusion_matrix": confusion.tolist(),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
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
            f"E2 {dataset_name} validation | layer {layer_number} | "
            f"epoch {checkpoint['epoch']}"
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