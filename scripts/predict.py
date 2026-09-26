import argparse
import json
from pathlib import Path

import torch

from data_pipeline.audio_common import (
    NormalizationStats,
    read_records,
)
from inference.predictor import RecordingPredictor
from models.short_chunk_cnn import ShortChunkCNN
from data_pipeline.inspect_dataset import LABELS


@torch.inference_mode()
def predict_dataset(checkpoint_path, dataset_dir, device):
    """載入一個模型，預測對應資料集的全部 test WAV。"""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    stats = NormalizationStats(
        **checkpoint["normalization_stats"]
    )

    dataset_name = dataset_dir.name

    if checkpoint["dataset"] != dataset_name:
        raise ValueError(
            f"{dataset_name} 使用了其他資料集的 checkpoint"
        )

    if stats.dataset != dataset_name:
        raise ValueError("標準化統計與資料集不一致")

    # 先建立架構，再載入最佳權重。
    model = ShortChunkCNN(
        **checkpoint["model_config"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    # 沿用既有的完整 WAV 推論流程。
    predictor = RecordingPredictor(
        chunk_model=model,
        stats=stats,
        device=device,
    )

    # 只挑選官方 test，不讀取訓練資料來擬合統計。
    records = read_records(dataset_dir, "test")

    predictions = {}
    valid_labels = set(stats.class_names)

    print(
        f"\n{dataset_name} | "
        f"checkpoint epoch={checkpoint['epoch']} | "
        f"test samples={len(records)}",
        flush=True,
    )

    for index, row in enumerate(records, start=1):
        sample_id = row["sample_id"]
        audio_path = dataset_dir / row["audio_path"]

        result = predictor.predict_wav(audio_path)
        labels = result["top3_labels"]

        # 檢查每筆恰好有三個合法且不重複的標籤。
        if len(labels) != 3 or len(set(labels)) != 3:
            raise ValueError(f"{sample_id}: Top-3 標籤格式錯誤")

        if not set(labels).issubset(valid_labels):
            raise ValueError(f"{sample_id}: 出現不合法標籤")

        if sample_id in predictions:
            raise ValueError(f"重複的 sample_id: {sample_id}")

        predictions[sample_id] = labels

        if index == 1 or index % 20 == 0 or index == len(records):
            print(
                f"  Predicted {index}/{len(records)}",
                flush=True,
            )

    # 確認沒有漏掉 test，也沒有混入其他 split。
    expected_ids = {row["sample_id"] for row in records}

    if set(predictions) != expected_ids:
        raise ValueError(f"{dataset_name}: 預測 ID 與 test 清單不一致")

    print(f"{dataset_name}: ID 與 Top-3 格式檢查通過")

    return predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-a", type=Path, required=True
    )
    parser.add_argument(
        "--checkpoint-b", type=Path, required=True
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output", type=Path, required=True
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    torch.set_num_threads(2)
    device = torch.device("cuda")

    data_root = args.data_root.resolve()
    output = args.output.resolve()

    if output.suffix.lower() != ".json":
        parser.error("輸出檔案必須使用 .json 副檔名")

    if output.exists():
        parser.error("輸出檔案已存在，請換一個新檔名")

    for name in ("dataset_A", "dataset_B"):
        official_dir = (data_root / name).resolve()
        if output == official_dir or official_dir in output.parents:
            parser.error("預測檔不可寫入官方資料夾")

    submission = {
        "dataset_A": predict_dataset(
            args.checkpoint_a,
            data_root / "dataset_A",
            device,
        ),
        "dataset_B": predict_dataset(
            args.checkpoint_b,
            data_root / "dataset_B",
            device,
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as handle:
        json.dump(
            submission,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    print("\n已完成 test 預測：")
    print("Dataset A:", len(submission["dataset_A"]))
    print("Dataset B:", len(submission["dataset_B"]))
    print("JSON:", output)


if __name__ == "__main__":
    main()