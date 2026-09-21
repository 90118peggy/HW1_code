from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import NormalizationStats
from data_pipeline.audio_pipeline import HW1AudioDataset
from models.short_chunk_cnn import ShortChunkCNN


@torch.no_grad()
def evaluate_fixed_batch(model, features, targets, criterion):
    """關閉 Dropout，檢查這批固定資料目前學得如何。"""
    was_training = model.training
    model.eval()

    logits = model(features)
    loss = criterion(logits, targets).item()

    predictions = logits.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()

    model.train(was_training)

    return loss, accuracy


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]

    stats = NormalizationStats.load(
        root / "audio_stats" / "dataset_A.json"
    )

    dataset = HW1AudioDataset(
        dataset_dir=root / "dataset_A",
        split="train",
        stats=stats,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    # 只在這裡讀取一次，之後反覆使用同一批頻譜。
    batch = next(iter(loader))
    features = batch["features"].to(device)
    targets = batch["target"].to(device)

    print("Sample IDs:", batch["sample_id"])
    print("Targets:", targets.tolist())
    print("Features shape:", tuple(features.shape))
    print(
        "Class counts:",
        torch.bincount(targets, minlength=6).tolist(),
    )

    model = ShortChunkCNN(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    initial_loss, initial_accuracy = evaluate_fixed_batch(
        model, features, targets, criterion
    )

    print(
        f"Before training | "
        f"loss={initial_loss:.4f} | "
        f"accuracy={initial_accuracy:.1%}"
    )

    model.train()
    passed = False
    max_steps = 200

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = model(features)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Step {step}: loss 出現 NaN 或無限值")

        loss.backward()
        optimizer.step()

        if step == 1 or step % 20 == 0:
            eval_loss, eval_accuracy = evaluate_fixed_batch(
                model, features, targets, criterion
            )

            print(
                f"Step {step:03d} | "
                f"train loss={loss.item():.4f} | "
                f"fixed-batch eval loss={eval_loss:.4f} | "
                f"accuracy={eval_accuracy:.1%}"
            )

            if eval_accuracy == 1.0 and eval_loss < 0.1:
                passed = True
                print("固定八筆資料已全部答對，且 loss < 0.1")
                break

    if passed:
        print("小批次過擬合檢查通過")
    else:
        print("尚未達到檢查目標，請保留輸出，先檢查學習趨勢")

    print("這是 train 固定片段的檢查結果，不代表 validation 表現")


if __name__ == "__main__":
    main()