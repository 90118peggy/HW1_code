
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import NormalizationStats
from data_pipeline.audio_pipeline import HW1AudioDataset
from models.short_chunk_cnn import ShortChunkCNN


def main():
    # 1. 固定本次檢查的隨機種子，方便重現。
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU，請確認目前在遠端 GPU 環境")

    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]

    # 2. 載入 Dataset A 的既有 train 統計。
    stats = NormalizationStats.load(
        root / "audio_stats" / "dataset_A.json"
    )

    # 3. 建立 train Dataset 與 DataLoader。
    dataset = HW1AudioDataset(
        dataset_dir=root / "dataset_A",
        split="train",
        stats=stats,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    # 4. 只取一個 batch，並把資料搬到 GPU。
    batch = next(iter(loader))
    features = batch["features"].to(device)
    targets = batch["target"].to(device)

    print("Sample IDs:", batch["sample_id"])
    print("Features shape:", tuple(features.shape))
    print("Targets:", targets.tolist())
    print("Target dtype:", targets.dtype)

    # 5. 建立模型、損失函數與 optimizer。
    model = ShortChunkCNN(num_classes=6).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # 留下第一層卷積權重的副本，稍後檢查是否真的更新。
    watched_weight = model.features[0].conv.weight
    weight_before = watched_weight.detach().clone()

    # 6. 前向傳播：頻譜 → logits → loss。
    optimizer.zero_grad(set_to_none=True)

    logits = model(features)
    loss = criterion(logits, targets)

    assert tuple(logits.shape) == (4, 6)
    assert torch.isfinite(logits).all().item()
    assert torch.isfinite(loss).item()

    print("Logits shape:", tuple(logits.shape))
    print("Loss:", loss.item())

    # 7. 反向傳播：計算每個參數的梯度。
    loss.backward()

    gradient = watched_weight.grad

    assert gradient is not None, "第一層卷積沒有收到梯度"
    assert torch.isfinite(gradient).all().item(), "梯度含 NaN 或無限值"

    gradient_norm = gradient.norm().item()
    assert gradient_norm > 0, "第一層卷積梯度全部為零"

    print("First conv gradient norm:", gradient_norm)

    # 8. 更新模型參數。
    optimizer.step()

    weight_change = (
        watched_weight.detach() - weight_before
    ).abs().max().item()

    print("First conv max weight change:", weight_change)

    assert weight_change > 0, "optimizer 執行後權重沒有改變"

    print("真實資料的前向、反向傳播與權重更新檢查通過")


if __name__ == "__main__":
    main()