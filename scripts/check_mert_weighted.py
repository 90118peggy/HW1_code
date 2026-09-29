"""檢查 E3 的均勻初始化、層權重梯度與分類器更新。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.mert_weighted_classifier import MERTWeightedClassifier


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]

    dataset = HW1WaveformDataset(
        root / "dataset_A",
        "train",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    waveforms = batch["waveforms"]
    targets = batch["target"].to(device)

    model = MERTWeightedClassifier(num_classes=6).to(device)
    model.train()

    assert model.num_layers == 12
    assert not model.encoder.training
    assert model.classifier.training
    assert all(
        not parameter.requires_grad
        for parameter in model.encoder.parameters()
    )

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert set(trainable) == {
        "layer_logits",
        "classifier.weight",
        "classifier.bias",
    }

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable.values()
    )
    assert trainable_count == 4626

    weights_before = model.get_layer_weights().detach().clone()

    torch.testing.assert_close(
        weights_before,
        torch.full_like(weights_before, 1.0 / 12),
    )

    features = model.extract_features(waveforms)
    combined = model.combine_features(features)

    assert features.shape == (2, 12, 768)
    assert combined.shape == (2, 768)
    assert not features.requires_grad
    assert combined.requires_grad
    assert torch.isfinite(combined).all()

    # 均勻初始化時，加權結果應等於十二層的普通平均。
    torch.testing.assert_close(
        combined.detach(),
        features.mean(dim=1),
        rtol=1e-5,
        atol=1e-6,
    )

    # 同時包含層權重與分類器參數。
    optimizer = torch.optim.Adam(
        list(trainable.values()),
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in trainable.items()
    }

    optimizer.zero_grad(set_to_none=True)
    logits = model(waveforms)
    loss = criterion(logits, targets)

    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    loss.backward()

    # MERT 不應取得參數梯度。
    assert all(
        parameter.grad is None
        for parameter in model.encoder.parameters()
    )

    # 三組可訓練參數都應有有效且非零的梯度。
    for name, parameter in trainable.items():
        gradient = parameter.grad

        assert gradient is not None, f"{name} 沒有梯度"
        assert torch.isfinite(gradient).all(), f"{name} 梯度不合法"
        assert gradient.abs().sum().item() > 0, f"{name} 梯度全為零"

        print(
            f"{name} gradient norm: "
            f"{gradient.norm().item():.6f}"
        )

    optimizer.step()

    for name, parameter in trainable.items():
        change = (
            parameter.detach() - parameters_before[name]
        ).abs().max().item()

        assert change > 0, f"{name} 沒有更新"
        print(f"{name} max change: {change:.6f}")

    weights_after = model.get_layer_weights().detach()

    assert torch.isfinite(weights_after).all()
    assert (weights_after >= 0).all()

    torch.testing.assert_close(
        weights_after.sum(),
        weights_after.new_tensor(1.0),
    )

    assert (
        weights_after - weights_before
    ).abs().max().item() > 0

    print("\nDevice:", device)
    print("Layer features shape:", tuple(features.shape))
    print("Combined features shape:", tuple(combined.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Trainable parameter count:", trainable_count)
    print("Loss:", loss.item())

    print("\n各層權重：")
    for index, (before, after) in enumerate(
        zip(weights_before.cpu().tolist(), weights_after.cpu().tolist()),
        start=1,
    ):
        print(f"Layer {index:02d}: {before:.6f} -> {after:.6f}")

    print("\n均勻初始化與特徵平均檢查通過")
    print("層權重與分類器都取得梯度並成功更新")
    print("MERT 保持凍結，沒有參數梯度")
    print("E3 模型基本檢查通過")


if __name__ == "__main__":
    main()