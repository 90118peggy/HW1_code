"""用真實音訊檢查 MERT 特徵、凍結設定與分類器更新。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.mert_classifier import FrozenMERTClassifier


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]

    dataset = HW1WaveformDataset(
        dataset_dir=root / "dataset_A",
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    # 波形暫留 CPU；模型內的 processor 處理完成後才搬到 GPU。
    waveforms = batch["waveforms"]
    targets = batch["target"].to(device)

    print("正在載入 MERT；第一次執行需要下載權重。")
    model = FrozenMERTClassifier(num_classes=6).to(device)
    model.train()

    print("Device:", device)
    print("Waveforms shape:", tuple(waveforms.shape))
    print("Feature dimension:", model.feature_dim)
    print("Encoder training mode:", model.encoder.training)
    print("Classifier training mode:", model.classifier.training)

    assert not model.encoder.training
    assert model.classifier.training
    assert all(
        not parameter.requires_grad
        for parameter in model.encoder.parameters()
    )

    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    print("Trainable parameters:", trainable)

    assert set(trainable) == {
        "classifier.weight",
        "classifier.bias",
    }

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print("Trainable parameter count:", trainable_count)
    assert trainable_count == 4614

    # 使用完全相同的已裁切波形，重複提取特徵。
    features_1 = model.extract_features(waveforms)
    features_2 = model.extract_features(waveforms)

    assert features_1.shape == (2, 768)
    assert not features_1.requires_grad
    assert torch.isfinite(features_1).all()

    torch.testing.assert_close(
        features_1,
        features_2,
        rtol=1e-5,
        atol=1e-6,
    )
    print("Features shape:", tuple(features_1.shape))
    print("固定輸入的特徵重現檢查通過")

    # Optimizer 只持有線性分類器的參數。
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    before = model.classifier.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    logits = model(waveforms)
    loss = criterion(logits, targets)

    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    loss.backward()

    assert all(
        parameter.grad is None
        for parameter in model.encoder.parameters()
    )

    gradient = model.classifier.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0

    optimizer.step()

    change = (
        model.classifier.weight.detach() - before
    ).abs().max().item()
    assert change > 0

    print("Logits shape:", tuple(logits.shape))
    print("Loss:", loss.item())
    print("Classifier gradient norm:", gradient.norm().item())
    print("Classifier max weight change:", change)
    print("MERT 無參數梯度，分類器成功更新")
    print("E1 模型基本檢查通過")


if __name__ == "__main__":
    main()