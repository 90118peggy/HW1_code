"""檢查 E2 特徵形狀、最後一層一致性與分類器更新。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.mert_classifier import FrozenMERTClassifier
from models.mert_layer_probes import MERTLayerProbes


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

    model = MERTLayerProbes(num_classes=6).to(device)
    model.train()

    assert model.num_layers == 12
    assert model.feature_dim == 768
    assert not model.encoder.training
    assert all(head.training for head in model.classifiers)
    assert all(
        not parameter.requires_grad
        for parameter in model.encoder.parameters()
    )

    # 確認各分類器沒有共用同一塊權重記憶體。
    weight_addresses = {
        head.weight.data_ptr()
        for head in model.classifiers
    }
    assert len(weight_addresses) == 12

    # 確認所有可訓練參數都屬於分類器。
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    assert all(
        name.startswith("classifiers.")
        for name in trainable_names
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert trainable_count == 12 * 4614

    features = model.extract_features(waveforms)

    assert features.shape == (2, 12, 768)
    assert not features.requires_grad
    assert torch.isfinite(features).all()

    # 明確呼叫 E1 的特徵提取方法，
    # 使用同一個 encoder 與同一批波形作比較。
    e1_features = FrozenMERTClassifier.extract_features(
        model,
        waveforms,
    )

    torch.testing.assert_close(
        features[:, -1, :],
        e1_features,
        rtol=1e-5,
        atol=1e-6,
    )

    print("Device:", device)
    print("Waveforms shape:", tuple(waveforms.shape))
    print("Layer features shape:", tuple(features.shape))
    print("Encoder training mode:", model.encoder.training)
    print("Trainable parameter count:", trainable_count)
    print("第 12 層特徵與 E1 一致")

    optimizer = torch.optim.Adam(
        model.classifiers.parameters(),
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    before = [
        head.weight.detach().clone()
        for head in model.classifiers
    ]

    optimizer.zero_grad(set_to_none=True)
    logits = model(waveforms)

    assert logits.shape == (2, 12, 6)
    assert torch.isfinite(logits).all()

    # 每層各自計算一個 batch 平均 loss。
    layer_losses = torch.stack(
        [
            criterion(logits[:, index, :], targets)
            for index in range(model.num_layers)
        ]
    )

    # 相加後反向傳播，每個 head 接收自己的 loss 梯度。
    loss = layer_losses.sum()
    assert torch.isfinite(loss)

    loss.backward()

    assert all(
        parameter.grad is None
        for parameter in model.encoder.parameters()
    )

    for head in model.classifiers:
        gradient = head.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0

    optimizer.step()

    print("Logits shape:", tuple(logits.shape))

    for index, head in enumerate(model.classifiers):
        change = (
            head.weight.detach() - before[index]
        ).abs().max().item()

        assert change > 0

        print(
            f"Layer {index + 1:02d} | "
            f"loss={layer_losses[index].item():.4f} | "
            f"max weight change={change:.6f}"
        )

    print("十二個分類器彼此獨立，且全部成功更新")
    print("MERT 保持凍結，沒有參數梯度")
    print("E2 模型基本檢查通過")


if __name__ == "__main__":
    main()