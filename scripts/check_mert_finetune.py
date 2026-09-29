import argparse
import gc
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig
from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.mert_layer_probes import MERTLayerProbes
from models.mert_partial_finetune import MERTPartialFinetune


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def state_digest(module):
    """比較兩次載入的 encoder 是否完全一致。"""
    digest = hashlib.sha256()

    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())

    return digest.hexdigest()


def check_group_gradients(label, module):
    parameters = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    if not parameters:
        raise RuntimeError(f"{label} 沒有可訓練參數")

    squared_norm = 0.0

    for parameter in parameters:
        gradient = parameter.grad

        if gradient is None:
            raise RuntimeError(f"{label} 有參數沒有收到梯度")

        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"{label} 梯度含有 NaN 或 Inf")

        squared_norm += gradient.detach().double().square().sum().item()

    norm = squared_norm ** 0.5

    if norm == 0:
        raise RuntimeError(f"{label} 整組梯度都是 0")

    print(f"{label} gradient norm: {norm:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["A", "B"], required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("未偵測到 CUDA")

    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device("cuda")

    selected_layer = {"A": 6, "B": 5}[args.dataset]
    dataset_name = f"dataset_{args.dataset}"

    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / f"{args.dataset}_E2_seed42"
        / f"layer_{selected_layer:02d}_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("experiment") != "E2_mert_layer_probes"
        or checkpoint["dataset"] != dataset_name
        or checkpoint["layer"] != selected_layer
    ):
        raise RuntimeError("E2 checkpoint 的格式、資料集或層數不一致")

    dataset = HW1WaveformDataset(
        PROJECT_ROOT / dataset_name,
        split="train",
        config=AudioConfig(**checkpoint["audio_config"]),
    )

    if dataset.class_names != checkpoint["class_names"]:
        raise RuntimeError("資料集與 checkpoint 的類別順序不一致")

    batch = next(iter(DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )))

    # 固定這次抽到的音訊，供 E2、E4 使用完全相同的輸入。
    waveforms = batch["waveforms"]
    targets = batch["target"].to(device)

    print("Dataset:", dataset_name)
    print("Device:", device)
    print("Waveforms shape:", tuple(waveforms.shape))

    # 先取得原本 E2 的特徵與預測。
    reference = MERTLayerProbes(
        **checkpoint["model_config"]
    ).to(device)

    reference.classifiers[selected_layer - 1].load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True,
    )
    reference.eval()

    if reference.processor.to_dict() != checkpoint["processor_config"]:
        raise RuntimeError("E2 processor 設定與 checkpoint 不一致")

    reference_encoder_digest = state_digest(reference.encoder)

    with torch.no_grad():
        all_features = reference.extract_features(waveforms)
        reference_features = all_features[:, selected_layer - 1, :]
        reference_logits = reference.classifiers[
            selected_layer - 1
        ](reference_features)

        reference_features = reference_features.cpu()
        reference_logits = reference_logits.cpu()

    # 避免 E2、E4 同時占用 GPU。
    del all_features, reference
    gc.collect()
    torch.cuda.empty_cache()

    model = MERTPartialFinetune(
        feature_layer=selected_layer,
        unfreeze_last_n=2,
        **checkpoint["model_config"],
    ).to(device)

    model.classifier.load_state_dict(
        checkpoint["classifier_state_dict"],
        strict=True,
    )

    if model.processor.to_dict() != checkpoint["processor_config"]:
        raise RuntimeError("E4 processor 設定與 checkpoint 不一致")

    if state_digest(model.encoder) != reference_encoder_digest:
        raise RuntimeError(
            "E2 與 E4 載入的 encoder 權重不一致，"
            "請先檢查 MERT 權重載入流程"
        )

    # 確認只開放指定區塊與分類器更新。
    allowed_prefixes = ["classifier."]
    allowed_prefixes += [
        f"encoder.encoder.layers.{layer - 1}."
        for layer in model.unfrozen_layers
    ]

    for name, parameter in model.named_parameters():
        expected = any(
            name.startswith(prefix)
            for prefix in allowed_prefixes
        )
        if parameter.requires_grad != expected:
            raise RuntimeError(f"參數解凍設定錯誤：{name}")

    model.eval()
    with torch.no_grad():
        initial_features = model.extract_features(waveforms)
        initial_logits = model(waveforms)

    torch.testing.assert_close(
        initial_features.cpu(),
        reference_features,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        initial_logits.cpu(),
        reference_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    print("微調前，E4 與 E2 的特徵及 logits 一致")

    del initial_features, initial_logits

    # 將所有參數與 buffer 複製到 CPU，供更新後逐一檢查。
    before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    backbone_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    classifier_parameters = list(model.classifier.parameters())

    optimizer = torch.optim.Adam(
        [
            {"params": backbone_parameters, "lr": 1e-5},
            {"params": classifier_parameters, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )

    model.train()
    if not model.training or not model.classifier.training:
        raise RuntimeError("E4 或分類器沒有進入 train 模式")

    if any(module.training for module in model.encoder.modules()):
        raise RuntimeError("這一版的 MERT 應維持 eval 模式")

    print("Feature layer:", model.feature_layer)
    print("Unfrozen layers:", model.unfrozen_layers)
    print("Encoder training mode:", model.encoder.training)
    print(
        "Trainable parameter count:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    features = model.extract_features(waveforms)

    if not features.requires_grad:
        raise RuntimeError(
            "特徵沒有梯度！請檢查是否仍被 no_grad 或 detach 阻斷"
        )

    logits = model.classifier(features)
    loss = nn.CrossEntropyLoss()(logits, targets)

    if not torch.isfinite(loss):
        raise RuntimeError("Loss 不是有限數值")

    loss.backward()

    for layer in model.unfrozen_layers:
        check_group_gradients(
            f"Layer {layer:02d}",
            model.encoder.encoder.layers[layer - 1],
        )

    check_group_gradients("Classifier", model.classifier)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f"固定參數不應有梯度：{name}")

    torch.nn.utils.clip_grad_norm_(
        backbone_parameters + classifier_parameters,
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()

    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    changes = {}
    for name, value in model.state_dict().items():
        current = value.detach().cpu()

        if not torch.isfinite(current).all():
            raise RuntimeError(f"更新後出現非有限數值：{name}")

        if name not in trainable_names:
            if not torch.equal(current, before[name]):
                raise RuntimeError(f"固定參數或 buffer 發生改變：{name}")
        else:
            changes[name] = (
                current - before[name]
            ).abs().max().item()

    groups = [
        (
            f"Layer {layer:02d}",
            f"encoder.encoder.layers.{layer - 1}.",
        )
        for layer in model.unfrozen_layers
    ]
    groups.append(("Classifier", "classifier."))

    for label, prefix in groups:
        maximum_change = max(
            value
            for name, value in changes.items()
            if name.startswith(prefix)
        )

        if maximum_change == 0:
            raise RuntimeError(f"{label} 更新後完全沒有改變")

        print(f"{label} max weight change: {maximum_change:.8f}")

    print("Features shape:", tuple(features.shape))
    print("Logits shape:", tuple(logits.shape))
    print(f"Loss: {loss.item():.6f}")
    print(
        "Peak allocated GPU memory:",
        f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
    )
    print("指定區塊與分類器成功更新")
    print("其他參數與 buffers 完全不變")
    print("E4 單步梯度與更新檢查通過")
    print("本次沒有儲存或覆寫任何 checkpoint")


if __name__ == "__main__":
    main()