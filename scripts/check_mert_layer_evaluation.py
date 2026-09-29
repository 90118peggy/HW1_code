"""檢查 E2 多段評估的形狀、分批一致性與歌曲分組。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.waveform_dataset import HW1WaveformDataset
from inference.mert_layer_predictor import mert_layer_recording_logits
from models.mert_layer_probes import MERTLayerProbes


@torch.no_grad()
def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda")

    dataset = HW1WaveformDataset(
        root / "dataset_A",
        "validation",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    waveforms = batch["waveforms"]
    targets = batch["target"].to(device)

    model = MERTLayerProbes(num_classes=6).to(device)
    model.eval()

    # 每次處理兩個片段。
    logits = mert_layer_recording_logits(
        model,
        waveforms,
        chunk_batch_size=2,
    )

    assert logits.shape == (2, 12, 6)
    assert not logits.requires_grad
    assert torch.isfinite(logits).all()

    # 改成每次一個片段，結果應在浮點誤差內一致。
    logits_one_at_a_time = mert_layer_recording_logits(
        model,
        waveforms,
        chunk_batch_size=1,
    )

    torch.testing.assert_close(
        logits,
        logits_one_at_a_time,
        rtol=1e-4,
        atol=1e-5,
    )

    # 單獨評估第一首歌，確認沒有混入其他歌曲的片段。
    first_recording = mert_layer_recording_logits(
        model,
        waveforms[:1],
        chunk_batch_size=2,
    )

    torch.testing.assert_close(
        logits[:1],
        first_recording,
        rtol=1e-4,
        atol=1e-5,
    )

    # 若每首歌只提供一段，合併結果應等於直接預測該段。
    single_chunk_logits = mert_layer_recording_logits(
        model,
        waveforms[:, :1, :],
        chunk_batch_size=2,
    )
    direct_logits = model(waveforms[:, 0, :])

    torch.testing.assert_close(
        single_chunk_logits,
        direct_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    # 每一層都有自己的歌曲層級 loss。
    criterion = nn.CrossEntropyLoss()
    layer_losses = torch.stack(
        [
            criterion(logits[:, index, :], targets)
            for index in range(model.num_layers)
        ]
    )

    assert layer_losses.shape == (12,)
    assert torch.isfinite(layer_losses).all()

    # 最後一個維度才是類別。
    top3 = logits.topk(k=3, dim=-1).indices
    assert top3.shape == (2, 12, 3)

    print("Sample IDs:", batch["sample_id"])
    print("Waveforms shape:", tuple(waveforms.shape))
    print("Recording logits shape:", tuple(logits.shape))
    print("Targets shape:", tuple(targets.shape))
    print("Per-layer loss shape:", tuple(layer_losses.shape))
    print("Top-3 shape:", tuple(top3.shape))
    print("不同片段批次大小的結果一致")
    print("單首與多首一起評估的結果一致")
    print("單片段合併與直接預測一致")
    print("E2 九段評估檢查通過")
    print("分類器尚未正式訓練，此處只確認評估流程")


if __name__ == "__main__":
    main()