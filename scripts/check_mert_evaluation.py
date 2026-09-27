"""檢查 MERT 九段合併與不同片段批次大小的一致性。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline.waveform_dataset import HW1WaveformDataset
from inference.mert_predictor import mert_recording_logits
from models.mert_classifier import FrozenMERTClassifier


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU")

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda")

    dataset = HW1WaveformDataset(
        dataset_dir=root / "dataset_A",
        split="validation",
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

    model = FrozenMERTClassifier(num_classes=6).to(device)
    model.eval()

    # 一次處理兩個片段。
    logits = mert_recording_logits(
        model,
        waveforms,
        chunk_batch_size=2,
    )

    assert logits.shape == (2, 6)
    assert not logits.requires_grad

    # 改成一次處理一個片段，結果應在浮點誤差內一致。
    logits_one_at_a_time = mert_recording_logits(
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

    # 單獨預測第一首歌，確認合併時沒有混入第二首歌。
    first_recording_logits = mert_recording_logits(
        model,
        waveforms[:1],
        chunk_batch_size=2,
    )

    torch.testing.assert_close(
        logits[:1],
        first_recording_logits,
        rtol=1e-4,
        atol=1e-5,
    )

    # loss 的單位是「歌曲」，不是把九段各算成一筆樣本。
    loss = nn.CrossEntropyLoss()(logits, targets)
    assert torch.isfinite(loss)

    probabilities = logits.softmax(dim=-1)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, device=device),
    )

    top3 = logits.topk(k=3, dim=-1).indices

    print("Sample IDs:", batch["sample_id"])
    print("Waveforms shape:", tuple(waveforms.shape))
    print("Recording logits shape:", tuple(logits.shape))
    print("Targets shape:", tuple(targets.shape))
    print("Top-3 shape:", tuple(top3.shape))
    print("Validation batch loss:", loss.item())
    print("不同 chunk_batch_size 的結果一致")
    print("單首與多首一起評估的結果一致")
    print("九段合併評估檢查通過")
    print("注意：分類器尚未正式訓練，此處只檢查流程")


if __name__ == "__main__":
    main()