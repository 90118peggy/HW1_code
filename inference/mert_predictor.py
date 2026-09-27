"""MERT 的錄音層級預測：合併同一首歌的多段 logits。"""

import torch


@torch.no_grad()
def mert_recording_logits(model, waveforms, chunk_batch_size=2):
    """
    輸入：
        waveforms: [B, K, S]
        chunk_batch_size: 每次送進 MERT 的音訊片段數

    輸出：
        recording_logits: [B, 6]
    """
    if model.training:
        raise ValueError("評估前請先呼叫 model.eval()")

    if waveforms.ndim != 3:
        raise ValueError(
            "輸入必須是 [B, K, S]，"
            f"目前收到 {tuple(waveforms.shape)}"
        )

    if not isinstance(chunk_batch_size, int) or chunk_batch_size < 1:
        raise ValueError("chunk_batch_size 必須是正整數")

    batch_size, num_chunks, num_samples = waveforms.shape

    if min(batch_size, num_chunks, num_samples) < 1:
        raise ValueError("歌曲數、片段數與取樣點數都必須大於零")

    # [B, K, S] → [B*K, S]
    # 順序：歌曲 1 的所有片段、歌曲 2 的所有片段……
    flat_waveforms = waveforms.reshape(
        batch_size * num_chunks,
        num_samples,
    )

    chunk_outputs = []

    # 分小批次處理，避免一次把全部片段送進 GPU。
    for start in range(0, len(flat_waveforms), chunk_batch_size):
        chunk_batch = flat_waveforms[
            start:start + chunk_batch_size
        ]

        logits = model(chunk_batch)

        expected_shape = (len(chunk_batch), 6)
        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"模型輸出應為 {expected_shape}，"
                f"目前收到 {tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise ValueError("模型輸出包含 NaN 或無限值")

        chunk_outputs.append(logits)

    # [B*K, 6]
    all_logits = torch.cat(chunk_outputs, dim=0)

    # [B*K, 6] → [B, K, 6]
    per_recording = all_logits.reshape(
        batch_size,
        num_chunks,
        6,
    )

    # 對片段維度取平均：[B, K, 6] → [B, 6]
    return per_recording.mean(dim=1)