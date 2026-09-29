"""E2：每層各自合併同一首歌的多段 logits。"""

import torch


@torch.no_grad()
def mert_layer_recording_logits(
    model,
    waveforms,
    chunk_batch_size=2,
):
    """
    輸入：[B, K, S]
    輸出：[B, L, C]

    B：歌曲數
    K：音訊片段數
    S：每段取樣點數
    L：MERT 層數
    C：類別數
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

    num_layers = model.num_layers
    num_classes = model.classifiers[0].out_features

    # [B, K, S] → [B*K, S]
    flat_waveforms = waveforms.reshape(
        batch_size * num_chunks,
        num_samples,
    )

    outputs = []

    for start in range(0, len(flat_waveforms), chunk_batch_size):
        chunk_batch = flat_waveforms[
            start:start + chunk_batch_size
        ]

        # 每段都得到所有層的預測。
        logits = model(chunk_batch)

        expected_shape = (
            len(chunk_batch),
            num_layers,
            num_classes,
        )

        if tuple(logits.shape) != expected_shape:
            raise ValueError(
                f"模型輸出應為 {expected_shape}，"
                f"目前收到 {tuple(logits.shape)}"
            )

        if not torch.isfinite(logits).all():
            raise ValueError("模型輸出包含 NaN 或無限值")

        outputs.append(logits)

    # [B*K, L, C]
    all_logits = torch.cat(outputs, dim=0)

    # [B*K, L, C] → [B, K, L, C]
    per_recording = all_logits.reshape(
        batch_size,
        num_chunks,
        num_layers,
        num_classes,
    )

    # 只平均片段，保留每層各自的預測。
    # [B, K, L, C] → [B, L, C]
    return per_recording.mean(dim=1)