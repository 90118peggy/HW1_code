"""E2：凍結 MERT，對第 1～12 層建立獨立線性分類器。"""

from copy import deepcopy

import torch
from torch import nn

from models.mert_classifier import (
    FrozenMERTClassifier,
    MODEL_NAME,
    MODEL_REVISION,
)


class MERTLayerProbes(FrozenMERTClassifier):
    def __init__(
        self,
        num_classes=6,
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
    ):
        # 沿用 E1 的官方 processor、MERT 載入與凍結設定。
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            revision=revision,
        )

        self.num_layers = self.encoder.config.num_hidden_layers

        # 每層各自持有獨立的分類器。
        # 使用相同初始數值，減少分類器初始化造成的比較差異。
        self.classifiers = nn.ModuleList(
            [
                deepcopy(self.classifier)
                for _ in range(self.num_layers)
            ]
        )

        # 原本 E1 的單一分類器已由上面的多個分類器取代。
        del self.classifier

    @torch.no_grad()
    def extract_features(self, waveforms):
        """
        輸入：[B, S]
        輸出：[B, L, D]

        L：Transformer 層數，本模型為 12。
        D：特徵維度，本模型為 768。
        """
        if waveforms.ndim != 2:
            raise ValueError(
                "輸入必須是 [B, S]，"
                f"目前收到 {tuple(waveforms.shape)}"
            )

        if waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
            raise ValueError("輸入音訊不可為空")

        if not torch.isfinite(waveforms).all():
            raise ValueError("輸入音訊包含 NaN 或無限值")

        # 和 E1 使用相同的波形前處理。
        cpu_waveforms = waveforms.detach().cpu().float()
        audio_list = [wave.numpy() for wave in cpu_waveforms]

        inputs = self.processor(
            audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
            return_attention_mask=True,
        )

        device = next(self.encoder.parameters()).device
        inputs = {
            name: value.to(device)
            for name, value in inputs.items()
        }

        # E2 需要取得所有層的表示。
        outputs = self.encoder(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        if hidden_states is None:
            raise RuntimeError("模型沒有回傳 hidden_states")

        if len(hidden_states) != self.num_layers + 1:
            raise RuntimeError(
                f"預期 {self.num_layers + 1} 組 hidden states，"
                f"實際收到 {len(hidden_states)} 組"
            )

        # 第 0 組是進入第一個 Transformer block 前的表示。
        # 第 1～12 組才是各個 Transformer block 的輸出。
        layer_states = hidden_states[1:]

        # 每層：[B, T, D] → [B, D]
        pooled_layers = [
            state.mean(dim=1)
            for state in layer_states
        ]

        # 12 個 [B, D] → [B, 12, D]
        features = torch.stack(pooled_layers, dim=1)

        return features

    def forward(self, waveforms):
        features = self.extract_features(waveforms)

        # 第 i 個分類器只使用第 i 層的特徵。
        layer_logits = [
            classifier(features[:, index, :])
            for index, classifier in enumerate(self.classifiers)
        ]

        # 12 個 [B, 6] → [B, 12, 6]
        return torch.stack(layer_logits, dim=1)