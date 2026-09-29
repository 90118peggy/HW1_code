"""E4：使用指定層特徵，微調該層與它前面的少量區塊。"""

import torch
from torch import nn

from models.mert_classifier import (
    FrozenMERTClassifier,
    MODEL_NAME,
    MODEL_REVISION,
)


class MERTPartialFinetune(FrozenMERTClassifier):
    def __init__(
        self,
        feature_layer,
        unfreeze_last_n=2,
        num_classes=6,
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
    ):
        # 沿用現有的官方 processor 與 MERT 載入方式。
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            revision=revision,
        )

        blocks = self.encoder.encoder.layers
        self.num_layers = len(blocks)

        if not 1 <= feature_layer <= self.num_layers:
            raise ValueError(
                f"feature_layer 必須介於 1～{self.num_layers}"
            )

        if not 1 <= unfreeze_last_n <= feature_layer:
            raise ValueError(
                "unfreeze_last_n 必須介於 1～feature_layer"
            )

        self.feature_layer = feature_layer
        self.unfreeze_last_n = unfreeze_last_n

        # 對外使用從 1 開始的層編號。
        self.unfrozen_layers = list(
            range(
                feature_layer - unfreeze_last_n + 1,
                feature_layer + 1,
            )
        )

        # 先固定所有 MERT 參數，再解凍指定區塊。
        self.encoder.requires_grad_(False)

        for layer in self.unfrozen_layers:
            blocks[layer - 1].requires_grad_(True)

        self.classifier.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        # 第一版控制隨機性：分類器跟隨外部模式，
        # MERT 維持 eval；這不會關閉 autograd。
        nn.Module.train(self, mode)
        self.encoder.eval()
        return self

    def extract_features(self, waveforms):
        # 注意：這裡不能加 @torch.no_grad()。
        if waveforms.ndim != 2:
            raise ValueError(
                f"輸入應為 [B, S]，收到 {tuple(waveforms.shape)}"
            )

        if waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
            raise ValueError("輸入音訊不可為空")

        if not torch.isfinite(waveforms).all():
            raise ValueError("輸入音訊含有 NaN 或 Inf")

        # 波形不需要梯度；需要梯度的是解凍的模型參數。
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

        # 保留完整 encoder，沿用 E2 的 hidden_states 定義。
        outputs = self.encoder(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        if (
            hidden_states is None
            or len(hidden_states) != self.num_layers + 1
        ):
            raise RuntimeError("MERT hidden_states 數量不符合預期")

        # hidden_states[0] 是第一個 Transformer 區塊前的表示。
        # hidden_states[k] 對應 E2 使用的第 k 層表示。
        selected = hidden_states[self.feature_layer]

        # [B, T, D] → [B, D]
        return selected.mean(dim=1)

    def forward(self, waveforms):
        features = self.extract_features(waveforms)
        return self.classifier(features)