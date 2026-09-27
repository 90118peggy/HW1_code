"""E1：Frozen MERT + 時間平均 + 線性分類器。"""

import torch
from torch import nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor


MODEL_NAME = "m-a-p/MERT-v1-95M"
MODEL_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"


class FrozenMERTClassifier(nn.Module):
    def __init__(
        self,
        num_classes=6,
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
    ):
        super().__init__()

        self.model_name = model_name
        self.revision = revision

        # 載入官方波形處理器，包含輸入正規化設定。
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            revision=revision,
        )

        # 載入預訓練 MERT。
        # 此模型使用官方 repository 內的自訂模型程式碼。
        self.encoder = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )

        if self.processor.sampling_rate != 24000:
            raise ValueError("本次資料流程要求模型使用 24000 Hz")

        self.sample_rate = self.processor.sampling_rate
        self.feature_dim = self.encoder.config.hidden_size

        # 凍結整個 MERT，不計算其參數梯度。
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        # 唯一需要學習的部分。
        self.classifier = nn.Linear(
            self.feature_dim,
            num_classes,
        )

    def train(self, mode=True):
        """
        外部呼叫 model.train() 時：
        分類器進入 train 模式，但 MERT 仍保持 eval 模式。
        """
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def extract_features(self, waveforms):
        """
        輸入：等長的 24 kHz 單聲道波形 [B, S]
        輸出：每段音訊的特徵 [B, 768]
        """
        if waveforms.ndim != 2:
            raise ValueError(
                "extract_features 需要 [B, S]，"
                f"目前收到 {tuple(waveforms.shape)}"
            )

        if waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
            raise ValueError("輸入音訊不可為空")

        if not torch.isfinite(waveforms).all():
            raise ValueError("輸入音訊包含 NaN 或無限值")

        # 官方 processor 使用 CPU 上的 NumPy 波形。
        cpu_waveforms = waveforms.detach().cpu().float()
        audio_list = [wave.numpy() for wave in cpu_waveforms]

        inputs = self.processor(
            audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
            return_attention_mask=True,
        )

        # 正規化完成後，才搬到 MERT 所在的裝置。
        device = next(self.encoder.parameters()).device
        inputs = {
            name: value.to(device)
            for name, value in inputs.items()
        }

        outputs = self.encoder(
            **inputs,
            output_hidden_states=False,
            return_dict=True,
        )

        # [B, T, 768] → [B, 768]
        features = outputs.last_hidden_state.mean(dim=1)

        return features

    def forward(self, waveforms):
        features = self.extract_features(waveforms)

        # 這一行必須放在 no_grad 範圍之外，
        # 才能計算分類器參數的梯度。
        logits = self.classifier(features)

        return logits