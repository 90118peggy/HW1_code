"""E3：凍結 MERT，學習層權重與單一線性分類器。"""

import torch
from torch import nn

from models.mert_classifier import MODEL_NAME, MODEL_REVISION
from models.mert_layer_probes import MERTLayerProbes


class MERTWeightedClassifier(MERTLayerProbes):
    def __init__(
        self,
        num_classes=6,
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
    ):
        # 重用 E2 的 MERT 載入、凍結與多層特徵提取。
        super().__init__(
            num_classes=num_classes,
            model_name=model_name,
            revision=revision,
        )

        # 保留一個剛初始化的 Linear，作為 E3 的分類器。
        # 這裡沒有載入任何 E2 訓練好的分類器權重。
        self.classifier = self.classifiers[0]

        # E3 只需要一個分類器。
        del self.classifiers

        # 十二個可學習的原始分數。
        # 全部初始化為 0，經 softmax 後就是每層 1/12。
        self.layer_logits = nn.Parameter(
            torch.zeros(self.num_layers)
        )

    def get_layer_weights(self):
        """將原始分數轉成非負、總和為 1 的層權重。"""
        return torch.softmax(self.layer_logits, dim=0)

    def combine_features(self, features):
        """
        輸入：[B, L, D]
        輸出：[B, D]
        """
        if features.ndim != 3:
            raise ValueError("多層特徵必須是 [B, L, D]")

        if tuple(features.shape[1:]) != (
            self.num_layers,
            self.feature_dim,
        ):
            raise ValueError(
                f"預期每筆特徵為 "
                f"[{self.num_layers}, {self.feature_dim}]，"
                f"目前收到 {tuple(features.shape[1:])}"
            )

        # [L] → [1, L, 1]
        weights = self.get_layer_weights().view(
            1,
            self.num_layers,
            1,
        )

        # 每層特徵乘上自己的權重，再沿層維度相加。
        # [B, L, D] → [B, D]
        return (features * weights).sum(dim=1)

    def forward(self, waveforms):
        # 繼承 E2 的方法，這部分在 no_grad 下執行。
        features = self.extract_features(waveforms)

        # 以下兩部分必須保留梯度。
        combined = self.combine_features(features)
        logits = self.classifier(combined)

        return logits