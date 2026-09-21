"""完整 WAV 推論與同一首錄音的多段分數合併。"""

import torch

from data_pipeline.audio_pipeline import AudioFrontend, crop_waveform, load_waveform


def recording_logits(chunk_model, features):
    """驗證用：[B,K,1,M,T] → [B,類別數]，每首歌先平均片段 logits。"""
    if features.ndim != 5 or features.shape[2] != 1:
        raise ValueError("輸入必須是 [batch, chunks, 1, mel, time]")
    batch, chunks = features.shape[:2]
    logits = chunk_model(features.flatten(0, 1))
    if logits.ndim != 2 or logits.shape != (batch * chunks, 6) or not torch.isfinite(logits).all():
        raise ValueError("片段模型必須回傳有限的 [batch*chunks, 6] logits")
    return logits.reshape(batch, chunks, 6).mean(dim=1)


class RecordingPredictor:
    """未來 CNN 的完整 WAV 推論入口；呼叫者不用先人工裁切或轉頻譜。"""
    def __init__(self, chunk_model, stats, device="cpu"):
        self.device = torch.device(device)
        self.model = chunk_model.to(self.device)
        self.frontend = AudioFrontend(stats).to(self.device)
        self.class_names = stats.class_names

    @torch.inference_mode()
    def predict_wav(self, path):
        waveform = load_waveform(path, self.frontend.config)
        chunks, starts = crop_waveform(waveform, self.frontend.config, training=False)
        # 關閉 Dropout 和 BatchNorm 的訓練行為；結束後恢復模型原狀態。
        was_training = self.model.training
        self.model.eval()
        try:
            features = self.frontend(chunks.to(self.device))
            logits = recording_logits(self.model, features.unsqueeze(0))[0]
            probabilities = logits.softmax(dim=-1)
            order = torch.argsort(probabilities, descending=True, stable=True)[:3]
            return {
                "top3_labels": [self.class_names[index] for index in order.tolist()],
                "probabilities": probabilities.cpu().tolist(),
                "class_names": list(self.class_names),
                "logits": logits.cpu().tolist(),
                "crop_start_seconds": [start / self.frontend.config.sample_rate for start in starts],
            }
        finally:
            self.model.train(was_training)
