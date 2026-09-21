"""需在已安裝 PyTorch/SciPy 的環境執行；測試使用合成 WAV。"""

import csv
import tempfile
import unittest
import wave
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from data_pipeline.audio_common import AudioConfig, NormalizationStats, read_records, training_fingerprint
from inference.predictor import RecordingPredictor, recording_logits
from data_pipeline.audio_pipeline import (
    AudioFrontend, HW1AudioDataset, LogMel,
    crop_waveform, fit_normalization, load_waveform,
)
from data_pipeline.inspect_dataset import LABELS


class MeanModel(nn.Module):
    def forward(self, features):
        value = features.mean(dim=(1, 2, 3))
        return value[:, None] * torch.arange(1, 7, device=features.device)[None, :]


class AudioPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = AudioConfig()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset = Path(self.temp.name) / "dataset_A"
        (self.dataset / "audio").mkdir(parents=True)
        self.rows = []
        for split in ("train", "validation", "test"):
            path = self.dataset / "audio" / f"{split}.wav"
            self.write_sine(path, 24000, 4.0)
            self.rows.append(dict(sample_id=f"A_{split}", split=split,
                                  label="" if split == "test" else "1960s",
                                  audio_path=f"audio/{split}.wav", sha256=split))
        with (self.dataset / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.stats = NormalizationStats(
            "dataset_A", asdict(self.config), -30.0, 10.0, 10, 1,
            training_fingerprint(self.rows[:1]), sorted(LABELS["dataset_A"]),
        )

    @staticmethod
    def write_sine(path, rate, seconds):
        t = torch.arange(round(rate * seconds)) / rate
        samples = (10000 * torch.sin(2 * torch.pi * 440 * t)).to(torch.int16)
        from array import array
        import sys
        pcm = array("h", samples.tolist())
        if sys.byteorder != "little":
            pcm.byteswap()
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(pcm.tobytes())

    def test_resampling_changes_length_not_tone_frequency(self):
        path = self.dataset / "audio" / "rate.wav"
        self.write_sine(path, 48000, 1.0)
        waveform = load_waveform(path, self.config)
        self.assertEqual(waveform.numel(), 24000)
        peak = torch.fft.rfft(waveform).abs().argmax().item()
        self.assertAlmostEqual(peak, 440, delta=1)

    def test_seed_reproducibility_without_resetting_every_crop(self):
        waveform = torch.arange(120000, dtype=torch.float32)
        first = torch.Generator().manual_seed(42)
        second = torch.Generator().manual_seed(42)
        starts = [crop_waveform(waveform, self.config, True, first)[1] for _ in range(5)]
        replay = [crop_waveform(waveform, self.config, True, second)[1] for _ in range(5)]
        self.assertEqual(starts, replay)
        self.assertGreater(len({s[0] for s in starts}), 1)

    def test_short_audio_padding_and_silence_stay_finite(self):
        chunks, starts = crop_waveform(torch.zeros(100), self.config)
        self.assertEqual(chunks.shape, (9, 88560))
        self.assertEqual(starts, [0] * 9)
        self.assertTrue(torch.isfinite(LogMel(self.config)(chunks)).all())

    def test_log_mel_batch_invariance_and_normalization(self):
        chunks = torch.randn(2, 88560) * torch.tensor([0.1, 0.01])[:, None]
        feature = LogMel(self.config)
        together = feature(chunks)
        separate = torch.cat([feature(chunk[None]) for chunk in chunks])
        torch.testing.assert_close(together, separate)
        self.assertEqual(together.shape, (2, 128, 346))
        standardized = AudioFrontend(self.stats)(chunks)
        torch.testing.assert_close(standardized[:, 0], (together + 30) / 10)

    def test_fit_uses_train_only_and_matches_direct_moments(self):
        # 若 fit 偷讀 validation/test，兩個不存在的 WAV 將讓測試失敗。
        for split in ("validation", "test"):
            (self.dataset / "audio" / f"{split}.wav").unlink()
        stats = fit_normalization(self.dataset, self.config)
        waveform = load_waveform(self.dataset / "audio/train.wav", self.config)
        chunks, _ = crop_waveform(waveform, self.config)
        values = LogMel(self.config)(chunks).double()
        self.assertAlmostEqual(stats.mean, values.mean().item(), places=6)
        self.assertAlmostEqual(stats.std, values.std(unbiased=False).item(), places=6)
        self.assertEqual(stats.recording_count, 1)

    def test_loader_train_eval_shapes_and_hidden_labels(self):
        train = HW1AudioDataset(self.dataset, "train", self.stats)[0]
        val = HW1AudioDataset(self.dataset, "validation", self.stats)[0]
        test = HW1AudioDataset(self.dataset, "test", self.stats)[0]
        self.assertEqual(train["features"].shape, (1, 128, 346))
        self.assertEqual(val["features"].shape, (9, 1, 128, 346))
        self.assertNotIn("target", test)
        self.assertIn("target", val)

    def test_statistics_mismatch_is_rejected(self):
        wrong = NormalizationStats(**{**asdict(self.stats), "training_fingerprint": "wrong"})
        with self.assertRaisesRegex(ValueError, "來源不符"):
            HW1AudioDataset(self.dataset, "train", wrong)

    def test_recording_logits_average_before_softmax(self):
        model = MeanModel()
        features = torch.randn(2, 9, 1, 4, 5)
        expected = torch.stack([model(record).mean(dim=0) for record in features])
        torch.testing.assert_close(recording_logits(model, features), expected)

    def test_full_wav_prediction_matches_manual_pipeline(self):
        model = MeanModel().train()
        predictor = RecordingPredictor(model, self.stats)
        path = self.dataset / "audio/test.wav"
        prediction = predictor.predict_wav(path)
        self.assertTrue(model.training)
        waveform = load_waveform(path, self.config)
        chunks, _ = crop_waveform(waveform, self.config)
        expected = model(AudioFrontend(self.stats)(chunks)).mean(dim=0)
        torch.testing.assert_close(torch.tensor(prediction["logits"]), expected)
        self.assertEqual(len(prediction["top3_labels"]), 3)
        self.assertEqual(len(set(prediction["top3_labels"])), 3)
        self.assertAlmostEqual(sum(prediction["probabilities"]), 1, places=5)
        self.assertEqual(prediction, predictor.predict_wav(path))

    @unittest.skipUnless(torch.cuda.is_available(), "需要 GPU 才執行 CPU/CUDA 一致性測試")
    def test_cuda_frontend_matches_cpu(self):
        chunks = torch.randn(2, 88560) * 0.1
        cpu = AudioFrontend(self.stats)(chunks)
        gpu = AudioFrontend(self.stats).cuda()(chunks.cuda()).cpu()
        torch.testing.assert_close(cpu, gpu, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
