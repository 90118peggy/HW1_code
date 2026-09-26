"""用合成 WAV 驗證前處理、資料隔離與整首音訊推論。"""

import csv
import tempfile
import unittest
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from audio_processing import (AudioClassifier, AudioConfig, AudioFrontend,
                              ManifestAudioDataset, collate_audio, load_frontend, read_wav)


def write_wav(path, values, rate=24000, width=2, channels=1):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(values)


class AudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset_A"
        (self.root / "audio").mkdir(parents=True)
        self.config = AudioConfig(crop_seconds=0.1, eval_crops=3, n_fft=128,
                                  hop_length=64, n_mels=12)
        rows = []
        for index, split in enumerate(("train", "train", "validation", "test")):
            time = np.arange(4800 + index * 400) / 24000
            pcm = (np.sin(2 * np.pi * (440 + index * 100) * time) * 16000).astype("<i2")
            relative = f"audio/{index}.wav"
            write_wav(self.root / relative, pcm.tobytes())
            rows.append({"sample_id": str(index), "split": split,
                         "audio_path": relative, "label": "" if split == "test" else "1960s"})
        with (self.root / "manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def fitted(self):
        frontend = AudioFrontend(self.config)
        frontend.fit_statistics(ManifestAudioDataset(self.root, "train"))
        return frontend

    def test_pcm_widths_and_stereo(self):
        cases = [(1, bytes([0, 128, 255]), [-1, 0, 127 / 128]),
                 (2, np.array([-32768, 0, 16384], dtype="<i2").tobytes(), [-1, 0, 0.5]),
                 (3, b"\x00\x00\x80\x00\x00\x00\x00\x00\x40", [-1, 0, 0.5]),
                 (4, np.array([-2147483648, 0, 1073741824], dtype="<i4").tobytes(), [-1, 0, 0.5])]
        path = self.root / "audio/check.wav"
        for width, raw, expected in cases:
            with self.subTest(width=width):
                write_wav(path, raw, width=width)
                torch.testing.assert_close(read_wav(path), torch.tensor(expected, dtype=torch.float32))
        write_wav(path, np.array([[16384, 0], [-16384, 16384]], dtype="<i2").tobytes(), channels=2)
        torch.testing.assert_close(read_wav(path), torch.tensor([0.25, 0.0]))

    def test_invalid_audio(self):
        path = self.root / "audio/check.wav"
        write_wav(path, b"\x00\x00", rate=16000)
        with self.assertRaisesRegex(ValueError, "取樣率"):
            read_wav(path)
        write_wav(path, b"")
        with self.assertRaisesRegex(ValueError, "空音訊"):
            read_wav(path)
        write_wav(path, bytes(100))
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "截斷"):
            read_wav(path)

    def test_fixed_positions_and_padding(self):
        frontend = AudioFrontend(self.config)
        signal = torch.arange(4800, dtype=torch.float32)
        crops = frontend.crop_waveform(signal, training=False)
        torch.testing.assert_close(crops[:, 0], torch.tensor([0., 1200., 2400.]))
        self.assertEqual(crops.shape, (3, 2400))
        short = frontend.crop_waveform(torch.ones(10), training=False)
        self.assertTrue((short[:, :10] == 1).all())
        self.assertTrue((short[:, 10:] == 0).all())
        center = AudioFrontend(AudioConfig(**{**asdict(self.config), "eval_crops": 1}))
        self.assertEqual(center.crop_waveform(signal, training=False)[0, 0], 1200)

    def test_random_crops_reproducible_and_vary(self):
        frontend = AudioFrontend(self.config)
        signal = torch.arange(4800, dtype=torch.float32)
        torch.manual_seed(42)
        first = [frontend.crop_waveform(signal, training=True)[0, 0].item() for _ in range(10)]
        torch.manual_seed(42)
        second = [frontend.crop_waveform(signal, training=True)[0, 0].item() for _ in range(10)]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)
        self.assertTrue(all(0 <= x <= 2400 for x in first))

    def test_log_mel_finite_and_frequency(self):
        frontend = AudioFrontend(self.config)
        silence = frontend.log_mel(torch.zeros(1, 2400))
        self.assertTrue(torch.isfinite(silence).all())
        torch.testing.assert_close(silence, torch.full_like(silence, np.log(self.config.log_floor)))
        def peak(hz):
            signal = torch.sin(2 * torch.pi * hz * torch.arange(2400) / 24000)
            return frontend.log_mel(signal[None]).mean(-1).argmax().item()
        self.assertLess(peak(440), peak(4000))

    def test_statistics_train_only_and_normalized(self):
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "train"):
                AudioFrontend(self.config).fit_statistics(ManifestAudioDataset(self.root, split))
        # 非 train 音檔無效，也不能影響統計計算。
        (self.root / "audio/2.wav").write_bytes(b"invalid")
        (self.root / "audio/3.wav").write_bytes(b"invalid")
        frontend = self.fitted().eval()
        dataset = ManifestAudioDataset(self.root, "train")
        output = frontend([dataset[i]["waveform"] for i in range(len(dataset))])
        values = output.squeeze(2).permute(2, 0, 1, 3).reshape(self.config.n_mels, -1)
        torch.testing.assert_close(values.mean(1), torch.zeros(self.config.n_mels), atol=2e-5, rtol=0)
        torch.testing.assert_close(values.std(1, correction=0), torch.ones(self.config.n_mels), atol=2e-5, rtol=0)

    def test_silence_and_unfitted_guard(self):
        frontend = AudioFrontend(self.config)
        with self.assertRaisesRegex(RuntimeError, "fit_statistics"):
            frontend([torch.zeros(100)])
        for index in (0, 1):
            write_wav(self.root / f"audio/{index}.wav", bytes(100))
        frontend.fit_statistics(ManifestAudioDataset(self.root, "train"))
        self.assertTrue(torch.isfinite(frontend([torch.zeros(100)])).all())

    def test_full_recording_prediction_aggregation_and_backward(self):
        frontend = self.fitted()
        classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(1, 6))
        model = AudioClassifier(classifier, frontend)
        batch = next(iter(DataLoader(ManifestAudioDataset(self.root, "train"), batch_size=2,
                                     collate_fn=collate_audio)))
        logits = model(batch["waveforms"])
        nn.functional.cross_entropy(logits, batch["targets"]).backward()
        self.assertTrue(torch.isfinite(classifier[-1].weight.grad).all())
        paths = [self.root / "audio/3.wav"]
        probs = model.predict_files(paths)
        self.assertTrue(model.training)
        torch.testing.assert_close(probs, model.predict_files(paths), atol=0, rtol=0)
        model.eval()
        features = frontend([read_wav(paths[0])]).flatten(0, 1)
        expected = classifier(features).softmax(-1).mean(0, keepdim=True)
        torch.testing.assert_close(probs, expected)
        torch.testing.assert_close(probs.sum(-1), torch.ones(1))
        test_item = ManifestAudioDataset(self.root, "test")[0]
        self.assertEqual(test_item["target"], -1)

    def test_checkpoint_round_trip(self):
        frontend = self.fitted().eval()
        path = Path(self.temp.name) / "stats.pt"
        torch.save({"config": asdict(self.config), "state_dict": frontend.state_dict()}, path)
        restored = load_frontend(path).eval()
        audio = [read_wav(self.root / "audio/0.wav")]
        torch.testing.assert_close(frontend(audio), restored(audio), atol=0, rtol=0)
        mismatch = AudioFrontend(AudioConfig(**{**asdict(self.config), "eval_crops": 5}))
        with self.assertRaisesRegex(ValueError, "設定不符"):
            mismatch.load_state_dict(frontend.state_dict())


if __name__ == "__main__":
    unittest.main()
