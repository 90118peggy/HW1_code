"""不依賴 PyTorch，可先驗證 WAV 解碼、切分與統計算法。"""

import csv
import math
import statistics
import struct
import tempfile
import unittest
import wave
from dataclasses import asdict
from pathlib import Path

from data_pipeline.audio_common import (
    AudioConfig, NormalizationStats, RunningMoments, fixed_crop_starts,
    read_pcm16_wav, read_records, training_fingerprint,
)
from data_pipeline.inspect_dataset import LABELS


class AudioCommonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_wav(self, values, channels=1):
        path = self.root / "sample.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(struct.pack("<" + "h" * len(values), *values))
        return path

    def test_pcm_scale_and_stereo_average(self):
        path = self.write_wav([-32768, 0, 32767, 32767], channels=2)
        samples, rate = read_pcm16_wav(path)
        self.assertEqual(rate, 24000)
        self.assertEqual(list(samples), [-0.5, 32767 / 32768])

    def test_truncated_wav_is_rejected(self):
        path = self.write_wav([100] * 20)
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "截斷"):
            read_pcm16_wav(path)

    def test_empty_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "空"):
            read_pcm16_wav(self.write_wav([]))

    def test_nine_crops_cover_head_tail_and_have_no_gaps(self):
        config = AudioConfig()
        self.assertEqual(config.crop_samples, 88560)
        starts = fixed_crop_starts(720000, config.crop_samples, 9)
        self.assertEqual(starts, [0, 78930, 157860, 236790, 315720, 394650, 473580, 552510, 631440])
        self.assertTrue(all(b <= a + config.crop_samples for a, b in zip(starts, starts[1:])))
        self.assertEqual(starts[-1] + config.crop_samples, 720000)

    def test_short_audio_reuses_zero_start(self):
        self.assertEqual(fixed_crop_starts(50, 100, 9), [0] * 9)

    def test_streaming_statistics_match_direct_calculation(self):
        values = [-100.0, -5.0, 0.0, 3.0, 40.0, 55.0]
        moments = RunningMoments()
        for group in (values[:2], values[2:]):
            mean = statistics.mean(group)
            moments.update(len(group), mean, sum((x - mean) ** 2 for x in group))
        self.assertAlmostEqual(moments.mean, statistics.mean(values))
        self.assertAlmostEqual(moments.std, statistics.pstdev(values))

    def test_statistics_save_reload_and_no_overwrite(self):
        stats = NormalizationStats(
            "dataset_A", asdict(AudioConfig()), -30.0, 10.0, 20, 1,
            "example", sorted(LABELS["dataset_A"]),
        )
        path = self.root / "stats.json"
        stats.save(path)
        self.assertEqual(NormalizationStats.load(path), stats)
        with self.assertRaises(FileExistsError):
            stats.save(path)
        with self.assertRaises(ValueError):
            NormalizationStats(**{**asdict(stats), "std": math.nan})

    def test_read_train_does_not_require_test_audio_or_labels(self):
        dataset = self.root / "dataset_A"
        dataset.mkdir()
        rows = [dict(sample_id="A_train", split="train", label="1960s", audio_path="audio/a.wav", sha256="a"),
                dict(sample_id="A_test", split="test", label="", audio_path="audio/missing.wav", sha256="b")]
        with (dataset / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        selected = read_records(dataset, "train")
        self.assertEqual([r["sample_id"] for r in selected], ["A_train"])
        self.assertEqual(training_fingerprint(selected), training_fingerprint(rows[:1]))
        self.assertNotEqual(training_fingerprint(selected), training_fingerprint(rows))


if __name__ == "__main__":
    unittest.main()
