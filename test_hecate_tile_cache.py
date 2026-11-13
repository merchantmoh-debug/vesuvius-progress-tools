"""Storage/invalidation regressions; the neural real-data check is separate."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hecate_tile_cache import CacheIntegrityError, TileCache


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "cache"
        self.raw = np.arange(16*64*64, dtype=np.uint8).reshape(16,64,64)
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def predict(self, raw):
        self.calls += 1
        return raw.mean(axis=0, dtype=np.float32) / np.float32(255)

    def test_same_input_reuses_exact_float_values(self):
        cache = TileCache(self.root, {"model":"a"})
        first, hit = cache.get_or_compute(self.raw, self.predict)
        self.assertFalse(hit)
        second, hit = cache.get_or_compute(self.raw.copy(), self.predict)
        self.assertTrue(hit)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(self.calls, 1)

    def test_changed_pixel_or_model_cannot_reuse(self):
        cache = TileCache(self.root, {"model":"a"})
        cache.get_or_compute(self.raw, self.predict)
        changed = self.raw.copy(); changed[0,0,0] ^= np.uint8(1)
        self.assertFalse(cache.get_or_compute(changed, self.predict)[1])
        self.assertFalse(TileCache(self.root, {"model":"b"}).get_or_compute(self.raw, self.predict)[1])
        self.assertEqual(self.calls, 3)

    def test_completed_tile_survives_interrupted_next_forward(self):
        cache = TileCache(self.root, {"model":"a"})
        expected = cache.get_or_compute(self.raw, self.predict)[0]
        changed = self.raw.copy(); changed[0,0,0] ^= np.uint8(1)
        def interrupted(raw):
            raise RuntimeError("simulated interruption before a completed prediction")
        with self.assertRaises(RuntimeError):
            cache.get_or_compute(changed, interrupted)
        resumed = TileCache(self.root, {"model":"a"})
        got, hit = resumed.get_or_compute(self.raw, interrupted)
        self.assertTrue(hit); self.assertTrue(np.array_equal(got, expected))
        self.assertFalse(resumed.get_or_compute(changed, self.predict)[1])

    def test_corrupted_prediction_is_rejected(self):
        cache = TileCache(self.root, {"model":"a"})
        cache.get_or_compute(self.raw, self.predict)
        path = next(cache.directory.glob("*.npz"))
        with np.load(path, allow_pickle=False) as saved:
            values = {k:saved[k] for k in saved.files}
        values["probability"] = values["probability"].copy()
        values["probability"][0,0] = np.float32(.99)
        np.savez(path, **values)
        with self.assertRaises(CacheIntegrityError):
            cache.get_or_compute(self.raw, self.predict)

    def test_capacity_is_bounded_across_model_namespaces(self):
        a = TileCache(self.root, {"model":"a"}, max_bytes=20000)
        a.get_or_compute(self.raw, self.predict)
        b = TileCache(self.root, {"model":"b"}, max_bytes=20000)
        b.get_or_compute(self.raw, self.predict)
        self.assertLessEqual(b.stats()["cache_prediction_bytes"], 20000)
        self.assertEqual(b.evictions, 1)
        self.assertEqual(len(list(self.root.rglob("*.npz"))), 1)

    def test_unrelated_directory_is_not_used_or_modified(self):
        self.root.mkdir(); note=self.root/'keep.txt';note.write_text('keep')
        with self.assertRaises(ValueError):TileCache(self.root,{"model":"a"})
        self.assertEqual(note.read_text(),'keep')


if __name__ == "__main__":
    unittest.main()
