"""Bounded, content-addressed storage for unchanged Hecate tile predictions.

MIT License. Copyright (c) 2026 Mohamad Al-Zawahreh.
Use one inference process per cache directory. No model or raw CT is stored.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable

import numpy as np


class CacheIntegrityError(ValueError):
    """A retained tile disagrees with its declared input, model, or checksum."""


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _publish(path: Path, data: bytes) -> None:
    """Publish complete bytes; retry only a transient Windows sharing denial."""
    fd, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(10):
            try:
                temporary.replace(path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) != 5 or attempt == 9:
                    raise
                time.sleep(.05)
    finally:
        temporary.unlink(missing_ok=True)


class TileCache:
    """Reuse only identical uint8 tiles under an identical execution binding.

    The caller binds model, preprocessing, implementation and runtime identity.
    The byte ceiling covers all prediction entries under this cache root.
    Corruption raises an error instead of returning or silently replacing data.
    """

    def __init__(self, root: Path, binding: dict, max_bytes: int = 128 * 2**20):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        if not isinstance(binding, dict) or not binding:
            raise ValueError("An explicit execution binding is required")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / ".hecate-cache-format"
        magic = b"hecate-probability-cache-v1\n"
        if marker.exists():
            if marker.read_bytes() != magic:
                raise ValueError("Not a Hecate tile-cache directory")
        elif any(self.root.iterdir()):
            raise ValueError("Choose an empty directory for a new tile cache")
        else:
            _publish(marker, magic)
        self.binding = _json_bytes(binding)
        self.namespace = hashlib.sha256(self.binding).hexdigest()
        self.directory = self.root / self.namespace
        self.directory.mkdir(exist_ok=True)
        manifest = self.directory / "binding.json"
        if manifest.exists():
            if manifest.read_bytes() != self.binding:
                raise CacheIntegrityError("Execution binding changed")
        else:
            _publish(manifest, self.binding)
        self.max_bytes = max_bytes
        self.hits = self.misses = self.evictions = 0
        self._enforce_capacity(0)

    def _entries(self) -> list[Path]:
        return [p for d in self.root.iterdir()
                if d.is_dir() and not d.is_symlink() and len(d.name) == 64
                and all(c in "0123456789abcdef" for c in d.name)
                for p in d.glob("*.npz")
                if not p.is_symlink() and len(p.stem) == 64
                and all(c in "0123456789abcdef" for c in p.stem)]

    def _enforce_capacity(self, incoming: int) -> None:
        entries = sorted(self._entries(), key=lambda p: (p.stat().st_mtime_ns, p.name))
        total = sum(p.stat().st_size for p in entries)
        for path in entries:
            if total + incoming <= self.max_bytes:
                break
            size = path.stat().st_size
            path.unlink()
            total -= size
            self.evictions += 1

    @staticmethod
    def _check_prediction(value: np.ndarray) -> None:
        if (not isinstance(value, np.ndarray) or value.shape != (64, 64)
                or value.dtype != np.float32 or not np.isfinite(value).all()
                or value.min() < 0 or value.max() > 1):
            raise ValueError("Expected finite float32 64x64 probabilities in [0,1]")

    def get_or_compute(self, raw: np.ndarray,
                       predict: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, bool]:
        if raw.shape != (16, 64, 64) or raw.dtype != np.uint8:
            raise ValueError("Expected uint8 16x64x64 model input")
        raw = np.ascontiguousarray(raw)
        key = hashlib.sha256(raw.tobytes()).hexdigest()
        path = self.directory / (key + ".npz")
        if path.is_symlink():
            raise CacheIntegrityError("Cached prediction must not be a symbolic link")
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    if saved["input_sha256"].item() != key or saved["binding_sha256"].item() != self.namespace:
                        raise CacheIntegrityError("Cached tile belongs to different inputs or execution")
                    value = saved["probability"]
                    self._check_prediction(value)
                    actual = hashlib.sha256(value.tobytes()).hexdigest()
                    if saved["probability_sha256"].item() != actual:
                        raise CacheIntegrityError("Cached prediction checksum differs")
            except (OSError, KeyError, ValueError) as exc:
                raise CacheIntegrityError(f"Invalid cached prediction: {path}") from exc
            path.touch()
            self.hits += 1
            return value, True
        value = predict(raw)
        self._check_prediction(value)
        self.misses += 1
        buffer = io.BytesIO()
        np.savez(buffer, probability=value, input_sha256=key,
                 binding_sha256=self.namespace,
                 probability_sha256=hashlib.sha256(value.tobytes()).hexdigest())
        data = buffer.getvalue()
        if len(data) <= self.max_bytes:
            self._enforce_capacity(len(data))
            _publish(path, data)
        return value, False

    def stats(self) -> dict:
        return {"cache_hits": self.hits, "cache_misses": self.misses,
                "cache_evictions": self.evictions,
                "cache_prediction_bytes": sum(p.stat().st_size for p in self._entries()),
                "cache_limit_bytes": self.max_bytes}
