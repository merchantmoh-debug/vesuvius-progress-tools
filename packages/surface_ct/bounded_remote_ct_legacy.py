"""Bounded, single-process HTTP transport for the upstream CT chunk reader.

Only the cache-miss transport changes. Upstream interpolation, normalization,
metadata validation and LRU logic remain unchanged. The task owns this cache;
thread locks prevent duplicate downloads without requiring POSIX file locks.
"""
from pathlib import Path
import hashlib
import json
import threading

import requests


def install_bounded_http_cache(*, remote_url, level, cache_root, maximum_network_bytes, receipt_path):
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import RemoteChunkedArray, ChunkedArray
    cache_root = Path(cache_root).resolve()
    receipt_path = Path(receipt_path)
    byte_count = 128**3
    state = dict(reserved_bytes=0, downloaded_bytes=0, requests=0, hits=0, sparse_chunks=0)
    guard = threading.Lock()
    locks = {}
    original = RemoteChunkedArray._load

    def load(array, key):
        assert array.remote_url == remote_url and array.level == level
        assert Path(array.path).resolve().is_relative_to(cache_root)
        assert array.chunks == (128,)*3 and str(array.dtype) == 'uint8' and array.codec is None
        key = tuple(int(v) for v in key)
        assert len(key) == 3 and all(0 <= v < (s+127)//128 for v, s in zip(key, array.shape))
        cache_key = (array.path, key)
        with guard:
            lock = locks.setdefault(cache_key, threading.Lock())
        with lock:
            value = ChunkedArray._load(array, key)
            if value is not None:
                with guard:
                    state['hits'] += 1
                return value
            if array.cache_only:
                raise FileNotFoundError(f'Cache-only chunk missing: {key}')
            with guard:
                if state['reserved_bytes']+byte_count > maximum_network_bytes:
                    raise RuntimeError('Bounded CT network budget exhausted')
                state['reserved_bytes'] += byte_count
                state['requests'] += 1
            base = remote_url.replace('s3://vesuvius-challenge-open-data',
                'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com').rstrip('/')
            url = base+'/'+str(level)+'/'+('/'.join(map(str, key)))
            with requests.get(url, headers={'Accept-Encoding': 'identity'}, stream=True, timeout=(10, 20)) as response:
                if response.status_code == 404:
                    assert array.fill == 0
                    data, fetched, sparse = bytes(byte_count), 0, True
                else:
                    response.raise_for_status()
                    assert response.status_code == 200
                    data = response.raw.read(byte_count+1)
                    assert len(data) == byte_count
                    fetched, sparse = byte_count, False
                etag = response.headers.get('ETag')
            target = Path(array.path)/array.sep.join(map(str, key))
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name+'.partial')
            assert not temp.exists()
            with temp.open('xb') as stream:
                stream.write(data)
            temp.replace(target)
            entry = dict(key=key, url=url, etag=etag, sha256=hashlib.sha256(data).hexdigest(),
                         stored_bytes=byte_count, downloaded_bytes=fetched,
                         status='declared_fill_404' if sparse else 'downloaded_200')
            with guard:
                state['downloaded_bytes'] += fetched
                state['sparse_chunks'] += int(sparse)
                with receipt_path.open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(entry)+'\n')
            return ChunkedArray._load(array, key)

    RemoteChunkedArray._load = load
    return state, original
