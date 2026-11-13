"""Gather exact native CT sections and nearest vertex samples without a full cube."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice, product

import numpy as np


def required_keys(origin, shape, xy_z, yz_x, nearest_zyx, chunk_size=128):
    origin = np.asarray(origin, dtype=int)
    shape = np.asarray(shape, dtype=int)
    points = np.asarray(nearest_zyx, dtype=int)
    assert np.all(origin % chunk_size == 0) and np.all(shape % chunk_size == 0)
    assert points.ndim == 2 and points.shape[1] == 3
    assert np.all(points >= origin) and np.all(points < origin + shape)
    ranges = [range(int(o // chunk_size), int((o + s) // chunk_size))
              for o, s in zip(origin, shape)]
    keys = set(map(tuple, points // chunk_size))
    for z in xy_z:
        assert type(z) is int and origin[0] <= z < origin[0] + shape[0]
        keys.update((z // chunk_size, y, x) for y, x in product(ranges[1], ranges[2]))
    for x in yz_x:
        assert type(x) is int and origin[2] <= x < origin[2] + shape[2]
        keys.update((z, y, x // chunk_size) for z, y in product(ranges[0], ranges[1]))
    return sorted(keys)


def ordered_chunks(keys, acquire, workers):
    """Keep only a bounded number of independent chunk reads in flight."""
    assert type(workers) is int and 1 <= workers <= 4
    if workers == 1:
        for key in keys:
            yield key, acquire(key)
        return
    remaining = iter(keys)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque((key, pool.submit(acquire, key))
                        for key in islice(remaining, workers))
        while pending:
            key, future = pending.popleft()
            yield key, future.result()
            next_key = next(remaining, None)
            if next_key is not None:
                pending.append((next_key, pool.submit(acquire, next_key)))


def gather(origin, shape, xy_z, yz_x, nearest_zyx, acquire, chunk_size=128, workers=1):
    """Acquire each needed chunk once; preserve the full-cube slice/index semantics."""
    origin = np.asarray(origin, dtype=int)
    shape = np.asarray(shape, dtype=int)
    points = np.asarray(nearest_zyx, dtype=int)
    keys = required_keys(origin, shape, xy_z, yz_x, points, chunk_size)
    xy = {z: np.empty((shape[1], shape[2]), np.uint8) for z in xy_z}
    yz = {x: np.empty((shape[0], shape[1]), np.uint8) for x in yz_x}
    xy_done = {z: np.zeros_like(field, dtype=bool) for z, field in xy.items()}
    yz_done = {x: np.zeros_like(field, dtype=bool) for x, field in yz.items()}
    values = np.empty(len(points), np.uint8)
    assigned = np.zeros(len(points), dtype=bool)
    point_keys = points // chunk_size
    for key, chunk in ordered_chunks(keys, acquire, workers):
        assert chunk.shape == (chunk_size,) * 3 and chunk.dtype == np.uint8
        global_start = np.array(key) * chunk_size
        z0, y0, x0 = global_start - origin
        for z in xy:
            if z // chunk_size == key[0]:
                xy[z][y0:y0+chunk_size, x0:x0+chunk_size] = chunk[z-global_start[0]]
                xy_done[z][y0:y0+chunk_size, x0:x0+chunk_size] = True
        for x in yz:
            if x // chunk_size == key[2]:
                yz[x][z0:z0+chunk_size, y0:y0+chunk_size] = chunk[:, :, x-global_start[2]]
                yz_done[x][z0:z0+chunk_size, y0:y0+chunk_size] = True
        selected = np.all(point_keys == key, axis=1)
        if selected.any():
            local = points[selected] - global_start
            values[selected] = chunk[tuple(local.T)]
            assigned[selected] = True
    assert assigned.all()
    assert all(mask.all() for mask in [*xy_done.values(), *yz_done.values()])
    return xy, yz, values, keys
