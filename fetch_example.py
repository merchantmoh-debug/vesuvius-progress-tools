"""Fetch a small PHerc0139 example from the official server; no CT is redistributed.

Data terms: https://dl.ash2txt.org/LICENSE.txt
This reconstructs the exposed backend-comparison input, not held-out ink labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.ndimage import map_coordinates

BASE = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
        'PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/'
        '9.362um-1.2m-113keV-volume-20250728140407.zarr/0')
CHUNKS = {
    (35, 20): 'e1594169b84799d3556d2907c94be7aa145f53f37741b3fabc674b1b187d29c1',
    (35, 21): 'f3781ae4d3391ab433457874c2d57901ded5e9ef4a7a81bc4165bf4b6b1b1ca8',
    (36, 20): '7e103b25b0851a6b340ab03e128c5d006a5ac1bebb7c6ea7084ca2a9772150a4',
    (36, 21): 'b982a846f80eb78a472d760994446bad9824864e8e42c9d97e69e9ee005776cc',
}
DEPTH = [4.4291245739319605, 5.454546492325465, 6.479968410718972,
         7.505390329112477, 8.530812247505981, 9.55623416589949,
         10.581656084292995, 11.6070780026865, 12.632499921080004,
         13.65792183947351, 14.683343757867018, 15.708765676260523,
         16.734187594654028, 17.759609513047533, 18.785031431441038,
         19.810453349834543]
EXPECTED = '04224e26f565c4a5c1ff7f018e4904a381e7d0d6a217a6ade09ab6ec313e312a'


def fetch(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    with urlopen(BASE + '/.zarray', timeout=30) as response:
        meta = json.load(response)
    expected = {'shape': [28, 6120, 8120], 'chunks': [28, 128, 128], 'dtype': '|u1',
                'compressor': None, 'filters': None, 'order': 'C', 'dimension_separator': '/'}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('The public source metadata changed')
    block = np.empty((28, 256, 256), dtype=np.uint8)
    for (y, x), digest in CHUNKS.items():
        with urlopen(f'{BASE}/0/{y}/{x}', timeout=30) as response:
            raw = response.read(28 * 128 * 128 + 1)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'Source chunk changed: {y}/{x}')
        block[:, (y-35)*128:(y-34)*128, (x-20)*128:(x-19)*128] = np.frombuffer(raw, np.uint8).reshape(28, 128, 128)
    # The original 512-pixel crop began at Y=4320, X=2440. Its resampled
    # extent was 499 pixels; this example takes [192:288,192:288] from it.
    xy = (np.arange(192, 288) + .5) * 512 / 499 - .5
    coordinates = np.array(np.broadcast_arrays(np.array(DEPTH)[:, None, None],
        (xy - 160)[None, :, None], (xy - 120)[None, None, :]))
    image = map_coordinates(block.astype(np.float32), coordinates, order=1, prefilter=False)
    image = image.clip(0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.save(stream, image, allow_pickle=False)
    if hashlib.sha256(output.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('The reconstructed example differs from the recorded input')
    print(f'{output}: {EXPECTED}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('pherc0139_crop.npy'))
    fetch(parser.parse_args().output)
