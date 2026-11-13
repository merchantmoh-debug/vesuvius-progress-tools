"""Three exact brightness-slope coefficient fields, written in bounded Z slabs."""
from pathlib import Path
import gc
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

def build(volume, folder, gradient_sigma, tensor_sigma, slab_depth=32):
    """Return exact float32 Jxy, Jyy and Jyz maps for the supplied CT volume."""
    folder = Path(folder); folder.mkdir(exist_ok=True)
    assert volume.ndim == 3 and gradient_sigma > 0 and tensor_sigma > 0
    assert type(slab_depth) is int and slab_depth > 0
    # scipy.ndimage uses radius=int(truncate*sigma+0.5), with truncate=4.
    halo = int(4 * gradient_sigma + .5) + int(4 * tensor_sigma + .5)
    for key in ['01', '11', '12']:
        path = folder / (key + '.npy'); assert not path.exists()
        mapped = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=volume.shape)
        mapped.flush(); del mapped
    for start in range(0, volume.shape[0], slab_depth):
        stop = min(start + slab_depth, volume.shape[0])
        low = max(0, start - halo); high = min(volume.shape[0], stop + halo)
        block = volume[low:high]; keep = slice(start - low, stop - low)
        gy = gaussian_filter(block, gradient_sigma, order=(0, 1, 0), output=np.float32)
        for key, order in [('01', (0, 0, 1)), ('11', None), ('12', (1, 0, 0))]:
            other = gy if order is None else gaussian_filter(block, gradient_sigma, order=order, output=np.float32)
            product = gy * other
            smoothed = gaussian_filter(product, tensor_sigma, output=np.float32)
            # Flush writes, then close the mapping so dirty/resident output pages
            # do not accumulate across all slabs in this process's working set.
            mapped = np.load(folder / (key + '.npy'), mmap_mode='r+')
            mapped[start:stop] = smoothed[keep]
            mapped.flush(); del mapped
            del product, smoothed, other
        del gy, block
    gc.collect()
    return {key: np.load(folder / (key + '.npy'), mmap_mode='r') for key in ['01', '11', '12']}

def normals(fields, points, origin, spacing):
    """Use the same float64 normalized brightness slopes as sample_normals."""
    points = np.asarray(points)
    shape = points.shape[:-1]
    q = ((points[..., ::-1] - origin) / spacing).reshape(-1, 3).T
    field_shape = np.array(fields['11'].shape)
    assert np.all(q >= 1) and np.all(q < (field_shape - 2)[:, None]), 'surface left tensor domain'
    samples = {key: map_coordinates(field, q, order=1, prefilter=False).astype(np.float64).reshape(shape)
               for key, field in fields.items()}
    assert np.all(samples['11'] > 1e-6), 'no Y-gradient information'
    value = np.stack([samples['01'] / samples['11'], np.ones(shape), samples['12'] / samples['11']], -1)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)
