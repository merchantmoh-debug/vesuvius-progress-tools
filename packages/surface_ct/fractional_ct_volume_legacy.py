"""Opt-in fractional scale for the pinned source CPU CT sampler.

The upstream constructor alone rejects non-integral input_scale. Its native
block reader, float64 coordinate transforms and trilinear crop kernel are
inherited unchanged. This requires the pinned legacy CTVolume interface; see README.
"""
import numpy as np

from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume, open_ct_array
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import validate_record


class FractionalCTVolume(CTVolume):
    def __init__(self, spec, cache_bytes: int = 3 << 30, *, cache_only: bool = False) -> None:
        self.spec = spec
        self._cache_bytes = int(cache_bytes)
        scales = np.asarray([spec.grid_scale, spec.ct_grid_scale], dtype=np.float64)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Invalid voxel scale')
        self.input_scale = float(spec.grid_scale / spec.ct_grid_scale)
        if not spec.ct_zarr:
            raise ValueError('CT input mode needs ct_zarr')
        self.ct = open_ct_array(spec, int(cache_bytes * .75), cache_only=cache_only)
        if self.ct.dtype != np.dtype('uint8'):
            raise ValueError('CT intensity normalization currently requires uint8 data')
        if spec.ct_normalization is not None:
            validate_record(spec.ct_normalization, spec)
        self.shape = tuple(np.ceil(np.asarray(self.ct.shape) / self.input_scale).astype(int))
