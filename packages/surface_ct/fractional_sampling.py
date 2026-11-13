"""Explicit fractional-scale affine CT sampling, without an integer-ratio restriction."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates


def affine_sample(volume, origin_xyz, axes_xyz, shape_zyx, *, trace_pitch_um, ct_pitch_um):
    """axes_xyz rows are the output X/Y/Z directions; origin_xyz is the first sample centre.

    origin_xyz uses native CT voxel coordinates. Output-axis increments use the
    trace_pitch_um / ct_pitch_um ratio. No intensity normalization is applied.
    """
    volume=np.asarray(volume)
    origin=np.asarray(origin_xyz,dtype=float);axes=np.asarray(axes_xyz,dtype=float)
    shape=tuple(shape_zyx);scales=np.asarray([trace_pitch_um,ct_pitch_um],float)
    if volume.ndim!=3 or origin.shape!=(3,) or axes.shape!=(3,3) or len(shape)!=3:
        raise ValueError('Invalid volume, origin, axes or shape')
    if any(type(n) is not int or n<1 for n in shape) or not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Positive dimensions and physical scales required')
    if not np.isfinite(origin).all() or not np.isfinite(axes).all():raise ValueError('Non-finite frame')
    if not np.allclose(axes@axes.T,np.eye(3),atol=1e-6):raise ValueError('Axes must be orthonormal')
    z,y,x=np.indices(shape,dtype=np.float64)
    points=origin+(x[...,None]*axes[0]+y[...,None]*axes[1]+z[...,None]*axes[2])*(scales[0]/scales[1])
    upper=np.array(volume.shape[::-1])-1
    if np.any(points<0) or np.any(points>upper):raise ValueError('Sampling leaves the actual CT; no fabricated boundary fill')
    return map_coordinates(volume.astype(np.float64,copy=False),points[...,::-1].reshape(-1,3).T,
                           order=1,mode='nearest',prefilter=False).reshape(shape)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--frame',type=Path,required=True,help='JSON: origin_xyz, axes_xyz, shape_zyx, trace_pitch_um, ct_pitch_um')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.suffix!='.npy':raise ValueError('Output must end in .npy')
    if a.output.exists():raise FileExistsError(a.output)
    cfg=json.loads(a.frame.read_text(encoding='utf-8'))
    result=affine_sample(np.load(a.input,mmap_mode='r',allow_pickle=False),**cfg)
    np.save(a.output,result)


if __name__=='__main__':main()
