"""Preserve fibre endpoints and measure between-training-point phase consistency."""
import argparse
import json
from pathlib import Path
import numpy as np


def retained_indices(points, spacing, force_keep=()):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite Nx3 fibre required')
    if not np.isfinite(spacing) or spacing<0:raise ValueError('Nonnegative spacing required')
    forced=set(force_keep)
    if any(type(i) is not int or not 0<=i<len(points) for i in forced):raise ValueError('Invalid forced index')
    if len(points)<=1 or spacing==0:return np.arange(len(points),dtype=np.int64)
    forced.add(len(points)-1);keep=[0];last=points[0]
    for i in range(1,len(points)):
        if i in forced or np.linalg.norm(points[i]-last)>=spacing:
            keep.append(i);last=points[i]
    return np.asarray(keep,dtype=np.int64)


def compare_phase(control, candidate, control_training_indices, candidate_training_indices, target_phase=None):
    """Score a fixed common population excluded from both training memberships."""
    a,b=np.asarray(control),np.asarray(candidate)
    if (a.ndim!=1 or a.shape!=b.shape or not np.issubdtype(a.dtype,np.number)
        or not np.issubdtype(b.dtype,np.number) or np.iscomplexobj(a) or np.iscomplexobj(b)
        or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Matching finite phase arrays required')
    mask=np.ones(len(a),bool)
    for indices in (control_training_indices,candidate_training_indices):
        ix=np.asarray(indices)
        if ix.size:
            if ix.ndim!=1 or not np.issubdtype(ix.dtype,np.integer) or np.any(ix<0) or np.any(ix>=len(a)):
                raise ValueError('Invalid training membership')
            mask[ix]=False
    if not mask.any():raise ValueError('No excluded evaluation population')
    if target_phase is None:
        centre_a=float(np.median(a[mask]));centre_b=float(np.median(b[mask]))
    else:
        if not np.isfinite(target_phase):raise ValueError('Invalid phase target')
        centre_a=centre_b=float(target_phase)
    qa=np.quantile(abs(a[mask]-centre_a),[.5,.9,.99,1])
    qb=np.quantile(abs(b[mask]-centre_b),[.5,.9,.99,1])
    return {'evaluation_points':int(mask.sum()),'quantiles':[.5,.9,.99,1],
            'control_centre_phase':centre_a,'candidate_centre_phase':centre_b,
            'centering':'Per-arm excluded-evaluation median, preserving input arithmetic dtype' if target_phase is None else 'Fixed supplied phase',
            'control_absolute_phase_error':qa.tolist(),'candidate_absolute_phase_error':qb.tolist(),
            'p90_relative_reduction':None if qa[1]==0 else float(1-qb[1]/qa[1]),
            'scope':'Consistency along each guide; not anatomical sheet or ink acceptance.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--points',type=Path,required=True)
    p.add_argument('--spacing',type=float,required=True)
    p.add_argument('--output',type=Path,required=True,help='New JSON with retained indices and tail distances')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    pts=np.load(a.points,allow_pickle=False);idx=retained_indices(pts,a.spacing)
    result={'input_points':len(pts),'retained_points':len(idx),'indices':idx.tolist(),
            'first_and_last_preserved':not len(pts) or (idx[0]==0 and idx[-1]==len(pts)-1)}
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
