"""Fit a physical depth shift from CT profiles using fixed native-bin averaging."""
import argparse
import json
from pathlib import Path
import numpy as np


def correlation(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    good=np.isfinite(a)&np.isfinite(b)
    if good.sum()<3:return None
    x=a[good]-a[good].mean();y=b[good]-b[good].mean()
    denominator=np.linalg.norm(x)*np.linalg.norm(y)
    return None if denominator==0 else float(x@y/denominator)


def bin_profile(sample_z_um, values, centres_um, width_um):
    z=np.asarray(sample_z_um,float);v=np.asarray(values,float)
    if z.ndim!=1 or v.shape!=z.shape or np.any(np.diff(z)<=0) or not np.isfinite(z).all() or not np.isfinite(v).all():
        raise ValueError('Finite increasing physical depths and one value per sample are required')
    if not np.isfinite(width_um) or width_um<=0:raise ValueError('Positive native-bin width required')
    rows=[];counts=[]
    for centre in np.asarray(centres_um,float):
        use=(z>=centre-width_um/2)&(z<centre+width_um/2)
        counts.append(int(use.sum()));rows.append(float(v[use].mean()) if use.any() else np.nan)
    return np.asarray(rows),counts


def fit_shift(reference_z_um, reference, moving_z_um, moving, shifts_um, bin_width_um):
    z=np.asarray(reference_z_um,float);reference=np.asarray(reference,float);shifts=np.asarray(shifts_um,float)
    if z.ndim!=1 or z.shape!=reference.shape or not np.isfinite(z).all() or not np.isfinite(reference).all():
        raise ValueError('Invalid reference profile')
    if shifts.ndim!=1 or len(shifts)<3 or not np.isfinite(shifts).all() or np.any(np.diff(shifts)<=0):
        raise ValueError('Supply a fixed increasing shift grid')
    scores=[];support=[]
    for shift in shifts:
        samples,counts=bin_profile(moving_z_um,moving,z+shift,bin_width_um)
        scores.append(correlation(reference,samples));support.append(counts)
    finite=np.array([s is not None for s in scores])
    if not finite.any():raise ValueError('No informative common profile support')
    index=int(np.argmax([s if s is not None else -np.inf for s in scores]))
    return {'shift_um':float(shifts[index]),'correlation':scores[index],
            'at_search_boundary':index in (0,len(shifts)-1),'shifts_um':shifts.tolist(),'correlations':scores,
            'selected_bin_counts':support[index],
            'sign_convention':'Sample moving CT at reference_z_um + shift_um',
            'scope':'CT-profile correspondence on the supplied fitting region; no ink or face identification.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='JSON: reference_z_um, reference, moving_z_um, moving, shifts_um, bin_width_um')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=fit_shift(**json.loads(a.input.read_text(encoding='utf-8')))
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(a.output)


if __name__=='__main__':main()
