"""Recompute the two recorded between-node comparisons from model phase outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from fiber_guidance import compare_phase


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path(__file__).parent/'evidence/fibre_phases.npz')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.output and a.output.exists():raise FileExistsError(a.output)
    with np.load(a.input,allow_pickle=False) as data:
        rows={name:compare_phase(data[name+'_control'],data[name+'_dense'],data[name+'_spacing40'],data[name+'_spacing16']) for name in ['guide001','guide002']}
    text=json.dumps(rows,indent=2)
    if a.output:a.output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
