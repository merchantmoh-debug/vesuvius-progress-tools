"""Replay the actual upstream decimator without importing the GPU fitter."""
from pathlib import Path
import argparse
import ast
import hashlib
import json

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spiral_helpers',type=Path)
    parser.add_argument('--expect-baseline-defect',action='store_true')
    args=parser.parse_args()
    tree=ast.parse(args.spiral_helpers.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name=='_decimate_ordered_points_min_spacing']
    assert len(nodes)==1
    env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(args.spiral_helpers),'exec'),env)
    decimate=env[nodes[0].name]
    here=Path(__file__).resolve().parent
    evidence=json.loads((here/'real_curve_results.json').read_text())
    rows=[]
    for item in evidence['rows']:
        path=here/(item['guide']+'.npy')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['input_sha256']
        points=np.load(path,allow_pickle=False)
        values,indices=decimate(points,16,return_indices=True)
        assert indices[0]==0 and np.all(np.diff(indices)>0)
        assert np.array_equal(values,points[indices])
        expected=item['baseline_kept'] if args.expect_baseline_defect else item['patched_kept']
        assert len(indices)==expected
        final=item['baseline_last_index'] if args.expect_baseline_defect else len(points)-1
        assert indices[-1]==final
        rows.append({'guide':item['guide'],'kept':len(indices),'final_source_index':int(indices[-1])})
    if not args.expect_baseline_defect:
        for n in [0,1,2,6]:
            points=np.zeros((n,3),np.float32);points[:,0]=np.arange(n)
            for spacing in [0,1,4,100]:
                got,indices=decimate(points,spacing,True,force_keep={1} if n>1 else set())
                assert np.array_equal(got,points[indices])
                if n:
                    assert indices[0]==0 and indices[-1]==n-1
                    assert len(indices)==len(set(indices.tolist()))
                else:assert len(indices)==0
        points=np.zeros((8,3),np.float32);points[:,0]=np.arange(8)
        _,indices=decimate(points,4,True,force_keep={2})
        assert 2 in indices and indices[-1]==7
    print(json.dumps({'status':'BASELINE_DEFECT_REPRODUCED' if args.expect_baseline_defect else 'PASS',
                      'rows':rows,'scope':'Endpoint retention and boundary controls; no fitted-surface or ink claim.'},indent=2))


if __name__=='__main__':
    main()
