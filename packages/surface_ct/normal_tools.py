"""Command-line wrappers for bounded CT direction construction and sign alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('stream')
    a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--gradient-sigma',type=float,default=.8);a.add_argument('--tensor-sigma',type=float,default=2.)
    a.add_argument('--slab-depth',type=int,default=32)
    a=sub.add_parser('block')
    a.add_argument('--chunks',type=Path,required=True,help='JSON list of key_zyx, path, sha256; 128-cubed uint8 chunks')
    a.add_argument('--key-zyx',type=int,nargs=3,required=True)
    a.add_argument('--reference-xyz',type=float,nargs=3,required=True)
    a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('align')
    a.add_argument('--input',type=Path,required=True,help='Complete produced uint8 ZYX3 normal field')
    a.add_argument('--seed-zyx',type=int,nargs=3,required=True)
    a.add_argument('--reference-xyz',type=float,nargs=3,required=True)
    a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.action!='stream' and args.output.suffix!='.npy':raise ValueError('Array output must end in .npy')
    if args.output.exists():raise FileExistsError(args.output)
    if args.action=='stream':
        from streamed_directions import build
        volume=np.load(args.input,mmap_mode='r',allow_pickle=False)
        build(volume,args.output,args.gradient_sigma,args.tensor_sigma,args.slab_depth)
    elif args.action=='block':
        from ct_normal_blocks import compute_block
        sources={}
        for item in json.loads(args.chunks.read_text(encoding='utf-8')):
            path=Path(item['path'])
            if not path.is_absolute():path=args.chunks.parent/path
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if path.stat().st_size!=128**3 or actual!=item['sha256']:raise ValueError('Chunk identity/size mismatch')
            key=tuple(item['key_zyx'])
            if key in sources:raise ValueError('Duplicate chunk key')
            sources[key]=path
        ref=np.asarray(args.reference_xyz,float)
        if not np.isfinite(ref).all() or np.linalg.norm(ref)==0:raise ValueError('Nonzero finite reference required')
        encoded,report=compute_block(tuple(args.key_zyx),sources,ref)
        np.save(args.output,encoded)
        args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    else:
        from normal_alignment import align
        encoded=np.load(args.input,allow_pickle=False)
        ref=np.asarray(args.reference_xyz,float)
        if (encoded.dtype!=np.uint8 or encoded.ndim!=4 or encoded.shape[-1]!=3
                or not np.isfinite(ref).all() or np.linalg.norm(ref)==0):raise ValueError('Invalid field/reference')
        aligned,report=align(encoded,tuple(args.seed_zyx),ref)
        np.save(args.output,aligned)
        args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
