"""Compute a 32-cubed native direction block from original uint8 CT chunks."""
from pathlib import Path
import hashlib
import time
import numpy as np
from scipy.ndimage import gaussian_filter

def compute_block(key: tuple[int,int,int], sources: dict, reference: np.ndarray) -> tuple[np.ndarray,dict]:
    started=time.monotonic();block=np.asarray(key,dtype=np.int64)*128
    origin=block-24;shape=np.full(3,176);raw=np.empty(tuple(shape),np.float32);assigned=np.zeros(tuple(shape),bool)
    first,last=origin//128,(origin+shape-1)//128
    for delta in np.ndindex(tuple(last-first+1)):
        source_key=tuple(map(int,first+delta));assert source_key in sources,source_key
        data=np.memmap(sources[source_key],np.uint8,mode="r",shape=(128,128,128))
        start=np.asarray(source_key)*128;lo=np.maximum(origin,start);hi=np.minimum(origin+shape,start+128)
        src=tuple(slice(int(a),int(b)) for a,b in zip(lo-start,hi-start));dst=tuple(slice(int(a),int(b)) for a,b in zip(lo-origin,hi-origin))
        raw[dst]=data[src];assigned[dst]=True;del data
    assert assigned.all();del assigned
    smooth=gaussian_filter(raw,1.,truncate=4.);del raw
    gradients=np.gradient(smooth);del smooth
    tensor=np.empty((32,32,32,3,3),np.float32);sample=(slice(24,152,4),)*3
    for a in range(3):
        for b in range(a,3):
            product=gradients[a]*gradients[b]
            smooth=gaussian_filter(product,8.,truncate=2.)
            tensor[...,a,b]=smooth[sample];tensor[...,b,a]=tensor[...,a,b]
            del product,smooth
    del gradients
    values,vectors=np.linalg.eigh(tensor);del tensor
    normal=vectors[...,:,2][...,::-1].astype(np.float64);del vectors
    ref=np.asarray(reference,float);ref=ref/np.linalg.norm(ref)
    normal*=np.where(normal@ref<0,-1.,1.)[...,None]
    assert np.isfinite(normal).all() and np.isfinite(values).all()
    valid=values[...,2]>0
    encoded=np.clip(np.rint(128+127*normal),1,255).astype(np.uint8)
    encoded[~valid]=128
    assert not np.any(np.all(encoded[valid]==128,axis=-1))
    hashes={a:hashlib.sha256(np.ascontiguousarray(encoded[...,i]).tobytes()).hexdigest() for i,a in enumerate("xyz")}
    return encoded,{"key_zyx":list(key),"seconds":time.monotonic()-started,"valid_nodes":int(valid.sum()),
                    "zero_tensor_nodes":int((~valid).sum()),"component_sha256":hashes}
