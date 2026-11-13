"""Orient a complete produced normal field and leave conflicting cells unset."""
import numpy as np
from numba import njit

@njit(cache=True)
def transport(encoded, shape, seed, reference):
    count=encoded.shape[0];signs=np.zeros(count,np.int8);queue=np.empty(count,np.int32)
    ny,nx=shape[1],shape[2];plane=ny*nx;components=0;visited=0
    for turn in range(count+1):
        start=seed if turn==0 else turn-1
        if signs[start]!=0:continue
        if encoded[start,0]==128 and encoded[start,1]==128 and encoded[start,2]==128:continue
        dot=0.
        for axis in range(3):dot+=(int(encoded[start,axis])-128)*reference[axis]
        signs[start]=1 if dot>=0 else -1;queue[0]=start;head=0;tail=1;components+=1
        while head<tail:
            current=queue[head];head+=1;visited+=1
            z=current//plane;y=(current//nx)%ny;x=current%nx
            for axis in range(3):
                for step in (-1,1):
                    if axis==0:
                        if z+step<0 or z+step>=shape[0]:continue
                        other=current+step*plane
                    elif axis==1:
                        if y+step<0 or y+step>=ny:continue
                        other=current+step*nx
                    else:
                        if x+step<0 or x+step>=nx:continue
                        other=current+step
                    if signs[other]!=0:continue
                    if encoded[other,0]==128 and encoded[other,1]==128 and encoded[other,2]==128:continue
                    dot=0
                    for component in range(3):dot+=(int(encoded[current,component])-128)*(int(encoded[other,component])-128)
                    signs[other]=signs[current] if dot>=0 else -signs[current]
                    queue[tail]=other;tail+=1
    return signs,components,visited

def align(encoded: np.ndarray, seed: tuple, reference: np.ndarray) -> tuple[np.ndarray,dict]:
    shape=encoded.shape[:3];flat=encoded.reshape(-1,3)
    seed_index=int(np.ravel_multi_index(seed,shape));assert not np.all(flat[seed_index]==128)
    signs,components,visited=transport(flat,np.array(shape),seed_index,np.asarray(reference,float))
    result=encoded.copy();flip=signs.reshape(shape)<0
    result[flip]=(256-result[flip].astype(np.int16)).astype(np.uint8)
    raw=(encoded.astype(np.int16)-128);changed=(result.astype(np.int16)-128)
    assert np.all(np.all(raw==changed,axis=-1)|np.all(raw==-changed,axis=-1))
    del raw
    valid=np.any(result!=128,axis=-1);reject=np.zeros(shape,bool);conflicts=[]
    for axis in range(3):
        a=[slice(None)]*3;b=[slice(None)]*3;a[axis]=slice(None,-1);b[axis]=slice(1,None);a=tuple(a);b=tuple(b)
        dot=np.einsum('...i,...i->...',changed[a].astype(np.int32),changed[b].astype(np.int32))
        bad=valid[a]&valid[b]&(dot<0);conflicts.append(int(bad.sum()));reject[a]|=bad;reject[b]|=bad
    excluded=int(reject.sum());result[reject]=128;valid&=~reject
    changed[reject]=0
    for axis in range(3):
        a=[slice(None)]*3;b=[slice(None)]*3;a[axis]=slice(None,-1);b[axis]=slice(1,None);a=tuple(a);b=tuple(b)
        dot=np.einsum('...i,...i->...',changed[a].astype(np.int32),changed[b].astype(np.int32))
        assert not np.any(valid[a]&valid[b]&(dot<0))
    return result,{"components":int(components),"visited_valid_nodes":int(visited),"flipped_nodes":int(flip.sum()),
                   "conflicting_axis_edges_before_exclusion":conflicts,"excluded_conflicting_nodes":excluded,
                   "valid_nodes_after":int(valid.sum()),"remaining_axis_sign_conflicts":0}

def controls() -> dict:
    base=np.array([165,177,239],np.uint8);a=np.broadcast_to(base,(3,3,3,3)).copy();a[::2]=(256-a[::2].astype(np.int16)).astype(np.uint8)
    result,info=align(a,(1,1,1),base.astype(float)-128);assert np.all(result==base) and info['excluded_conflicting_nodes']==0
    a[0,0,0]=128;result,info=align(a,(1,1,1),base.astype(float)-128);assert np.all(result[0,0,0]==128) and info['visited_valid_nodes']==26
    theta=np.deg2rad(np.array([[0.,45.],[135.,90.]]));v=np.stack([np.cos(theta),np.sin(theta),np.zeros_like(theta)],-1)[None]
    bad=np.clip(np.rint(128+127*v),1,255).astype(np.uint8);result,info=align(bad,(0,0,0),np.array([1.,0.,0.]))
    assert sum(info['conflicting_axis_edges_before_exclusion'])>0 and info['excluded_conflicting_nodes']>0
    return {"alternating_signs":"PASS","neutral_hole":"PASS","inconsistent_cycle_is_exposed":"PASS"}
