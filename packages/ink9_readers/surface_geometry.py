"""Numerical surface sampling, exact triangle-distance checks and safe grid comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares


def load_grid(path):
    path=Path(path)
    if path.is_dir():
        import tifffile
        grid=np.stack([tifffile.imread(path/(a+'.tif')) for a in 'xyz'],-1)
    else:
        grid=np.load(path,allow_pickle=False)
    if grid.ndim!=3 or grid.shape[2]!=3:
        raise ValueError('Expected an HxWx3 XYZ grid or a TIFXYZ directory')
    return np.asarray(grid,dtype=np.float64)


def bilinear(grid, uv):
    """uv is (...,2) in column,row order. No extrapolation or hidden axis swap."""
    uv=np.asarray(uv,dtype=float)
    if uv.shape[-1]!=2 or not np.isfinite(uv).all():raise ValueError('Invalid UV coordinates')
    upper=np.array([grid.shape[1]-1,grid.shape[0]-1],float)
    if np.any(uv<0) or np.any(uv>upper):raise ValueError('Point leaves grid')
    ij=np.minimum(np.floor(uv).astype(int),upper.astype(int)-1)
    u,v=(uv-ij)[...,0,None],(uv-ij)[...,1,None]
    x,y=ij[...,0],ij[...,1]
    a,b,c,d=grid[y,x],grid[y,x+1],grid[y+1,x],grid[y+1,x+1]
    if not np.isfinite(np.stack([a,b,c,d])).all():raise ValueError('Unsupported grid cell')
    return (1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*c+u*v*d


def triangles(grid, valid=None):
    """Use the native ABF quad diagonal p10-p01; only complete valid cells survive."""
    g=np.asarray(grid,dtype=np.float64)
    valid=np.isfinite(g).all(-1) if valid is None else np.asarray(valid,dtype=bool)
    if valid.shape!=g.shape[:2]:raise ValueError('Validity mask shape differs')
    cells=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    a,b,c,d=g[:-1,:-1][cells],g[:-1,1:][cells],g[1:,:-1][cells],g[1:,1:][cells]
    out=np.concatenate([np.stack([a,b,c],1),np.stack([b,d,c],1)])
    area=np.linalg.norm(np.cross(out[:,1]-out[:,0],out[:,2]-out[:,0]),axis=1)
    if not len(out) or np.any(area<=0):raise ValueError('No surface or degenerate source triangles')
    return out


def exact_surface_distances(points, source_triangles, batch_size=4096, max_pairs=4000000):
    """Complete radius-bounded candidate search, not a fixed-K nearest-centroid estimate."""
    import trimesh
    p=np.asarray(points,dtype=float);t=np.asarray(source_triangles,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or t.ndim!=3 or t.shape[1:]!=(3,3) or not len(t):
        raise ValueError('Expected Nx3 points and Mx3x3 triangles')
    if not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('Non-finite geometry')
    if batch_size<1 or max_pairs<1:raise ValueError('Invalid work budget')
    centers=t.mean(1);radius=np.linalg.norm(t-centers[:,None],axis=-1).max(1);tree=cKDTree(centers)
    result=np.empty(len(p))
    for begin in range(0,len(p),batch_size):
        block=p[begin:begin+batch_size]
        _,first=tree.query(block)
        upper=np.linalg.norm(trimesh.triangles.closest_point(t[first],block)-block,axis=1)
        candidates=tree.query_ball_point(block,upper+radius.max()+1e-7)
        counts=np.array([len(ids) for ids in candidates])
        if counts.sum()>max_pairs:raise MemoryError('Triangle candidate budget exceeded; reduce batch_size')
        pi=np.repeat(np.arange(len(block)),counts);ti=np.concatenate(candidates)
        near=trimesh.triangles.closest_point(t[ti],block[pi])
        distances=np.linalg.norm(near-block[pi],axis=1)
        best=np.full(len(block),np.inf);np.minimum.at(best,pi,distances)
        result[begin:begin+len(block)]=best
    return result


def grid_pitch(grid, voxel_um, valid=None):
    if not np.isfinite(voxel_um) or voxel_um<=0:raise ValueError('Positive voxel spacing required')
    g=np.asarray(grid,dtype=float)
    valid=np.isfinite(g).all(-1) if valid is None else np.asarray(valid,dtype=bool)
    dx=np.diff(g,axis=1);dy=np.diff(g,axis=0)
    goodx=valid[:,:-1]&valid[:,1:];goody=valid[:-1]&valid[1:]
    cells=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]
    if not goodx.any() or not goody.any() or not cells.any():raise ValueError('Insufficient supported geometry')
    area=np.linalg.norm(np.cross(dx[:-1],dy[:,:-1]),axis=-1)[cells]*voxel_um**2
    q=[.01,.5,.99]
    return {'quantiles':q,'column_pitch_um':np.quantile(np.linalg.norm(dx[goodx],axis=-1)*voxel_um,q).tolist(),
            'row_pitch_um':np.quantile(np.linalg.norm(dy[goody],axis=-1)*voxel_um,q).tolist(),
            'area_equivalent_pitch_um':np.quantile(np.sqrt(area),q).tolist()}


def parent_motion(parent, child, offset_xy, parent_valid=None):
    """The child's immediate offset maps old (column,row) to new (column,row)."""
    offset=np.asarray(offset_xy)
    if offset.shape!=(2,) or not np.issubdtype(offset.dtype,np.integer):raise ValueError('Integer XY offset required')
    valid=np.isfinite(parent).all(-1) if parent_valid is None else np.asarray(parent_valid,bool)
    y,x=np.nonzero(valid);xx=x+offset[0];yy=y+offset[1]
    if np.any(xx<0) or np.any(yy<0) or np.any(xx>=child.shape[1]) or np.any(yy>=child.shape[0]):
        raise ValueError('Parent correspondence leaves child grid')
    target=child[yy,xx]
    if not np.isfinite(target).all():raise ValueError('Parent points map to unsupported child values')
    return np.linalg.norm(target-parent[y,x],axis=1)


def trim_invalid_border(channels, valid):
    """Trim outer padding only; refuse genuine interior holes and crop all channels alike."""
    valid=np.asarray(valid,dtype=bool)
    if valid.ndim!=2 or not valid.any():raise ValueError('Nonempty 2D validity required')
    y,x=np.nonzero(valid);bounds=(slice(y.min(),y.max()+1),slice(x.min(),x.max()+1))
    if not valid[bounds].all():raise ValueError('Interior holes are not removable outer padding')
    if any(np.asarray(a).shape[:2]!=valid.shape for a in channels.values()):raise ValueError('Unaligned channel shape')
    return {k:np.asarray(a)[bounds].copy() for k,a in channels.items()},[int(x.min()),int(y.min())]


def world_to_grid(grid, point, max_distance, initial_uv=None):
    """Local bilinear inverse; caller supplies the intended sheet. Residual is always returned."""
    point=np.asarray(point,dtype=float)
    if not np.isfinite(max_distance) or max_distance<0:raise ValueError('Finite nonnegative distance limit required')
    if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('Finite XYZ point required')
    if initial_uv is None:
        good=np.isfinite(grid).all(-1);yx=np.argwhere(good)
        if not len(yx):raise ValueError('Empty surface')
        _,index=cKDTree(grid[good]).query(point);initial_uv=yx[index,::-1].astype(float)
    upper=np.array([grid.shape[1]-1,grid.shape[0]-1],float)
    initial_uv=np.clip(initial_uv,0,upper)
    fit=least_squares(lambda uv:bilinear(grid,uv)-point,initial_uv,bounds=(np.zeros(2),upper),
                      xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100)
    residual=float(np.linalg.norm(bilinear(grid,fit.x)-point))
    if not fit.success or residual>max_distance:raise ValueError(f'No acceptable local mapping: residual={residual}')
    return fit.x,residual


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--voxel-um',type=float,required=True)
    p.add_argument('--points',type=Path,help='Optional Nx3 native XYZ query points')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    grid=load_grid(a.mesh);valid=np.isfinite(grid).all(-1)&(grid>0).all(-1)
    result={'pitch':grid_pitch(grid,a.voxel_um,valid),'validity':'TIFXYZ positive XYZ convention'}
    if a.points:
        d=exact_surface_distances(np.load(a.points,allow_pickle=False),triangles(grid,valid))
        result['distances']={'count':len(d),'quantiles':[0,.5,.9,.99,1],
                             'native_voxels':np.quantile(d,[0,.5,.9,.99,1]).tolist(),
                             'above_one_voxel':int((d>1).sum())}
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
