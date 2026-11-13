"""Map physical reader-window centres through a changed TIFXYZ chart, without inspecting ink scores."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def map_centres(original_grid, new_grid, centres_xy, original_scale_xy, new_scale_xy, window_size, max_distance):
    from surface_geometry import bilinear,world_to_grid
    old_scale=np.asarray(original_scale_xy,float);new_scale=np.asarray(new_scale_xy,float)
    if old_scale.shape!=(2,) or new_scale.shape!=(2,) or np.any(old_scale<=0) or np.any(new_scale<=0):
        raise ValueError('Positive per-axis TIFXYZ sampling scales required')
    if window_size<1:raise ValueError('Positive window size required')
    rows=[]
    for xy in np.asarray(centres_xy,float):
        world=bilinear(original_grid,(xy+.5)*old_scale)
        uv,error=world_to_grid(new_grid,world,max_distance)
        mapped=uv/new_scale-.5
        corner=np.rint(mapped-(window_size-1)/2).astype(int)
        if np.any(corner<0):raise ValueError('Mapped context needs surrounding source material outside this chart')
        rows.append({'original_pixel_center_xy':xy.tolist(),'world_xyz':world.tolist(),
                     'mapped_pixel_center_xy':mapped.tolist(),'crop_xywh':[int(corner[0]),int(corner[1]),window_size,window_size],
                     'source_mapping_distance_voxels':error})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface-tools',type=Path,default=Path(__file__).resolve().parent,help='Directory containing surface_geometry.py (bundled by default)')
    p.add_argument('--original-mesh',type=Path,required=True)
    p.add_argument('--new-mesh',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True,help='JSON with centres_xy, original_scale_xy, new_scale_xy, window_size, max_distance')
    p.add_argument('--volume',type=Path,help='Optional new-chart uint8 ZYX .npy; save each mapped context beside the JSON')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(a.surface_tools.resolve()))
    from surface_geometry import load_grid
    rows=map_centres(load_grid(a.original_mesh),load_grid(a.new_mesh),**json.loads(a.mapping.read_text(encoding='utf-8')))
    if a.volume:
        volume=np.load(a.volume,mmap_mode='r',allow_pickle=False)
        if volume.ndim!=3 or volume.dtype!=np.uint8:raise ValueError('Expected uint8 ZYX surface volume')
        destinations=[]
        for i,row in enumerate(rows):
            x,y,w,h=row['crop_xywh'];dest=a.output.with_name(a.output.stem+f'_context_{i+1:03d}.npy')
            if x+w>volume.shape[2] or y+h>volume.shape[1]:raise ValueError('A mapped context leaves the supplied volume')
            if dest.exists():raise FileExistsError(dest)
            destinations.append(dest)
        for row,dest in zip(rows,destinations):
            x,y,w,h=row['crop_xywh'];np.save(dest,np.asarray(volume[:,y:y+h,x:x+w]));row['context_file']=dest.name
    a.output.write_text(json.dumps({'contexts':rows,'selection':'Geometry only; no model score search'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
