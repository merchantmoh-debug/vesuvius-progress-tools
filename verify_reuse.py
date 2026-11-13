"""Verify cache growth, replay, interruption and input changes on the public 96x96 example."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import hecate_directml as revised
from hecate_tile_cache import TileCache


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main(folder, graph, input_path):
    import torch
    import onnxruntime as ort
    torch.set_num_threads(2)
    folder.mkdir(parents=True,exist_ok=False)
    baseline = revised
    session = revised.load_session(graph,profile_prefix=str(folder/'dispatch'))
    raw = np.load(input_path,allow_pickle=False)
    assert raw.dtype == np.uint8 and raw.shape == (16,96,96)
    input_hash=digest(input_path)
    binding={'graph_sha256':digest(graph),'helper_sha256':digest(Path(revised.__file__)),
             'runtime':ort.__version__,'provider':'DmlExecutionProvider:0',
             'host':platform.node(),'contract':'uint8/255;16x64x64;FP32-v1'}
    expected = np.empty((96,96), np.uint8)
    start = time.perf_counter()
    base_stats = baseline.predict(session, raw, expected)
    base_seconds = time.perf_counter()-start
    first = np.empty((64,96), np.uint8)
    first_stats = revised.predict(session,raw[:,:64],first,
        tile_cache=TileCache(folder/'reuse_cache',binding,2**20))
    expanded = np.empty_like(expected)
    start = time.perf_counter()
    expanded_stats = revised.predict(session,raw,expanded,
        tile_cache=TileCache(folder/'reuse_cache',binding,2**20))
    expanded_seconds=time.perf_counter()-start
    assert expanded_stats['cache_hits']==2 and expanded_stats['model_forwards']==2
    assert np.array_equal(expected,expanded)
    replay=np.empty_like(expected)
    replay_stats=revised.predict(session,raw,replay,
        tile_cache=TileCache(folder/'reuse_cache',binding,2**20))
    assert replay_stats['cache_hits']==4 and replay_stats['model_forwards']==0
    assert np.array_equal(replay,expected)
    calls=0;predict_patch=revised.predict_patch
    def interrupt_after_one(session,patch):
        nonlocal calls
        calls+=1
        if calls==2:raise InterruptedError('Controlled interruption after one durable neural tile')
        return predict_patch(session,patch)
    revised.predict_patch=interrupt_after_one
    interrupted=False
    try:
        revised.predict(session,raw,np.empty_like(expected),
            tile_cache=TileCache(folder/'restart_cache',binding,2**20))
    except InterruptedError:interrupted=True
    finally:revised.predict_patch=predict_patch
    assert interrupted
    resumed=np.empty_like(expected)
    resume_stats=revised.predict(session,raw,resumed,
        tile_cache=TileCache(folder/'restart_cache',binding,2**20))
    assert resume_stats['cache_hits']==1 and resume_stats['model_forwards']==3
    assert np.array_equal(resumed,expected)
    changed=raw.copy();changed[0,0,0]^=np.uint8(1)
    changed_result=np.empty_like(expected)
    changed_stats=revised.predict(session,changed,changed_result,
        tile_cache=TileCache(folder/'reuse_cache',binding,2**20))
    assert changed_stats['cache_hits']==3 and changed_stats['model_forwards']==1
    changed_fresh=np.empty_like(expected)
    baseline.predict(session,changed,changed_fresh)
    assert np.array_equal(changed_fresh,changed_result)
    profile=Path(session.end_profiling())
    providers={e.get('args',{}).get('provider') for e in json.loads(profile.read_text())
               if e.get('args',{}).get('provider')}
    assert providers=={'DmlExecutionProvider'},providers
    from PIL import Image
    for name,array in [('baseline',expected),('expanded_cached',expanded),('resumed',resumed)]:
        Image.fromarray(array).save(folder/(name+'.png'))
    result={'status':'PASS','source':'PHerc0139 published real-CT crop from the awarded October example',
        'input_sha256':input_hash,'input_shape':list(raw.shape),
        'all_9216_output_pixels_exact':True,'first_crop_forwards':first_stats['model_forwards'],
        'baseline_forwards':base_stats['model_forwards'],'expanded':expanded_stats,
        'fully_cached_replay':replay_stats,'interrupted_resume':resume_stats,
        'changed_input':changed_stats,'baseline_seconds':base_seconds,'expanded_seconds':expanded_seconds,
        'providers':sorted(providers),'binding':binding,
        'scope':'Cache correctness and avoided model calls on this case. No new ink accuracy claim. Timing excludes graph/session loading; one sequential run, not a stable speed benchmark.'}
    assert digest(input_path)==input_hash
    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','all_9216_output_pixels_exact','baseline_forwards','providers']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.output,a.model,a.input)
