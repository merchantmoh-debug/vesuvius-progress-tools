"""FP32 DirectML runtime for the canonical 62x256x256 fine-ink ONNX contract."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load_session(graph, device_id=0):
    import onnxruntime as ort
    if 'DmlExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('Install onnxruntime-directml in a separate Windows environment')
    options=ort.SessionOptions();options.enable_mem_pattern=False
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    session=ort.InferenceSession(str(graph),sess_options=options,providers=[('DmlExecutionProvider',{'device_id':device_id})])
    if session.get_inputs()[0].name!='volume' or session.get_inputs()[0].shape!=[1,1,62,256,256]:
        raise ValueError('Graph has a different input contract')
    return session


def predict(session, raw):
    import torch
    from torch.nn.functional import interpolate
    if raw.shape!=(62,256,256) or raw.dtype!=np.uint8:raise ValueError('Expected uint8 62x256x256 input')
    volume=(np.clip(raw,0,200).astype(np.float32)/200)[None,None]
    logits=session.run(['logits'],{'volume':volume})[0]
    if logits.shape!=(1,1,64,64) or not np.isfinite(logits).all():raise ValueError('Invalid graph output')
    return interpolate(torch.from_numpy(logits).sigmoid(),size=(256,256),mode='bilinear',align_corners=False)[0,0].numpy().copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New float32 probability .npy')
    p.add_argument('--device-id',type=int,default=0)
    a=p.parse_args()
    if a.output.suffix!='.npy':raise ValueError('Output must end in .npy')
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    with a.graph.open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
    session=load_session(a.graph,a.device_id)
    result=predict(session,np.load(a.input,allow_pickle=False));np.save(a.output,result)
    a.output.with_suffix('.json').write_text(json.dumps({'graph_sha256':identity,'cpu_fallback_disabled':True,
        'normalization':'clip uint8 to [0,200], then divide by 200','output':'float32 sigmoid probabilities, bilinear upsample, align_corners=False'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
