"""Stream exact F32/I64 weights to safetensors and retain read-only mmap tensor views."""
from __future__ import annotations
import argparse
import json
import math
import mmap
from pathlib import Path
import struct


def write_state(state, output):
    import torch
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    header={};offset=0
    for name,tensor in state.items():
        if not isinstance(name,str) or not isinstance(tensor,torch.Tensor):
            raise ValueError('State must map names to tensors')
        dtype={torch.float32:'F32',torch.int64:'I64'}.get(tensor.dtype)
        if dtype is None:
            raise ValueError('This verified loader supports F32/I64 weights only')
        size=tensor.numel()*tensor.element_size()
        header[name]={'dtype':dtype,'shape':list(tensor.shape),'data_offsets':[offset,offset+size]}
        offset+=size
    encoded=json.dumps(header,separators=(',',':')).encode('utf-8')
    encoded+=b' '*((-len(encoded))%8)
    if len(encoded)>8*2**20:
        raise ValueError('Header exceeds 8 MiB')
    with output.open('xb') as stream:
        stream.write(struct.pack('<Q',len(encoded)));stream.write(encoded)
        for tensor in state.values():
            array=tensor.detach().cpu().contiguous().numpy()
            stream.write(memoryview(array).cast('B'))
    return {'tensors':len(header),'tensor_bytes':offset,'header_bytes':len(encoded)}


class ReadOnlyState:
    """Keep this object alive while any tensor/model references its mapping. Never train it."""
    def __init__(self, path):
        import torch
        self.handle=Path(path).open('rb')
        self.mapping=mmap.mmap(self.handle.fileno(),0,access=mmap.ACCESS_READ)
        self.tensors={}
        try:
            if len(self.mapping)<8:
                raise ValueError('Truncated safetensors file')
            length=struct.unpack('<Q',self.mapping[:8])[0]
            if not 0<length<=8*2**20 or 8+length>len(self.mapping):
                raise ValueError('Invalid safetensors header size')
            metadata=json.loads(self.mapping[8:8+length]);base=8+length
            spans=[]
            for name,entry in metadata.items():
                if name=='__metadata__':continue
                dtype={'F32':torch.float32,'I64':torch.int64}.get(entry['dtype'])
                if dtype is None:raise ValueError('Unsupported dtype')
                shape=entry['shape']
                if any(type(v) is not int or v<0 for v in shape):raise ValueError('Invalid tensor shape')
                count=math.prod(shape);start,end=entry['data_offsets'];size=4 if dtype==torch.float32 else 8
                if (type(start) is not int or type(end) is not int or start<0 or end<start
                        or end-start!=count*size or base+end>len(self.mapping)):
                    raise ValueError('Invalid tensor byte range')
                spans.append((start,end))
            cursor=0
            for start,end in sorted(spans):
                if start!=cursor:raise ValueError('Overlapping or unaccounted tensor bytes')
                cursor=end
            if base+cursor!=len(self.mapping):raise ValueError('Unaccounted file tail')
            for name,entry in metadata.items():
                if name=='__metadata__':continue
                dtype={'F32':torch.float32,'I64':torch.int64}[entry['dtype']]
                count=math.prod(entry['shape']);start,_=entry['data_offsets']
                self.tensors[name]=(torch.frombuffer(self.mapping,dtype=dtype,count=count,offset=base+start)
                                    .reshape(entry['shape']) if count else torch.empty(entry['shape'],dtype=dtype))
        except Exception:
            self.tensors.clear();self.mapping.close();self.handle.close();raise

    def load_into(self, model):
        model.requires_grad_(False)
        model.load_state_dict(self.tensors,strict=True,assign=True)
        model.eval()
        model._readonly_weight_backing=self
        return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state-key',default='state_dict',help='Top-level checkpoint key; empty string for root tensor dictionary')
    p.add_argument('--strip-prefix',default='')
    a=p.parse_args()
    import torch
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    state=payload[a.state_key] if a.state_key else payload
    if a.strip_prefix:
        if not all(k.startswith(a.strip_prefix) for k in state):raise ValueError('Prefix does not match every tensor')
        state={k[len(a.strip_prefix):]:v for k,v in state.items()}
    print(json.dumps(write_state(state,a.output)))


if __name__=='__main__':main()
