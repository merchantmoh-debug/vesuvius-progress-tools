"""Remove only malformed all-zero ONNX Conv biases absent from verified source weights."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import tempfile
import numpy as np


def repair(model, source_tensor_names, base_dir):
    import onnx
    source_tensor_names=set(source_tensor_names)
    constants={t.name:t for t in model.graph.initializer}
    before=[t.SerializeToString() for t in model.graph.initializer]
    for node in model.graph.node:
        if node.op_type=='Constant':
            attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            if 'value' in attrs:constants[node.output[0]]=attrs['value']
    changes=[]
    for index,node in enumerate(model.graph.node):
        if node.op_type!='Conv' or len(node.input)<3 or not node.input[2]:continue
        weight=constants.get(node.input[1]);bias=constants.get(node.input[2])
        if weight is None or bias is None:raise ValueError('Unknown convolution constant')
        if list(bias.dims)==[weight.dims[0]]:continue
        if weight.data_type!=onnx.TensorProto.FLOAT or list(bias.dims)!=[weight.dims[0],2]:
            raise ValueError('Malformed bias is outside the demonstrated repair contract')
        name=node.input[1]
        if name not in source_tensor_names or not name.endswith('.weight') or name[:-7]+'.bias' in source_tensor_names:
            raise ValueError('Source does not establish a bias-free convolution')
        value=onnx.numpy_helper.to_array(bias,base_dir=str(base_dir))
        if not np.isfinite(value).all() or np.any(value!=0):raise ValueError('Refusing to remove a nonzero bias')
        changes.append({'node_index':index,'node':node.name,'weight':name,'removed_bias':node.input[2],'shape':list(bias.dims)})
    # Validation precedes all edits, so a later invalid node cannot produce a partial repair.
    for item in changes:
        del model.graph.node[item['node_index']].input[2:]
    if before!=[t.SerializeToString() for t in model.graph.initializer]:
        raise RuntimeError('Initializer data changed')
    return changes


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--source-weights',type=Path,required=True,help='Exact source safetensors file')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-repairs',type=int)
    a=p.parse_args()
    if a.output.suffix.lower()!='.onnx' or a.output.exists() or a.output.with_suffix('.json').exists() or a.output.parent.resolve()!=a.graph.parent.resolve():
        raise ValueError('Choose new graph/JSON paths in the original graph directory to preserve external-data references')
    import onnx
    with a.source_weights.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if not 0<n<=8*2**20:raise ValueError('Invalid source header size')
        header=json.loads(f.read(n))
    model=onnx.load(str(a.graph),load_external_data=False)
    external={}
    tensors=list(model.graph.initializer)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type==onnx.AttributeProto.TENSOR:tensors.append(attr.t)
    for tensor in tensors:
        for item in tensor.external_data:
            if item.key=='location':
                path=(a.graph.parent/item.value).resolve()
                if not path.is_relative_to(a.graph.parent.resolve()):raise ValueError('External tensor path leaves graph directory')
                external[str(path)]=digest(path)
    before=digest(a.graph)
    changes=repair(model,header,a.graph.parent)
    if a.expected_repairs is not None and len(changes)!=a.expected_repairs:
        raise ValueError('Repair count differs from the supplied source contract')
    with tempfile.NamedTemporaryFile(dir=a.graph.parent,suffix='.onnx',delete=False) as handle:
        temporary=Path(handle.name)
    try:
        onnx.save(model,str(temporary));onnx.checker.check_model(str(temporary))
        temporary.replace(a.output)
    finally:
        temporary.unlink(missing_ok=True)
    if digest(a.graph)!=before or any(digest(Path(k))!=v for k,v in external.items()):
        raise RuntimeError('Original graph or external parameters changed')
    a.output.with_suffix('.json').write_text(json.dumps({'changed':len(changes),'changes':changes,
        'source_graph_sha256':before,'output_sha256':digest(a.output),'external_parameters_unchanged':True,
        'scope':'Optional zero-bias graph repair; execute a separate model-output comparison before deployment.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
