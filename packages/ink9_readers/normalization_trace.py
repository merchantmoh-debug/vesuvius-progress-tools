"""Trace an identical physical point through convolution and per-crop InstanceNorm."""
import torch


def point_trace(model, conv_name, norm_name, tensor, point_zyx, output=lambda value:value):
    modules=dict(model.named_modules())
    conv,norm=modules[conv_name],modules[norm_name]
    if not isinstance(norm,torch.nn.InstanceNorm3d) or norm.track_running_stats:
        raise ValueError('This diagnostic requires per-input InstanceNorm3d statistics')
    if model.training or tensor.ndim!=5 or tensor.shape[0]!=1:
        raise ValueError('Use an eval model and one NCDHW input')
    point=tuple(map(int,point_zyx));captured={}
    def conv_hook(module,args,value):
        if any(v<0 or v>=n for v,n in zip(point,value.shape[2:])):raise ValueError('Point leaves features')
        captured['convolution']=value[(0,slice(None),*point)].clone()
        captured['mean']=value.mean(dim=(2,3,4))[0].clone()
        captured['variance']=value.var(dim=(2,3,4),unbiased=False)[0].clone()
    def norm_hook(module,args,value):
        captured['normalized']=value[(0,slice(None),*point)].clone()
    handles=[conv.register_forward_hook(conv_hook),norm.register_forward_hook(norm_hook)]
    try:
        with torch.inference_mode():
            prediction=output(model(tensor)).detach().clone()
    finally:
        for handle in handles:handle.remove()
    value=(captured['convolution']-captured['mean'])/torch.sqrt(captured['variance']+norm.eps)
    if norm.affine:value=value*norm.weight+norm.bias
    error=float((value-captured['normalized']).detach().abs().max())
    return prediction, {**{k:v.detach().cpu().tolist() for k,v in captured.items()},
                        'analytic_instance_norm_max_error':error}
