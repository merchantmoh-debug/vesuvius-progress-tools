"""Controlled GroupNorm-statistic interventions on an existing evaluation-mode model."""
from __future__ import annotations
import torch


class FrozenGroupNorm:
    """Capture one baseline's statistics, replay them, and remove every hook on exit.

    Use mode='capture' for the baseline, 'ordinary' for the untouched candidate,
    and 'frozen' for the baseline/candidate intervention. Weights never change.
    """
    def __init__(self, model):
        if model.training:
            raise ValueError('Use an evaluation-mode model')
        self.layers = {name:m for name,m in model.named_modules() if isinstance(m,torch.nn.GroupNorm)}
        if not self.layers:
            raise ValueError('The model has no GroupNorm layers')
        self.mode = 'ordinary'
        self.statistics = {}
        self.handles = []

    def _hook(self, name):
        def hook(module, inputs, output):
            if self.mode == 'ordinary':
                return output
            x = inputs[0]
            n, c = x.shape[:2]
            if self.mode == 'capture':
                with torch.autocast(x.device.type, enabled=False):
                    native, mean, rstd = torch.native_group_norm(x,module.weight,module.bias,n,c,
                        x.numel()//(n*c),module.num_groups,module.eps)
                if not torch.equal(native,output):
                    raise ValueError(f'Native GroupNorm replay differs at {name}')
                self.statistics[name]=(mean.detach().clone(),rstd.detach().clone())
                return output
            if self.mode != 'frozen' or name not in self.statistics:
                raise ValueError('Capture baseline statistics before requesting frozen execution')
            mean, rstd = self.statistics[name]
            if mean.shape != (n,module.num_groups):
                raise ValueError('Batch/group layout differs from captured statistics')
            mean=mean.repeat_interleave(c//module.num_groups,dim=1)
            rstd=rstd.repeat_interleave(c//module.num_groups,dim=1)
            weight=module.weight.float() if module.weight is not None else torch.ones(c,device=x.device)
            bias=module.bias.float() if module.bias is not None else torch.zeros(c,device=x.device)
            scale=rstd*weight[None]
            shift=bias[None]-mean*scale
            shape=(n,c)+(1,)*(x.ndim-2)
            with torch.autocast(x.device.type,enabled=False):
                return torch.addcmul(shift.reshape(shape),x.float(),scale.reshape(shape)).to(output.dtype)
        return hook

    def __enter__(self):
        if self.handles:
            raise RuntimeError('This probe is already active')
        self.handles=[m.register_forward_hook(self._hook(name)) for name,m in self.layers.items()]
        return self

    def __exit__(self,*args):
        for handle in self.handles:
            handle.remove()
        self.handles=[]


def compare(model, baseline, candidate, output=lambda x:x, mask=None, replay_tolerance=1e-6):
    """Run the four matched forwards and return predictions plus a scoped error report."""
    if baseline.shape != candidate.shape:
        raise ValueError('The two input tensors must have identical shape')
    predictions={}
    with torch.inference_mode(), FrozenGroupNorm(model) as probe:
        for name,mode,value in [('baseline_original','capture',baseline),('baseline_frozen','frozen',baseline),
                                ('candidate_original','ordinary',candidate),('candidate_frozen','frozen',candidate)]:
            probe.mode=mode
            predictions[name]=output(model(value)).detach().float().clone()
    if not all(torch.isfinite(x).all() for x in predictions.values()):
        raise ValueError('Non-finite model output')
    replay=float((predictions['baseline_original']-predictions['baseline_frozen']).abs().max())
    if replay>replay_tolerance:
        raise ValueError(f'Baseline replay error {replay} exceeds {replay_tolerance}; do not interpret this intervention')
    if mask is None:
        mask=torch.ones_like(predictions['baseline_original'],dtype=torch.bool)
    if mask.dtype != torch.bool or mask.shape != predictions['baseline_original'].shape or not mask.any():
        raise ValueError('Supply a nonempty boolean output mask')
    ordinary=float((predictions['candidate_original']-predictions['baseline_original']).abs()[mask].mean())
    frozen=float((predictions['candidate_frozen']-predictions['baseline_frozen']).abs()[mask].mean())
    return predictions, {'groupnorm_layers':len(probe.layers),'baseline_max_replay_error':replay,
                        'ordinary_mean_change':ordinary,'frozen_mean_change':frozen,
                        'relative_change_removed':None if ordinary==0 else 1-frozen/ordinary,
                        'scope':'Conditional intervention; not a production model or additive causal attribution.'}
