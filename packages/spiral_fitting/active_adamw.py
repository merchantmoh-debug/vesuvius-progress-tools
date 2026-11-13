"""Fresh AdamW with state only for coordinates ever receiving nonzero gradients.

Exact inactive-coordinate preservation requires zero weight decay. Every active
coordinate remains in the state even when its current gradient is zero, so its
momentum keeps evolving. The parameter-wide step counter also advances through
empty steps. This is not a loader for arbitrary existing dense optimizer history.
"""
import torch
from torch.optim import _functional


class ActiveAdamW:
    def __init__(self, *, lr: float, betas: tuple[float,float]=(.9,.999), eps: float=1e-8, weight_decay: float=0., block_size: int=262144):
        if weight_decay!=0:raise ValueError('ActiveAdamW requires zero weight decay')
        if block_size<=0 or block_size%256:raise ValueError('Blocks must preserve CPU vector-kernel alignment')
        self.lr,self.betas,self.eps=float(lr),betas,float(eps)
        self.block_size=block_size
        self.indices=torch.empty(0,dtype=torch.int64)
        self.exp_avg=torch.empty(0,dtype=torch.float32)
        self.exp_avg_sq=torch.empty(0,dtype=torch.float32)
        self.step_count=torch.tensor(0.,dtype=torch.float32)

    @torch.no_grad()
    def step(self, parameter: torch.Tensor, gradient: torch.Tensor) -> None:
        assert parameter.device.type=='cpu' and gradient.device.type=='cpu'
        assert parameter.dtype==gradient.dtype==torch.float32
        assert parameter.shape==gradient.shape and parameter.is_contiguous() and gradient.is_contiguous()
        flat=parameter.view(-1);grad=gradient.view(-1)
        # Packing coordinates changes the scalar/vector arithmetic path and can
        # change moment bits. Fixed aligned blocks retain each original address's
        # vector/tail position. One CPU thread removes parallel partition tails.
        threads=torch.get_num_threads();torch.set_num_threads(1)
        ids_out=[];means=[];variances=[]
        try:
            active_all=torch.unique(torch.cat([self.indices,torch.nonzero(grad,as_tuple=True)[0]]),sorted=True)
            block_ids=torch.div(active_all,self.block_size,rounding_mode='floor').unique_consecutive()
            for block_id in block_ids.tolist():
                start=block_id*self.block_size
                end=min(start+self.block_size,len(flat));block_grad=grad[start:end]
                low=int(torch.searchsorted(self.indices,start));high=int(torch.searchsorted(self.indices,end))
                prior=self.indices[low:high]-start
                begin=int(torch.searchsorted(active_all,start));finish=int(torch.searchsorted(active_all,end))
                active=active_all[begin:finish]-start
                mean=torch.zeros_like(block_grad);variance=torch.zeros_like(block_grad)
                mean[prior]=self.exp_avg[low:high];variance[prior]=self.exp_avg_sq[low:high]
                # Preserve physical lane positions while avoiding writes to the
                # mapped fine-field pages that have no active coordinates.
                values=torch.zeros_like(block_grad);values[active]=flat[active+start]
                step=self.step_count.clone()
                _functional.adamw([values],[block_grad],[mean],[variance],[],[step],
                    foreach=False,capturable=False,differentiable=False,fused=False,grad_scale=None,found_inf=None,
                    has_complex=False,amsgrad=False,beta1=self.betas[0],beta2=self.betas[1],lr=self.lr,
                    weight_decay=0.,eps=self.eps,maximize=False)
                assert float(step)==float(self.step_count)+1
                flat.index_copy_(0,active+start,values[active])
                ids_out.append(active+start);means.append(mean[active]);variances.append(variance[active])
            self.indices=torch.cat(ids_out) if ids_out else torch.empty(0,dtype=torch.int64)
            self.exp_avg=torch.cat(means) if means else torch.empty(0,dtype=torch.float32)
            self.exp_avg_sq=torch.cat(variances) if variances else torch.empty(0,dtype=torch.float32)
            self.step_count.add_(1)
        finally:torch.set_num_threads(threads)

    def snapshot(self, parameter: torch.Tensor) -> dict[str,torch.Tensor]:
        return {'indices':self.indices,'values':parameter.detach().view(-1)[self.indices],
                'exp_avg':self.exp_avg,'exp_avg_sq':self.exp_avg_sq,'step':self.step_count}


def parity_control() -> dict:
    """Exercise delayed activation, zero steps, sign reversals and old momentum."""
    assert torch.__version__=='2.10.0+cpu'
    torch.manual_seed(4139)
    reference=torch.nn.Parameter(torch.randn(2,3,7,11))
    actual=torch.nn.Parameter(reference.detach().clone())
    dense=torch.optim.AdamW([reference],lr=3e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    sparse=ActiveAdamW(lr=3e-5,block_size=256)
    schedules=[[],[0,11,200],[11],[300,401],[],[0,401],[13,211,350],[],[11,211,350],[]]
    active=set();per_step=[]
    for number,indices in enumerate(schedules,1):
        gradient=torch.zeros_like(reference)
        if indices:
            values=torch.randn(len(indices))*(1e-6 if number==6 else 1.)
            gradient.view(-1)[indices]=values
        reference.grad=gradient.clone();dense.step()
        sparse.step(actual,gradient)
        active.update(indices)
        assert actual.detach().numpy().tobytes()==reference.detach().numpy().tobytes(),f'Parameter mismatch at step{number}'
        state=dense.state[reference]
        assert sparse.exp_avg.numpy().tobytes()==state['exp_avg'].view(-1)[sparse.indices].numpy().tobytes()
        assert sparse.exp_avg_sq.numpy().tobytes()==state['exp_avg_sq'].view(-1)[sparse.indices].numpy().tobytes()
        assert torch.equal(sparse.step_count,state['step'])
        assert set(sparse.indices.tolist())==active
        inactive=torch.ones(reference.numel(),dtype=torch.bool);inactive[sparse.indices]=False
        assert torch.count_nonzero(state['exp_avg'].view(-1)[inactive])==0
        assert torch.count_nonzero(state['exp_avg_sq'].view(-1)[inactive])==0
        per_step.append({'step':number,'active_coordinates':len(active),'parameter_exact':True,'moments_exact':True})
    rejected=False
    try:ActiveAdamW(lr=3e-5,weight_decay=.01)
    except ValueError:rejected=True
    assert rejected
    return {'status':'PASS','torch':torch.__version__,'steps':per_step,'weight_decay_guard':'PASS',
            'scope':'Bit-exact parity with standard CPU AdamW on this fixed multistep control. Fresh state and zero weight decay are required.'}
