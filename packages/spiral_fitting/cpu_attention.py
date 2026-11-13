"""Reduce CPU attention workspace by partitioning independent batch members."""
import torch
import torch.nn.functional as functional


def partitioned_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                          is_causal=False, *, scale=None, enable_gqa=False, batch_limit=32):
    """Keep every token and head together; split only independent batch items."""
    assert query.ndim == key.ndim == value.ndim == 4
    assert query.shape[0] == key.shape[0] == value.shape[0]
    assert dropout_p == 0.0, 'Dropout would change random-number assignment'
    assert query.device.type == 'cpu'
    assert batch_limit > 0
    if attn_mask is not None:
        assert attn_mask.ndim in [2,3,4]
        if attn_mask.ndim == 4:
            assert attn_mask.shape[0] in [1,query.shape[0]]
    outputs=[]
    for first in range(0,query.shape[0],batch_limit):
        last=min(first+batch_limit,query.shape[0])
        mask=attn_mask
        if mask is not None and mask.ndim==4 and mask.shape[0]>1:mask=mask[first:last]
        outputs.append(functional.scaled_dot_product_attention(
            query[first:last],key[first:last],value[first:last],attn_mask=mask,
            dropout_p=0.,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa))
    return torch.cat(outputs,dim=0)


class FunctionalProxy:
    """Change only this loaded model module's attention dispatch."""
    def __init__(self, original):
        self.original=original

    def __getattr__(self,name):
        return partitioned_attention if name=='scaled_dot_product_attention' else getattr(self.original,name)


def control():
    torch.manual_seed(792)
    results=[]
    with torch.inference_mode():
        for length,mask_rank in [(384,3),(256,4)]:
            q=torch.randn(35,6,length,32,dtype=torch.bfloat16)
            k=torch.randn_like(q);v=torch.randn_like(q)
            mask=torch.randn(6,length,length,dtype=torch.bfloat16)
            if mask_rank==4:mask=mask[None]
            expected=functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
            actual=partitioned_attention(q,k,v,attn_mask=mask)
            assert torch.equal(expected,actual), ('batch partition changed results',length,float((expected-actual).abs().max()))
            results.append({'length':length,'batch':35,'heads':6,'head_dim':32,'mask_rank':mask_rank,'bit_identical':True})
    return {'status':'PASS','cases':results,'scope':'CPU bfloat16 attention with zero dropout; independent batch partition only'}
