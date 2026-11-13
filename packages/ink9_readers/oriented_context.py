"""Rotate a fixed input window and restore output orientation before blending."""

def orient_forward(forward,turns,torch):
    def wrapped(value,*args,**kwargs):
        result=forward(torch.rot90(value,turns,(-2,-1)).contiguous(),*args,**kwargs)
        result=dict(result)
        ink=result['ink']
        if isinstance(ink,(tuple,list)):
            result['ink']=type(ink)(torch.rot90(x,-turns,(-2,-1)) for x in ink)
        else:
            result['ink']=torch.rot90(ink,-turns,(-2,-1))
        return result
    return wrapped
