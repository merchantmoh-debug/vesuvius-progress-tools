"""Execute the released winding network with a spatially tiled pointwise head."""
import torch
import torch.nn.functional as F


def tiled_detail_head(model,features,skip,rows=8):
    """Preserve interpolation halos; every learned head operation is pointwise."""
    assert features.shape[2]*2==skip.shape[2] and features.shape[3]*2==skip.shape[3]
    assert features.shape[4]==skip.shape[4] and rows>0 and rows%2==0
    for layer in model.full_resolution_head:
        if isinstance(layer,torch.nn.Conv3d):
            assert layer.kernel_size==(1,1,1) and layer.stride==(1,1,1) and layer.padding==(0,0,0)
        else:assert type(layer).__name__ in ['ChannelLayerNorm','GELU'],type(layer).__name__
    outputs=[]
    for first in range(0,skip.shape[2],rows):
        last=min(first+rows,skip.shape[2]);lower=max(0,first//2-1);upper=min(features.shape[2],(last+1)//2+1)
        fine=F.interpolate(features[:,:,lower:upper],scale_factor=(2,2,1),mode='trilinear',align_corners=False)
        fine=fine[:,:,first-2*lower:last-2*lower]
        outputs.append(model.full_resolution_head(torch.cat([fine,skip[:,:,first:last]],dim=1)))
    return torch.cat(outputs,dim=2)


def memory_bounded_forward(model,slab_image,slab_valid):
    """Same normalization, encoder, attention, decoder, and phase accumulation."""
    assert model.full_resolution_head is not None and model.log_variance_head is None and model.crossing_head is None
    assert len(model.extra_decoders)==0 and len(model.stages)==3
    image=slab_image.float();valid=slab_valid.float()
    mass=valid.sum(dim=(1,2,3)).clamp_min(1.0)
    mean=(image*valid).sum(dim=(1,2,3))/mass
    centered=(image-mean[:,None,None,None])*valid
    std=(centered.square().sum(dim=(1,2,3))/mass).sqrt()
    image=centered/(std[:,None,None,None]+1e-6)
    features=torch.stack([image,valid],dim=1)
    del image,valid,centered,mass,mean,std
    skips=[]
    for stage in model.stages:
        features=stage(features);skips.append(features)
    features=model.trunk_in(features)
    for block in model.stem_blocks:features=block(features)
    tokens=features.permute(0,2,3,4,1)
    del features
    for block in model.attention_blocks:tokens=block(tokens)
    features=tokens.permute(0,4,1,2,3);del tokens
    features=F.interpolate(features,scale_factor=(2,2,1),mode='trilinear',align_corners=False)
    features=model.decoder(torch.cat([features,skips[-2]],dim=1))
    first_skip=skips[0];del skips
    logits=model._pixel_shuffle_output(tiled_detail_head(model,features,first_skip),2)
    increments=F.softplus(logits.float())
    return {'phase':increments.cumsum(dim=-1),'phase_increments':increments}
