import torch.nn.functional as F
import torch
import torch.nn as nn
from .utils import *




def non_local_correlation(tar, refs, per_ref=True, flatten=True, temprature=1.0, mask=None, scaling=False, norm=False, att_only=False):
    
    """ Given refs and tar, return transform tar non-local.

    Returns:
        att: attention for tar wrt each ref (concat) 
        out: transform tar for each ref if per_ref else for all refs
    """
    
    if isinstance(refs, list):
        refs = torch.stack(refs, 1)

    tar = tar.flatten(2).permute(0, 2, 1)
    _, t, feat_dim, w_, h_ = refs.shape
    refs = refs.flatten(3).permute(0, 1, 3, 2)
    if norm:
        tar = F.normalize(tar, dim=-1)
        refs = F.normalize(refs, dim=-1)
        
    # calc correlation
    corr = torch.einsum("bic,btjc -> btij", (tar, refs)) / temprature 
    
    if att_only: return corr
    
    if scaling:
        # scaling
        corr = corr / torch.sqrt(torch.tensor(feat_dim).float()) 

    if mask is not None:
        # att *= mask
        corr.masked_fill_(~mask.bool(), float('-inf'))
    
    if per_ref:
        # return att for each ref
        corr = F.softmax(corr, dim=-1)
        out = frame_transform(corr, refs, per_ref=per_ref, flatten=flatten)
        return out, corr
    else:
        corr_ = corr.permute(0, 2, 1, 3).flatten(2)
        corr_ = F.softmax(corr_, -1)
        out = frame_transform(corr_, refs, per_ref=per_ref, flatten=flatten)
        return out, corr_



def frame_transform(corr, refs, per_ref=True, local=False, patch_size=-1, flatten=True):
    
    """transform a target frame given refs and att

    Returns:
        out: transformed feature map (B*T*H*W) x C  if per_ref else (B*H*W) x C
        
    """
    if isinstance(refs, list):
        refs = torch.stack(refs, 1)
        refs = refs.flatten(3).permute(0, 1, 3, 2)
        
    if local:
        assert patch_size != -1
        bsz, t, feat_dim, w_, h_ = refs.shape
        unfold_fs = list([ F.unfold(ref, kernel_size=patch_size, \
            padding=int((patch_size-1)/2)).reshape(bsz, feat_dim, -1, w_, h_) for ref in refs])
        unfold_fs = torch.cat(unfold_fs, 2)
        out = (unfold_fs * corr).sum(2).reshape(bsz, feat_dim, -1).permute(0,2,1).reshape(-1, feat_dim)                                                                          
    else:
        if not per_ref:
            if refs.dim() == 4: 
                out =  torch.einsum('bij,bjc -> bic', [corr, refs.flatten(1,2)])
            else:
                out =  torch.einsum('bij,jc -> bic', [corr, refs.flatten(0,1)])
        else:
            # "btij,btjc -> btic“
            out = torch.matmul(corr, refs)
            
    if flatten:
        out = out.reshape(-1, refs.shape[-1])
        
    return out


def torch_unravel_index(indices, shape):
    rows = indices / shape[0]
    cols = indices % shape[1]

    return (rows, cols)