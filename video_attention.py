"""Broadcast shared LTX prompt masks across Ulysses-sharded attention heads."""
from functools import wraps


def broadcast_ltx_prompt_mask(mask,torch_module=None):
    if mask is None or mask.ndim!=4 or mask.shape[1]==1:return mask
    if torch_module is None:
        import torch as torch_module
    shared=mask[:,:1]
    if not torch_module.equal(mask,shared.expand_as(mask)):
        raise ValueError('LTX parallel attention requires a shared prompt mask')
    return shared


def enable_ltx_parallel_mask_broadcast():
    # This module patch lives only inside the dedicated LTX rank process.
    from diffusers.models.transformers import transformer_ltx
    original=transformer_ltx.dispatch_attention_fn
    if getattr(original,'_cc_ltx_mask_broadcast',False):return
    @wraps(original)
    def dispatch(*args,**kwargs):
        if kwargs.get('parallel_config') is not None:
            kwargs['attn_mask']=broadcast_ltx_prompt_mask(kwargs.get('attn_mask'))
        return original(*args,**kwargs)
    dispatch._cc_ltx_mask_broadcast=True
    transformer_ltx.dispatch_attention_fn=dispatch
