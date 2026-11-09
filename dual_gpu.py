"""Experimental CFG branch parallelism with transactional forward restoration.

Only the unconditional transformer runs on GPU 1. Sampling and conditional
inference stay on GPU 0. Tensor copies happen before concurrent launches.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import wraps


class ParallelCFG:
    def __init__(self,primary,secondary,negative,torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch=torch_module
        self.primary=primary;self.secondary=secondary
        self.original=None;self.pending=None;self.pool=None
        self.negative=tuple(value.to('cuda:1') if value is not None else None for value in negative)
        self.cache=None;self.full_mask=None;self.calls=0

    def __enter__(self):
        self.original=self.primary.forward
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cc-cfg')
        original=self.original
        @wraps(original)
        def forward(*args,**kwargs):
            if args or kwargs.get('return_dict') is not False:
                raise ValueError('Parallel CFG requires keyword arguments and tuple output')
            if self.pending is not None:
                pending,self.pending=self.pending,None
                result,event=pending.result()
                event.synchronize()
                self.calls+=1
                return tuple(value.to('cuda:0') if hasattr(value,'to') else value for value in result)
            if kwargs.get('kv_cache_mode')=='extract':
                current=kwargs.get('kv_cache')
                self.cache=type(current)(len(self.secondary.transformer_blocks)) if current is not None else None
            # Copies are completed before either branch is launched, avoiding
            # cross-thread CUDA stream races on the previous step's latents.
            secondary_kwargs=dict(kwargs)
            for name in ('hidden_states','timestep'):
                secondary_kwargs[name]=kwargs[name].to('cuda:1')
            embeds,mask=self.negative[:2]
            if self.full_mask is None:
                prefix=self.negative[2] if len(self.negative)>2 and self.negative[2] is not None else self.torch.zeros_like(mask,dtype=self.torch.bool)
                target=prefix.new_ones(prefix.shape[0],kwargs['hidden_states'].shape[1]//4)
                self.full_mask=self.torch.cat((prefix,target),dim=1)
            secondary_kwargs.update(encoder_hidden_states=embeds,encoder_hidden_states_mask=mask,
                img_mask=self.full_mask,kv_cache=self.cache)
            self.pending=self.pool.submit(self._negative_forward,secondary_kwargs)
            return original(**kwargs)
        self.primary.forward=forward
        return self

    def _negative_forward(self,kwargs):
        with self.torch.inference_mode(),self.torch.cuda.device(1):
            context=getattr(self.secondary,'cache_context',None)
            with context('uncond') if context else nullcontext():
                result=self.secondary(**kwargs)
            event=self.torch.cuda.Event();event.record(self.torch.cuda.current_stream(1))
            return result,event

    def __exit__(self,*error):
        self.primary.forward=self.original
        try:
            if self.pool:self.pool.shutdown(wait=True,cancel_futures=True)
        finally:
            self.pending=None;self.cache=None;self.full_mask=None
        if not error[0] and self.calls==0:raise RuntimeError('No paired CFG forwards were executed')


def load_replica(asset_dir,device='cuda:1',checkpoint_path=None):
    from pathlib import Path
    import torch
    from diffusers import QwenImage21Transformer2DModel,GGUFQuantizationConfig
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
    folder=Path(asset_dir)
    replica=QwenImage21Transformer2DModel.from_single_file(str(checkpoint_path or folder/'transformer-q4.gguf'),
        config=str(folder),subfolder='transformer',quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        dtype=torch.float16,local_files_only=True).to(device)
    for name,value in list(replica.named_parameters()):
        if value.ndim==1 and hasattr(value,'quant_type'):
            parent,_,field=name.rpartition('.')
            module=replica.get_submodule(parent) if parent else replica
            decoded=dequantize_gguf_tensor(value).to(dtype=torch.float16)
            if getattr(value,'quant_shape',None) is not None:decoded=decoded.reshape(tuple(value.quant_shape))
            setattr(module,field,torch.nn.Parameter(decoded,requires_grad=False))
    return replica


def prepare_parallel_replica(asset_dir,encoder,torch_module=None):
    """Move encoder aside before allocating replica. Caller owns rollback."""
    if torch_module is None:
        import torch as torch_module
    encoder.to('cpu')
    torch_module.cuda.empty_cache()
    try:return load_replica(asset_dir)
    except Exception:
        torch_module.cuda.empty_cache()
        encoder.to('cuda:1')
        raise


def restore_encoder(encoder,replica=None,torch_module=None):
    if torch_module is None:
        import torch as torch_module
    if replica is not None:replica.to('cpu')
    torch_module.cuda.empty_cache()
    encoder.to('cuda:1')
