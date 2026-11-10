"""Transactional two-GPU CFG for Wan's separate conditional/unconditional calls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import wraps


class WanParallelCFG:
    def __init__(self,primary,secondary,negative,torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch=torch_module;self.primary=primary;self.secondary=secondary
        self.negative=negative.to('cuda:1');self.pending=None;self.calls=0

    def __enter__(self):
        self.original=self.primary.forward
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cc-video-cfg')
        @wraps(self.original)
        def forward(*args,**kwargs):
            if args or kwargs.get('return_dict') is not False:
                raise ValueError('Unsupported Wan CFG forward signature')
            if self.pending is not None:
                pending,self.pending=self.pending,None
                output,event=pending.result();event.synchronize();self.calls+=1
                return tuple(x.to('cuda:0') if hasattr(x,'to') else x for x in output)
            copy={k:self._copy(v) for k,v in kwargs.items()}
            copy['encoder_hidden_states']=self.negative
            # Finish peer copies before launching either branch on another thread.
            self.torch.cuda.synchronize(1)
            self.pending=self.pool.submit(self._unconditional,copy)
            return self.original(**kwargs)
        self.primary.forward=forward
        return self

    def _copy(self,value):
        if self.torch.is_tensor(value):return value.to('cuda:1')
        if isinstance(value,dict):return {k:self._copy(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return type(value)(self._copy(v) for v in value)
        return value

    def _unconditional(self,kwargs):
        with self.torch.inference_mode(),self.torch.cuda.device(1):
            context=getattr(self.secondary,'cache_context',None)
            with context('uncond') if context else nullcontext():output=self.secondary(**kwargs)
            event=self.torch.cuda.Event();event.record(self.torch.cuda.current_stream(1))
            return output,event

    def __exit__(self,kind,*rest):
        self.primary.forward=self.original
        self.pool.shutdown(wait=True,cancel_futures=True)
        unpaired=self.pending is not None;self.pending=None
        if kind is None and (unpaired or self.calls==0):raise RuntimeError('Unpaired Wan CFG calls')
