"""Shared Qwen generation runtime, also installed into adopted live kernels."""
import gc
import time
import torch

# Updating the generated code must retain the already loaded model replica.
if globals().get('_CC_DUAL_STATE',{}).get('primary_id')!=id(transformer):
    _CC_DUAL_STATE={'primary_id':id(transformer),'replica':None,'disabled':False}
_CC_ACCELERATION=globals().get('_CC_ACCELERATION','auto')


def _cc_encoder_ready():
    replica=_CC_DUAL_STATE['replica']
    if replica is not None and str(next(replica.parameters()).device)!='cpu':
        replica.to('cpu');torch.cuda.synchronize(1)
    gc.collect();torch.cuda.empty_cache()
    if str(encoder.device)!='cuda:1':
        move_int8_encoder(encoder,'cuda:1');torch.cuda.synchronize(1)
    pipe.register_modules(text_encoder=encoder)


def _cc_release_parallel():
    replica=_CC_DUAL_STATE.pop('replica',None)
    if replica is not None:replica.to('cpu')
    del replica
    _CC_DUAL_STATE['replica']=None
    gc.collect();torch.cuda.empty_cache()
    _cc_encoder_ready()


@torch.inference_mode()
def generate_image(prompt,seed=42,width=512,height=512,steps=20,negative_prompt=None):
    for device in range(2):torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter();encoding_seconds=0;movement=0;replica_load=0
    positive=negative=None;events=[];fallback=None
    parallel_enabled=_CC_ACCELERATION=='auto' and not _CC_DUAL_STATE['disabled']
    replica_reused=_CC_DUAL_STATE['replica'] is not None

    def encode():
        nonlocal encoding_seconds,movement
        before=time.perf_counter()
        _cc_encoder_ready()
        movement+=time.perf_counter()-before
        before=time.perf_counter()
        p=pipe.encode_prompt(prompt,device=torch.device('cuda:1'))
        n=pipe.encode_prompt(negative_prompt or ' ',device=torch.device('cuda:1'))
        p=tuple(v.to('cuda:0') if v is not None else None for v in p)
        n=tuple(v.to('cuda:0') if v is not None else None for v in n)
        torch.cuda.synchronize(0);torch.cuda.synchronize(1)
        encoding_seconds+=time.perf_counter()-before
        pipe.register_modules(text_encoder=None)
        return p,n

    def sample():
        nonlocal events
        events=[]
        def step(pipeline,index,timestep,kwargs):
            event=torch.cuda.Event(enable_timing=True);event.record(torch.cuda.current_stream(0));events.append(event)
            return kwargs
        before=torch.cuda.Event(enable_timing=True);before.record(torch.cuda.current_stream(0))
        image=pipe(prompt_embeds=positive[0],prompt_embeds_mask=positive[1],
            negative_prompt_embeds=negative[0],negative_prompt_embeds_mask=negative[1],
            width=width,height=height,num_inference_steps=steps,true_cfg_scale=5.0,use_kv_cache=True,
            sigmas=torch.linspace(1,0,steps+1)[:-1].tolist(),callback_on_step_end=step,
            generator=torch.Generator(device='cuda:0').manual_seed(seed)).images[0]
        after=torch.cuda.Event(enable_timing=True);after.record(torch.cuda.current_stream(0));torch.cuda.synchronize(0)
        denoise=before.elapsed_time(events[-1])/1000 if events else None
        decode=events[-1].elapsed_time(after)/1000 if events else None
        return image,denoise,decode

    try:
        if not parallel_enabled:
            before=time.perf_counter();_cc_release_parallel()
            movement+=time.perf_counter()-before
        positive,negative=encode()
        if parallel_enabled:
            before=time.perf_counter()
            move_int8_encoder(encoder,'cpu');torch.cuda.synchronize(1)
            gc.collect();torch.cuda.empty_cache()
            movement+=time.perf_counter()-before
            if _CC_DUAL_STATE['replica'] is None:
                emit_progress('loading','正在准备双卡加速模型；后续请求直接复用。')
                before=time.perf_counter()
                _CC_DUAL_STATE['replica']=load_replica(ASSET_DIR,'cuda:1')
                replica_load=time.perf_counter()-before
            else:
                before=time.perf_counter()
                _CC_DUAL_STATE['replica'].to('cuda:1');torch.cuda.synchronize(1)
                movement+=time.perf_counter()-before
            with ParallelCFG(transformer,_CC_DUAL_STATE['replica'],negative) as parallel:
                result,denoise,decode=sample()
            if parallel.calls!=steps:raise RuntimeError('CFG step count mismatch')
        else:result,denoise,decode=sample()
    except (RuntimeError,OSError) as error:
        if not parallel_enabled:raise
        fallback=type(error).__name__
        _CC_DUAL_STATE['disabled']=True
        before=time.perf_counter();_cc_release_parallel()
        movement+=time.perf_counter()-before
        # This is the same request and seed; no external task is resubmitted.
        if positive is None:positive,negative=encode()
        pipe.register_modules(text_encoder=None)
        result,denoise,decode=sample()
        parallel_enabled=False
    finally:
        pipe.register_modules(text_encoder=encoder if str(encoder.device)=='cuda:1' else None)
    elapsed=time.perf_counter()-started
    return result,{'model_id':'qwen-image-2.1','encoding_seconds':round(encoding_seconds,3),
        'generation_seconds':round(elapsed-encoding_seconds,3),'total_generation_seconds':round(elapsed,3),
        'migration_seconds':round(movement,3),'replica_load_seconds':round(replica_load,3),
        'dual_model_reused':bool(parallel_enabled and replica_reused),
        'acceleration_mode':_CC_ACCELERATION,'acceleration_fallback':fallback,
        'denoising_seconds':round(denoise,3) if denoise is not None else None,
        'decode_seconds':round(decode,3) if decode is not None else None,
        'denoising_steps':len(events),'milliseconds_per_step':round(denoise*1000/len(events),2) if events else None,
        'attention_backend':'sdpa','parallel_strategy':'cfg_parallel' if parallel_enabled else 'encoder_split',
        'width':width,'height':height,'steps':steps,'seed':seed,'task':'text',
        'peak_gpu_gib':[round(torch.cuda.max_memory_allocated(i)/2**30,2) for i in range(2)]}
