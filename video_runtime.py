"""GPU video runtime. Imported only on the remote Kaggle kernel."""
import gc
import json
import time
from pathlib import Path
import torch


class VideoRuntime:
    def __init__(self,model_id='wan21-t2v-1.3b',acceleration='auto',
                 root='/kaggle/temp/cc-media-video',progress=None,cache_handle=None,cache_token=None):
        from video_registry import video_profile
        if acceleration not in ('auto','split'):raise ValueError('Invalid acceleration')
        self.profile=video_profile(model_id);self.acceleration=acceleration
        self.folder=Path(root)/model_id;self.progress=progress or (lambda e:None)
        self.pipe=None;self.encoder=None;self.replica=None;self.calls=0;self.disabled=False
        self.cache_handle=cache_handle;self.cache_token=cache_token;self.cache=None;self.context=None

    def emit(self,phase,**extra):self.progress(dict(phase=phase,**extra))

    def prepare(self):
        import torch
        from transformers import BitsAndBytesConfig,T5EncoderModel,UMT5EncoderModel
        from diffusers import (LTXPipeline,WanPipeline,
                               WanTransformer3DModel,AutoencoderKLWan)
        from video_assets import prepare_video_assets,video_cache_context
        if torch.cuda.device_count()<2:raise RuntimeError('Select GPU T4 x2')
        start=time.perf_counter();p=self.profile
        restored=False
        if self.cache_handle and self.cache_token:
            self.cache=video_cache_context(p,self.cache_handle,self.cache_token,self.progress)
            try:
                self.emit('cache-check')
                metadata=self.cache['cache_metadata'](self.cache_handle,self.cache_token)
                if metadata:
                    self.cache['download_cache_assets'](self.folder,self.cache_handle,self.cache_token,metadata)
                    restored=True;self.cache['_QWEN_CACHE_STATUS']='saved'
            except Exception:
                self.cache['_QWEN_CACHE_STATUS']='unavailable';self.emit('cache-warning')
        if not restored:prepare_video_assets(p,self.folder,self.progress)
        self.emit('loading')
        encoder_type=T5EncoderModel if p['family']=='ltx' else UMT5EncoderModel
        self.encoder=encoder_type.from_pretrained(self.folder/'text_encoder',
            torch_dtype=torch.float16,device_map={'':1},low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),local_files_only=True)
        if p['family']=='ltx':
            self.pipe=LTXPipeline.from_single_file(str(self.folder/p['checkpoint']),
                config=str(self.folder),text_encoder=self.encoder,torch_dtype=torch.float16,
                local_files_only=True)
            from diffusers import FlowMatchEulerDiscreteScheduler
            self.pipe.scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler.config,
                use_dynamic_shifting=False,shift=1.0)
            self.pipe.transformer.to('cuda:0');self.pipe.vae.to(device='cuda:0',dtype=torch.float32)
        else:
            self.pipe_class=WanPipeline
            self.transformer_type=WanTransformer3DModel
            self.pipe=self.pipe_class.from_pretrained(self.folder,text_encoder=self.encoder,
                transformer=self.load_transformer(0),
                vae=AutoencoderKLWan.from_pretrained(self.folder/'vae',torch_dtype=torch.float32,
                    local_files_only=True).to('cuda:0'),torch_dtype=torch.float16,local_files_only=True)
        if hasattr(self.pipe.vae,'enable_tiling'):self.pipe.vae.enable_tiling()
        if hasattr(self.pipe.vae,'enable_slicing'):self.pipe.vae.enable_slicing()
        self.load_seconds=time.perf_counter()-start
        if self.cache and not restored:self.cache['save_private_cache'](self.folder,self.encoder)
        return self

    def load_transformer(self,device):
        import torch
        from diffusers import BitsAndBytesConfig
        kwargs={}
        if self.profile['family']=='wan22':kwargs['quantization_config']=BitsAndBytesConfig(load_in_8bit=True)
        return self.transformer_type.from_pretrained(self.folder/'transformer',
            torch_dtype=torch.float16,device_map={'':device},low_cpu_mem_usage=True,
            local_files_only=True,**kwargs)

    @torch.inference_mode()
    def generate(self,prompt,seed=42,steps=None,frames=None,output=None):
        import torch
        from video_registry import validate_video
        from video_precision import move_video_encoder
        from video_parallel import WanParallelCFG
        p=self.profile;params=validate_video(prompt,seed,steps,p['id'],frames)
        if self.pipe is None:self.prepare()
        for i in range(2):torch.cuda.reset_peak_memory_stats(i)
        start=time.perf_counter();self.emit('encoding')
        # The 5B replica and UMT5 encoding workspace should not overlap on GPU 1.
        if p['family']=='wan22' and self.replica is not None:
            move_video_encoder(self.replica,'cpu');torch.cuda.empty_cache()
        move_video_encoder(self.encoder,'cuda:1')
        encoded=self.pipe.encode_prompt(prompt=prompt,negative_prompt='',
            do_classifier_free_guidance=p['guidance']>1,device=torch.device('cuda:1'))
        if p['family']=='ltx':positive,mask,negative,negative_mask=encoded
        else:positive,negative=encoded[:2]
        if not torch.isfinite(positive).all() or (negative is not None and not torch.isfinite(negative).all()):
            raise RuntimeError('Non-finite video prompt embeddings')
        positive=positive.to('cuda:0');negative=negative.to('cuda:0') if negative is not None else None
        torch.cuda.synchronize();encoding=time.perf_counter()-start
        move_video_encoder(self.encoder,'cpu');torch.cuda.empty_cache()
        if self.acceleration=='split' and self.replica is not None:
            self.replica=None;gc.collect();torch.cuda.empty_cache()
        replica_reused=self.replica is not None;replica_seconds=0;strategy='encoder_split'
        if p['family']!='ltx' and self.acceleration=='auto' and not self.disabled:
            try:
                if self.replica is None:
                    tick=time.perf_counter();self.replica=self.load_transformer(1);replica_seconds=time.perf_counter()-tick
                elif p['family']=='wan22':move_video_encoder(self.replica,'cuda:1')
                strategy='cfg_parallel'
            except (RuntimeError,ValueError):
                self.replica=None;self.disabled=True;gc.collect();torch.cuda.empty_cache()
        latents_finite=[]
        def progress(pipeline,index,timestep,kwargs):
            self.emit('generating',step=index+1,steps=params['steps'])
            if index==params['steps']-1:
                latents_finite.append(bool(torch.isfinite(kwargs['latents']).all().item()))
                if not latents_finite[-1]:raise RuntimeError('Non-finite video latents')
            return kwargs
        kwargs=dict(prompt_embeds=positive,negative_prompt_embeds=negative,
            width=p['width'],height=p['height'],num_frames=params['frames'],
            num_inference_steps=params['steps'],guidance_scale=p['guidance'],
            generator=torch.Generator(device='cuda:0').manual_seed(seed),
            callback_on_step_end=progress,output_type='np')
        if p['family']=='ltx':kwargs.update(decode_timestep=.05,decode_noise_scale=.025,frame_rate=p['fps'],
            prompt_attention_mask=mask.to('cuda:0'),negative_prompt_attention_mask=negative_mask,
            timesteps=p['timesteps'])
        # Wan2.2 hybrid generation accepts no input image for text-to-video.
        self.pipe.register_modules(text_encoder=None)
        sampling=time.perf_counter();fallback=None;cfg_pairs=0
        original_decode=self.pipe.vae.decode
        def decode(*args,**kwargs):
            self.emit('decoding')
            return original_decode(*args,**kwargs)
        self.pipe.vae.decode=decode
        try:
            if p['family']=='ltx' and self.acceleration=='auto' and not self.disabled:
                from video_context import LTXContextRunner
                if self.context is None:
                    runner=None;original=self.pipe.transformer
                    try:
                        runner=LTXContextRunner(self.folder,p,original,self.pipe.vae.config,
                            self.pipe.scheduler.config,self.progress)
                        original.to('cpu');self.pipe.register_modules(transformer=None)
                        move_video_encoder(self.encoder,'cuda:1');gc.collect();torch.cuda.empty_cache()
                        self.context=runner.start()
                    except Exception as error:
                        print('CC_VIDEO_CONTEXT_ERROR:'+str(error),flush=True)
                        if runner is not None:runner.close()
                        self.pipe.register_modules(transformer=original.to('cuda:0'))
                        self.disabled=True;fallback=type(error).__name__
                    finally:del original
                if self.context is not None:
                    try:packed=self.context.generate(positive,mask,params)
                    except (RuntimeError,ValueError,TimeoutError) as error:
                        print('CC_VIDEO_CONTEXT_ERROR:'+str(error),flush=True)
                        self.context.close();self.context=None;self.disabled=True;fallback=type(error).__name__
                        from diffusers import LTXVideoTransformer3DModel
                        from video_context import context_transformer_folder
                        self.pipe.register_modules(transformer=LTXVideoTransformer3DModel.from_pretrained(
                            context_transformer_folder(self.folder,p),torch_dtype=torch.float16,
                            device_map={'':0},local_files_only=True))
                        move_video_encoder(self.encoder,'cpu');torch.cuda.empty_cache()
                        result=self.pipe(**kwargs)
                    else:
                        generator=torch.Generator(device='cuda:0');generator.set_state(packed['generator_state'])
                        decoded=self.decode_context_latents(packed['latents'].to('cuda:0'),params['frames'],generator)
                        from types import SimpleNamespace
                        result=SimpleNamespace(frames=decoded);strategy='context_parallel'
                else:result=self.pipe(**kwargs)
            elif strategy=='cfg_parallel':
                try:
                    router=WanParallelCFG(self.pipe.transformer,self.replica,negative)
                    try:
                        with router:result=self.pipe(**kwargs)
                        cfg_pairs=router.calls
                    finally:del router
                except RuntimeError as error:
                    if not isinstance(error,torch.cuda.OutOfMemoryError):raise
                    fallback=type(error).__name__;self.replica=None;self.disabled=True
                if fallback:
                    gc.collect();torch.cuda.empty_cache();strategy='encoder_split'
                    kwargs['generator']=torch.Generator(device='cuda:0').manual_seed(seed)
                    result=self.pipe(**kwargs)
            else:
                if self.context is not None:
                    self.context.close();self.context=None
                    from diffusers import LTXVideoTransformer3DModel
                    from video_context import context_transformer_folder
                    self.pipe.register_modules(transformer=LTXVideoTransformer3DModel.from_pretrained(
                        context_transformer_folder(self.folder,p),torch_dtype=torch.float16,
                        device_map={'':0},local_files_only=True))
                result=self.pipe(**kwargs)
        finally:
            self.pipe.vae.decode=original_decode
            self.pipe.register_modules(text_encoder=self.encoder)
        torch.cuda.synchronize();sampling_seconds=time.perf_counter()-sampling
        import numpy as np
        video=np.asarray(result.frames[0])
        if not np.isfinite(video).all() or float(video.std())<.001:raise RuntimeError('Invalid or blank video')
        self.emit('exporting');from diffusers.utils import export_to_video
        output=Path(output or '/kaggle/temp/cc-media-video-result.mp4');output.parent.mkdir(parents=True,exist_ok=True)
        export_to_video(video,str(output),fps=p['fps'])
        self.calls+=1
        peaks=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]
        if strategy=='context_parallel':peaks=[max(peaks[i],packed['worker_peak_gib'][i]+torch.cuda.memory_allocated(i)/2**30) for i in range(2)]
        return output,dict(model_id=p['id'],media_type='video',width=p['width'],height=p['height'],
            frames=params['frames'],fps=p['fps'],steps=params['steps'],seed=seed,
            encoding_seconds=round(encoding,3),generation_seconds=round(sampling_seconds,3),
            request_seconds=round(time.perf_counter()-start,3),loading_seconds=round(self.load_seconds,3),
            model_reused=self.calls>1,model_load_count=1,generation_number=self.calls,
            parallel_strategy=strategy,acceleration_mode=self.acceleration,cfg_pairs=cfg_pairs,
            replica_load_seconds=round(replica_seconds,3),dual_model_reused=replica_reused,
            acceleration_fallback=fallback,peak_gpu_gib=[round(v,3) for v in peaks])

    def decode_context_latents(self,latents,frames,generator):
        import torch
        pipe=self.pipe;p=self.profile
        latents=pipe._unpack_latents(latents,(frames-1)//pipe.vae_temporal_compression_ratio+1,
            p['height']//pipe.vae_spatial_compression_ratio,p['width']//pipe.vae_spatial_compression_ratio,
            pipe.transformer_spatial_patch_size,pipe.transformer_temporal_patch_size)
        latents=pipe._denormalize_latents(latents,pipe.vae.latents_mean,pipe.vae.latents_std,pipe.vae.config.scaling_factor)
        timestep=None
        if pipe.vae.config.timestep_conditioning:
            from diffusers.utils.torch_utils import randn_tensor
            noise=randn_tensor(latents.shape,generator=generator,device=latents.device,dtype=latents.dtype)
            latents=.975*latents+.025*noise
            timestep=torch.tensor([.05],device=latents.device,dtype=latents.dtype)
        decoded=pipe.vae.decode(latents.to(pipe.vae.dtype),timestep,return_dict=False)[0]
        return pipe.video_processor.postprocess_video(decoded,output_type='np')

    def close(self):
        if self.context is not None:self.context.close();self.context=None
        self.pipe=None;self.encoder=None;self.replica=None
        self.cache=None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
