"""GPU video runtime. Imported only on the remote Kaggle kernel."""
import gc
import json
import time
from pathlib import Path


class VideoRuntime:
    def __init__(self,model_id='wan21-t2v-1.3b',acceleration='auto',
                 root='/kaggle/temp/cc-media-video',progress=None):
        from video_registry import video_profile
        if acceleration not in ('auto','split'):raise ValueError('Invalid acceleration')
        self.profile=video_profile(model_id);self.acceleration=acceleration
        self.folder=Path(root)/model_id;self.progress=progress or (lambda e:None)
        self.pipe=None;self.encoder=None;self.replica=None;self.calls=0;self.disabled=False

    def emit(self,phase,**extra):self.progress(dict(phase=phase,**extra))

    def prepare(self):
        import torch
        from transformers import BitsAndBytesConfig,T5EncoderModel,UMT5EncoderModel
        from diffusers import (LTXPipeline,WanPipeline,
                               WanTransformer3DModel,AutoencoderKLWan)
        from video_assets import prepare_video_assets
        if torch.cuda.device_count()<2:raise RuntimeError('Select GPU T4 x2')
        start=time.perf_counter();p=self.profile
        prepare_video_assets(p,self.folder,self.progress)
        self.emit('loading')
        encoder_type=T5EncoderModel if p['family']=='ltx' else UMT5EncoderModel
        self.encoder=encoder_type.from_pretrained(self.folder/'text_encoder',
            torch_dtype=torch.float16,device_map={'':1},low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),local_files_only=True)
        if p['family']=='ltx':
            self.pipe=LTXPipeline.from_single_file(str(self.folder/p['checkpoint']),
                config=str(self.folder),text_encoder=self.encoder,torch_dtype=torch.float16,
                local_files_only=True)
            self.pipe.transformer.to('cuda:0');self.pipe.vae.to('cuda:0')
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
        return self

    def load_transformer(self,device):
        import torch
        from diffusers import BitsAndBytesConfig
        kwargs={}
        if self.profile['family']=='wan22':kwargs['quantization_config']=BitsAndBytesConfig(load_in_8bit=True)
        return self.transformer_type.from_pretrained(self.folder/'transformer',
            torch_dtype=torch.float16,device_map={'':device},low_cpu_mem_usage=True,
            local_files_only=True,**kwargs)

    def generate(self,prompt,seed=42,steps=None,frames=None,output=None):
        import torch
        from video_registry import validate_video
        from dual_gpu import move_int8_encoder
        from video_parallel import WanParallelCFG
        p=self.profile;params=validate_video(prompt,seed,steps,p['id'],frames)
        if self.pipe is None:self.prepare()
        for i in range(2):torch.cuda.reset_peak_memory_stats(i)
        start=time.perf_counter();self.emit('encoding')
        move_int8_encoder(self.encoder,'cuda:1')
        positive,negative=self.pipe.encode_prompt(prompt=prompt,negative_prompt='',
            do_classifier_free_guidance=p['guidance']>1,device=torch.device('cuda:1'))[:2]
        if not torch.isfinite(positive).all() or (negative is not None and not torch.isfinite(negative).all()):
            raise RuntimeError('Non-finite video prompt embeddings')
        positive=positive.to('cuda:0');negative=negative.to('cuda:0') if negative is not None else None
        torch.cuda.synchronize();encoding=time.perf_counter()-start
        move_int8_encoder(self.encoder,'cpu');torch.cuda.empty_cache()
        replica_reused=self.replica is not None;replica_seconds=0;strategy='encoder_split'
        if p['family']!='ltx' and self.acceleration=='auto' and not self.disabled:
            try:
                if self.replica is None:
                    tick=time.perf_counter();self.replica=self.load_transformer(1);replica_seconds=time.perf_counter()-tick
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
        if p['family']=='ltx':kwargs.update(decode_timestep=.05,decode_noise_scale=.025)
        # Wan2.2 hybrid generation accepts no input image for text-to-video.
        self.pipe.register_modules(text_encoder=None)
        sampling=time.perf_counter();fallback=None
        try:
            if strategy=='cfg_parallel':
                try:
                    with WanParallelCFG(self.pipe.transformer,self.replica,negative):result=self.pipe(**kwargs)
                except RuntimeError as error:
                    if not isinstance(error,torch.cuda.OutOfMemoryError):raise
                    fallback=type(error).__name__;self.replica=None;self.disabled=True
                    gc.collect();torch.cuda.empty_cache();strategy='encoder_split'
                    kwargs['generator']=torch.Generator(device='cuda:0').manual_seed(seed)
                    result=self.pipe(**kwargs)
            else:result=self.pipe(**kwargs)
        finally:self.pipe.register_modules(text_encoder=self.encoder)
        torch.cuda.synchronize();sampling_seconds=time.perf_counter()-sampling
        import numpy as np
        video=np.asarray(result.frames[0])
        if not np.isfinite(video).all() or float(video.std())<.001:raise RuntimeError('Invalid or blank video')
        self.emit('exporting');from diffusers.utils import export_to_video
        output=Path(output or '/kaggle/temp/cc-media-video-result.mp4');output.parent.mkdir(parents=True,exist_ok=True)
        export_to_video(video,str(output),fps=p['fps'])
        self.calls+=1
        return output,dict(model_id=p['id'],media_type='video',width=p['width'],height=p['height'],
            frames=params['frames'],fps=p['fps'],steps=params['steps'],seed=seed,
            encoding_seconds=round(encoding,3),generation_seconds=round(sampling_seconds,3),
            request_seconds=round(time.perf_counter()-start,3),loading_seconds=round(self.load_seconds,3),
            model_reused=self.calls>1,model_load_count=1,generation_number=self.calls,
            parallel_strategy=strategy,acceleration_mode=self.acceleration,
            replica_load_seconds=round(replica_seconds,3),dual_model_reused=replica_reused,
            acceleration_fallback=fallback,peak_gpu_gib=[round(torch.cuda.max_memory_allocated(i)/2**30,3) for i in range(2)])

    def close(self):
        self.pipe=None;self.encoder=None;self.replica=None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
