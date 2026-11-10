"""Additional model loaders. The caller embeds pinned catalog and asset helpers."""
import gc
import time
import torch

assert torch.cuda.device_count()>=2,'Select GPU T4 x2.'
ASSET_DIR=Path('/kaggle/temp/cc-media-models')/_CC_MODEL_ID
started=time.perf_counter()
prepare_selected_model_assets(ASSET_DIR)
_assets_ready=time.perf_counter()
emit_progress('loading','正在加载所选模型。')
if _CC_MODEL_ID=='z-image-turbo':
    from diffusers import ZImagePipeline,ZImageTransformer2DModel,AutoencoderKL
    from transformers import Qwen3Model,BitsAndBytesConfig
    encoder=Qwen3Model.from_pretrained(ASSET_DIR/'text_encoder',torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),device_map={'':1},local_files_only=True)
    transformer=ZImageTransformer2DModel.from_pretrained(ASSET_DIR/'transformer',torch_dtype=torch.float16,device_map={'':0},low_cpu_mem_usage=True,local_files_only=True)
    stabilize_zimage_fp16(transformer)
    vae=AutoencoderKL.from_pretrained(ASSET_DIR/'vae',torch_dtype=torch.float32,local_files_only=True).to('cuda:0')
    pipe=ZImagePipeline.from_pretrained(ASSET_DIR,torch_dtype=torch.float16,text_encoder=encoder,
        transformer=transformer,vae=vae,local_files_only=True)
else:
    from diffusers import StableDiffusionXLPipeline,UNet2DConditionModel,EulerDiscreteScheduler
    from safetensors.torch import load_file
    transformer=UNet2DConditionModel.from_config(ASSET_DIR/'base',subfolder='unet')
    transformer.load_state_dict(load_file(ASSET_DIR/'lightning/sdxl_lightning_4step_unet.safetensors',device='cpu'))
    transformer=transformer.to(device='cuda:0',dtype=torch.float16)
    pipe=StableDiffusionXLPipeline.from_pretrained(ASSET_DIR/'base',unet=transformer,
        torch_dtype=torch.float16,variant='fp16',local_files_only=True).to('cuda:0')
    pipe.scheduler=EulerDiscreteScheduler.from_config(pipe.scheduler.config,timestep_spacing='trailing')
    pipe.text_encoder.to('cuda:1');pipe.text_encoder_2.to('cuda:1')
    encoder=pipe.text_encoder
load_stats={'copy_seconds':round(_assets_ready-started,2),'loading_seconds':round(time.perf_counter()-_assets_ready,2)}


@torch.inference_mode()
def generate_image(prompt,seed=42,width=512,height=512,steps=None):
    steps=_CC_MODEL_PROFILE['default_steps'] if steps is None else steps
    if _CC_MODEL_ID=='sdxl-lightning' and steps!=4:raise ValueError('SDXL-Lightning requires 4 steps')
    for i in range(2):torch.cuda.reset_peak_memory_stats(i)
    started=time.perf_counter();encoding_seconds=0
    if _CC_MODEL_ID=='z-image-turbo':
        positive,_=pipe.encode_prompt(prompt,device=torch.device('cuda:1'),do_classifier_free_guidance=False)
        torch.cuda.synchronize(1);encoding_seconds=time.perf_counter()-started
        kwargs={'prompt_embeds':[value.to('cuda:0') for value in positive],'guidance_scale':0.0}
    else:
        positive,negative,pooled,negative_pooled=pipe.encode_prompt(prompt=prompt,device=torch.device('cuda:1'),do_classifier_free_guidance=False)
        torch.cuda.synchronize(0);encoding_seconds=time.perf_counter()-started
        kwargs={'prompt_embeds':positive.to('cuda:0'),'pooled_prompt_embeds':pooled.to('cuda:0'),'guidance_scale':0.0}
    embeds=kwargs['prompt_embeds'] if isinstance(kwargs['prompt_embeds'],list) else [kwargs['prompt_embeds'],kwargs['pooled_prompt_embeds']]
    if any(not torch.isfinite(value).all().item() for value in embeds):raise RuntimeError('Text encoder produced non-finite embeddings')
    # Explicit denoising/decode timing; CUDA events avoid synchronizing every step.
    timings=[]
    def record_step(pipeline,index,timestep,callback_kwargs):
        if index==steps-1 and not torch.isfinite(callback_kwargs['latents']).all().item():
            raise RuntimeError('Sampling produced non-finite latents')
        event=torch.cuda.Event(enable_timing=True);event.record(torch.cuda.current_stream(0));timings.append(event)
        return callback_kwargs
    before=torch.cuda.Event(enable_timing=True);before.record(torch.cuda.current_stream(0))
    # Prompt embeds are already on GPU 0. Exclude encoders while the pipeline
    # chooses its execution device, then restore them for the next request.
    encoders={'text_encoder':encoder}
    if _CC_MODEL_ID=='sdxl-lightning':encoders['text_encoder_2']=pipe.text_encoder_2
    pipe.register_modules(**{name:None for name in encoders})
    try:
        result=pipe(width=width,height=height,num_inference_steps=steps,
            generator=torch.Generator(device='cuda:0').manual_seed(seed),callback_on_step_end=record_step,**kwargs).images[0]
    finally:pipe.register_modules(**encoders)
    after=torch.cuda.Event(enable_timing=True);after.record(torch.cuda.current_stream(0));torch.cuda.synchronize(0)
    denoise=before.elapsed_time(timings[-1])/1000 if timings else None
    decode=timings[-1].elapsed_time(after)/1000 if timings else None
    return result,{'model_id':_CC_MODEL_ID,'encoding_seconds':round(encoding_seconds,3),
        'generation_seconds':round(time.perf_counter()-started-encoding_seconds,3),
        'denoising_seconds':round(denoise,3) if denoise is not None else None,
        'decode_seconds':round(decode,3) if decode is not None else None,
        'denoising_steps':len(timings),'milliseconds_per_step':round(denoise*1000/len(timings),2) if timings else None,
        'attention_backend':'sdpa','parallel_strategy':'encoder_split','acceleration_mode':globals().get('_CC_ACCELERATION','auto'),
        'width':width,'height':height,'steps':steps,'seed':seed,
        'peak_gpu_gib':[round(torch.cuda.max_memory_allocated(i)/2**30,2) for i in range(2)]}
