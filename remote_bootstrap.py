import json, shutil, time
from pathlib import Path
import torch
import kagglehub
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from diffusers import (QwenImage21Pipeline, QwenImage21Transformer2DModel,
                       AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler,
                       GGUFQuantizationConfig)

assert torch.cuda.device_count() == 2, 'Select GPU T4 x2 before running.'
started = time.perf_counter()
ASSET_DIR = Path('/kaggle/temp/qwen-image-assets')
_hf_mode = globals().get('_QWEN_ASSET_MODE') in ('huggingface','auto')
if globals().get('_QWEN_ASSET_MODE') == 'auto':
    source, _hf_mode = prepare_auto_assets(ASSET_DIR)
elif _hf_mode:
    source = prepare_hf_assets(ASSET_DIR)
else:
    emit_progress('dataset', '正在拉取 Kaggle 模型 Dataset。')
    source = Path(kagglehub.dataset_download('faberbruntch/qwen-image-t4-assets'))
    copy_assets(source, ASSET_DIR)
copy_seconds = time.perf_counter()-started
print(f'Local assets ready: {copy_seconds:.2f} seconds', flush=True)
print('Loading INT8 text encoder on GPU 1...', flush=True)
from transformers import BitsAndBytesConfig
_encoder_options = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True)} if _hf_mode else {}
encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    ASSET_DIR/'text_encoder', dtype=torch.float16, device_map={'': 1},
    attn_implementation='sdpa', local_files_only=True, **_encoder_options)
print('Loading Q4 transformer on GPU 0...', flush=True)
def _cc_load_transformer():
    return QwenImage21Transformer2DModel.from_single_file(
        str(ASSET_DIR/'transformer-q4.gguf'), config=str(ASSET_DIR), subfolder='transformer',
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        dtype=torch.float16, local_files_only=True).to('cuda:0')
try:
    transformer = _cc_load_transformer()
except Exception as _cc_error:
    # Seen intermittently on Kaggle: the GGUF failed to load. Check the file
    # against its download receipt, replace it if damaged, and retry once.
    _cc_first_failure = exception_chain(_cc_error)
    del _cc_error
    print('CC_LOAD_RETRY: ' + _cc_first_failure, flush=True)
    emit_progress('verify', '正在校验 Q4 权重并重试加载。')
    _cc_gguf = ASSET_DIR/'transformer-q4.gguf'
    if file_matches_receipt(_cc_gguf) is False:
        _cc_gguf.unlink(missing_ok=True)
        _cc_gguf.with_name(_cc_gguf.name + '.verified.json').unlink(missing_ok=True)
        prepare_hf_assets(ASSET_DIR)
    import gc
    gc.collect(); torch.cuda.empty_cache()
    try:
        transformer = _cc_load_transformer()
    except Exception as _cc_error:
        _cc_second_failure = exception_chain(_cc_error)
        del _cc_error
        raise RuntimeError('Q4 transformer failed to load twice: ' + _cc_second_failure + ' | first: ' + _cc_first_failure) from None
# Diffusers GGUFLinear handles packed matrices; custom RMSNorm vectors need
# explicit decoding when a pre-converted GGUF stores them as BF16 bytes.
from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
_gguf_vector_fixes = 0
for _name, _value in list(transformer.named_parameters()):
    if _value.ndim == 1 and hasattr(_value, 'quant_type'):
        _parent, _, _field = _name.rpartition('.')
        _module = transformer.get_submodule(_parent) if _parent else transformer
        _decoded = dequantize_gguf_tensor(_value).to(dtype=torch.float16)
        _logical_shape = getattr(_value, 'quant_shape', None)
        if _logical_shape is not None:
            _decoded = _decoded.reshape(tuple(_logical_shape))
        setattr(_module, _field, torch.nn.Parameter(_decoded, requires_grad=False))
        _gguf_vector_fixes += 1
print('GGUF vector compatibility:', _gguf_vector_fixes, flush=True)
print('Loading VAE and processor...', flush=True)
vae = AutoencoderKLQwenImage21.from_pretrained(
    ASSET_DIR/'vae', dtype=torch.float16, local_files_only=True).to('cuda:0')
processor = Qwen3VLProcessor.from_pretrained(ASSET_DIR/'processor', local_files_only=True)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    ASSET_DIR/'scheduler', local_files_only=True)
index = json.loads((ASSET_DIR/'model_index.json').read_text())
pipe = QwenImage21Pipeline(scheduler=scheduler, vae=vae, text_encoder=encoder,
    processor=processor, transformer=transformer, sample_sigmas=index.get('sample_sigmas'))
load_stats = {'copy_seconds': round(copy_seconds, 2),
              'loading_seconds': round(time.perf_counter()-started-copy_seconds, 2)}
print('Pipeline ready:', load_stats, flush=True)
for i in range(2):
    print(f'GPU {i}: {torch.cuda.memory_allocated(i)/2**30:.2f} GiB allocated')

import json, time, torch

PROMPT = "A small orange fox sitting beside a ceramic teacup, soft morning window light, detailed natural photography, clean background"
SEED = 42
WIDTH = HEIGHT = 512
STEPS = 35

@torch.inference_mode()
def generate_image(prompt, seed=42, width=512, height=512, steps=35):
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(i)
    started = time.perf_counter()
    pipe.register_modules(text_encoder=encoder)
    positive = pipe.encode_prompt(prompt, device=torch.device('cuda:1'))
    negative = pipe.encode_prompt(' ', device=torch.device('cuda:1'))
    torch.cuda.synchronize(1)
    encoded = time.perf_counter()
    positive = tuple(t.to('cuda:0') if t is not None else None for t in positive)
    negative = tuple(t.to('cuda:0') if t is not None else None for t in negative)
    pipe.register_modules(text_encoder=None)
    _step_events=[]
    def _record_step(pipeline,index,timestep,callback_kwargs):
        event=torch.cuda.Event(enable_timing=True);event.record(torch.cuda.current_stream(0));_step_events.append(event)
        return callback_kwargs
    _before=torch.cuda.Event(enable_timing=True);_before.record(torch.cuda.current_stream(0))
    try:
        result = pipe(
            prompt_embeds=positive[0], prompt_embeds_mask=positive[1],
            negative_prompt_embeds=negative[0], negative_prompt_embeds_mask=negative[1],
            width=width, height=height, num_inference_steps=steps,
            sigmas=torch.linspace(1, 0, steps+1)[:-1].tolist(),
            true_cfg_scale=5.0, use_kv_cache=True,callback_on_step_end=_record_step,
            generator=torch.Generator(device='cuda:0').manual_seed(seed)).images[0]
    finally:
        pipe.register_modules(text_encoder=encoder)
    _after=torch.cuda.Event(enable_timing=True);_after.record(torch.cuda.current_stream(0))
    torch.cuda.synchronize(0)
    _denoise=_before.elapsed_time(_step_events[-1])/1000 if _step_events else None
    _decode=_step_events[-1].elapsed_time(_after)/1000 if _step_events else None
    stats = {'model_id':'qwen-image-2.1','denoising_seconds':round(_denoise,3) if _denoise is not None else None,
             'decode_seconds':round(_decode,3) if _decode is not None else None,
             'denoising_steps':len(_step_events),
             'milliseconds_per_step':round(_denoise*1000/len(_step_events),2) if _step_events else None,
             'attention_backend':'sdpa','parallel_strategy':'encoder_split',
             'encoding_seconds': round(encoded-started, 2),
             'generation_seconds': round(time.perf_counter()-encoded, 2),
             'width': width, 'height': height, 'steps': steps, 'seed': seed,
             'peak_gpu_gib': [round(torch.cuda.max_memory_allocated(i)/2**30, 2) for i in range(2)]}
    return result, stats
