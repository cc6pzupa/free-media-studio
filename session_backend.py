"""Persistent image backend attached to the running Kaggle notebook kernel."""
import base64
import json
import re
import threading
import time
from pathlib import Path

if __package__:
    from .session_runtime import JupyterTransport, ModelSession
else:
    from session_runtime import JupyterTransport, ModelSession


if __package__:
    from .model_registry import DEFAULT_MODEL,model_profile,MODEL_SOURCES
else:
    from model_registry import DEFAULT_MODEL,model_profile,MODEL_SOURCES


def model_switch_code(model_id):
    model_profile(model_id)
    return f"""
_CC_REQUESTED_MODEL = {model_id!r}
_CC_PREVIOUS_MODEL = globals().get('_CC_MODEL_ID','qwen-image-2.1')
if _CC_PREVIOUS_MODEL != _CC_REQUESTED_MODEL:
    if (globals().get('pipe') is not None or globals().get('encoder') is not None) and not (globals().get('_COMFY_QWEN_STATE') or globals().get('_CC_MANAGED_MODEL')):
        raise RuntimeError('Existing unmanaged model; select the managed notebook kernel.')
    # Do not retain aliases to an old pipeline or its GPU modules during collection.
    for _cc_key in ('pipe','encoder','transformer','vae','processor','scheduler','generate_image','_COMFY_QWEN_STATE','_CC_NEGATIVE_CACHE','_CC_PROMPT_CACHE','_value','_module','_decoded'):
        globals().pop(_cc_key,None)
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:pass
_CC_MODEL_ID = _CC_REQUESTED_MODEL
_CC_MANAGED_MODEL = True
"""


def additional_bootstrap_code(model_id,install=True):
    profile=model_profile(model_id)
    sources=MODEL_SOURCES[model_id]
    helpers='\n'.join(Path(__file__).with_name(name).read_text() for name in ('asset_manager.py','cache_manager.py','model_assets.py'))
    runtime=Path(__file__).with_name('remote_models.py').read_text()
    return model_switch_code(model_id)+f"""
import json,uuid
_CC_MODEL_PROFILE = {profile!r}
_CC_MODEL_SOURCES = {sources!r}
# JSON arrays are used in saved manifests, including revision tuples.
_CC_MODEL_SOURCES = json.loads(json.dumps(_CC_MODEL_SOURCES))
exec(compile({helpers!r},'<cc-model-assets>','exec'),globals())
_existing_state=globals().get('_COMFY_QWEN_STATE')
if not (_existing_state and globals().get('pipe') is not None and callable(globals().get('generate_image'))):
    if {install!r}:
        import subprocess,sys
        subprocess.run([sys.executable,'-m','pip','install','-q','bitsandbytes==0.50.2','safetensors','diffusers>=0.36,<0.43','transformers>=4.51,<6','accelerate>=1.2'],check=True,timeout=900)
    for _cc_partial in ('pipe','encoder','transformer','vae','processor','scheduler','generate_image'):
        globals().pop(_cc_partial,None)
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    exec(compile({runtime!r},'<cc-model-runtime>','exec'),globals())
    _existing_state={{'dataset':'model/'+_CC_MODEL_ID,'model_id':_CC_MODEL_ID,'runtime_id':uuid.uuid4().hex,'load_count':1,'generations':0,'loaded_this_request':True}}
else:
    _existing_state['loaded_this_request']=False
_COMFY_QWEN_STATE=_existing_state
if globals().get('_QWEN_CACHE_HANDLE') and globals().get('_QWEN_CACHE_TOKEN'):
    save_private_cache(ASSET_DIR,encoder)
print('COMFY_SESSION_READY:'+json.dumps({{'runtime_id':_existing_state['runtime_id'],'load_count':_existing_state['load_count'],'reused':not _existing_state['loaded_this_request']}}),flush=True)
"""


def bootstrap_code(dataset, source=None, install=True, asset_mode="dataset", model_id=DEFAULT_MODEL):
    model_profile(model_id)
    if model_id!=DEFAULT_MODEL:
        if asset_mode not in ("auto","huggingface"):raise ValueError("Additional models require automatic account mode.")
        return additional_bootstrap_code(model_id,install)
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', dataset):
        raise ValueError('Dataset 请填写 用户名/数据集名称')
    requested_dataset = dataset
    if asset_mode in ('huggingface','auto'):
        dataset = 'huggingface/qwen-image-21'
    if source is None:
        source = Path(__file__).with_name('remote_bootstrap.py').read_text(encoding='utf-8')
        source = source.replace("'faberbruntch/qwen-image-t4-assets'", repr(dataset))
        helpers = Path(__file__).with_name('asset_manager.py').read_text(encoding='utf-8')
        source = helpers + "\n_QWEN_ASSET_MODE = " + repr(asset_mode) + "\n" + source
    cache_helpers = Path(__file__).with_name('asset_manager.py').read_text(encoding='utf-8') + '\n' + Path(__file__).with_name('cache_manager.py').read_text(encoding='utf-8')
    return model_switch_code(model_id)+f'''
import json, os, uuid
exec(compile({cache_helpers!r}, '<qwen-private-cache>', 'exec'), globals())
_expected_dataset = {dataset!r}
_existing_state = globals().get('_COMFY_QWEN_STATE')
_model_ready = (globals().get('pipe') is not None and globals().get('encoder') is not None
                and callable(globals().get('generate_image')))
if _existing_state and _existing_state['dataset'] != _expected_dataset:
    if {asset_mode!r} in ('huggingface','auto') and _existing_state['dataset'] == {requested_dataset!r} and globals().get('_hf_mode') is True:
        _existing_state['dataset'] = _expected_dataset
    else:
        raise RuntimeError('This notebook holds a different dataset. Use its existing dataset or restart the notebook explicitly.')
if _model_ready:
    if not _existing_state:
        if _expected_dataset not in str(globals().get('source', '')):
            raise RuntimeError('An existing model was found but its dataset cannot be verified. Select the correct notebook kernel; do not load a second model.')
        _existing_state = {{'dataset': _expected_dataset, 'runtime_id': uuid.uuid4().hex,
                           'load_count': 1, 'generations': 0, 'adopted': True}}
    _existing_state['loaded_this_request'] = False
else:
    if globals().get('pipe') is not None or globals().get('encoder') is not None:
        raise RuntimeError('A partially loaded model exists. Restart this notebook session before loading again.')
    if {install!r}:
        import subprocess, sys
        print('Installing image runtime dependencies once...', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'bitsandbytes==0.50.2', 'gguf'], check=True, timeout=900)
    exec(compile({source!r}, '<qwen-model-bootstrap>', 'exec'), globals())
    _existing_state = {{'dataset': _expected_dataset, 'runtime_id': uuid.uuid4().hex,
                       'load_count': (_existing_state or {{}}).get('load_count', 0) + 1,
                       'generations': 0, 'adopted': False, 'loaded_this_request': True}}
_COMFY_QWEN_STATE = _existing_state
if globals().get('_QWEN_CACHE_HANDLE') and globals().get('_QWEN_CACHE_TOKEN'):
    save_private_cache(ASSET_DIR, encoder)
print('COMFY_SESSION_READY:' + json.dumps({{'runtime_id': _COMFY_QWEN_STATE['runtime_id'],
    'load_count': _COMFY_QWEN_STATE['load_count'], 'reused': not _COMFY_QWEN_STATE['loaded_this_request']}}), flush=True)
'''


def generation_code(prompt, seed, steps, model_id=DEFAULT_MODEL):
    model_profile(model_id)
    args = repr(json.dumps({'prompt': prompt, 'seed': seed, 'steps': steps}, ensure_ascii=False))
    return (
        'import json, io, base64\n' + \
        'assert globals().get("_CC_MODEL_ID","qwen-image-2.1") == '+repr(model_id)+', "Active model changed; reconnect before generating."\n' + \
        'args = json.loads(' + args + ')\n'
        "im, stats = generate_image(args['prompt'], seed=args['seed'], steps=args['steps'], width=512, height=512)\n"
        "_COMFY_QWEN_STATE['generations'] += 1\n"
        "stats = dict(stats, model_id=globals().get('_CC_MODEL_ID','qwen-image-2.1'), runtime_id=_COMFY_QWEN_STATE['runtime_id'], "
        "model_load_count=_COMFY_QWEN_STATE['load_count'], "
        "model_reused=not _COMFY_QWEN_STATE['loaded_this_request'], "
        "generation_number=_COMFY_QWEN_STATE['generations'], asset_source=globals().get('_QWEN_RESOLVED_SOURCE',globals().get('_QWEN_ASSET_MODE','dataset')), "
        "cache_status=globals().get('_QWEN_CACHE_STATUS','disabled'))\n"
        "_COMFY_QWEN_STATE['loaded_this_request'] = False\n"
        "buf = io.BytesIO()\nim.save(buf, format='PNG')\n"
        "print('COMFY_RESULT:' + json.dumps({'png': base64.b64encode(buf.getvalue()).decode(), 'stats': stats}), flush=True)\n"
    )


class SessionBackend:
    mode = 'session'

    def __init__(self, user, dataset, jupyter_url, kernel_id=None, transport_factory=None, asset_mode="dataset", cache_handle=None, api_token=None, model_id=DEFAULT_MODEL):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', user):
            raise ValueError('请填写 Kaggle 用户名')
        self.user, self.dataset, self.token = user, dataset, ''
        self.model_id=model_profile(model_id)['id']
        self.jupyter_url = jupyter_url.strip()
        self.kernel_id = kernel_id or None
        self.progress = None
        self.progress_buffer = ""
        # Validate the connection before saving it, without starting a kernel.
        probe = JupyterTransport(self.jupyter_url, kernel_id=self.kernel_id)
        probe.close()
        factory = transport_factory or (lambda: JupyterTransport(self.jupyter_url, kernel_id=self.kernel_id, output_callback=self._output))
        bootstrap = bootstrap_code(dataset, asset_mode=asset_mode, model_id=self.model_id)
        if cache_handle and api_token:
            if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+',cache_handle):
                raise ValueError('Invalid private cache handle')
            bootstrap = ('_QWEN_CACHE_HANDLE = '+repr(cache_handle)+'\n_QWEN_CACHE_TOKEN = '+repr(api_token)+
                '\ntry:\n    exec('+repr(bootstrap)+', globals())\nfinally:\n'
                "    globals().pop('_QWEN_CACHE_TOKEN', None)\n    globals().pop('_QWEN_CACHE_HANDLE', None)\n")
        self.session = ModelSession(factory, bootstrap, lifetime_seconds=None)
        self.lock = threading.Lock()

    def _output(self, text):
        if not self.progress:
            return
        self.progress_buffer += text
        lines = self.progress_buffer.split('\n')
        self.progress_buffer = lines.pop()
        for line in lines:
            if line.startswith('COMFY_PROGRESS:'):
                try: self.progress(json.loads(line[len('COMFY_PROGRESS:'):]))
                except (ValueError, TypeError): pass
            else:
                if line.startswith(('Starting upload for file ', 'Upload successful: ')):
                    self.progress({'phase':'cache-upload','message':'正在上传私有缓存：'+line.rsplit('/',1)[-1]})
                for prefix, message in [
                    ('Installing image runtime', '正在安装运行依赖。'),
                    ('Loading INT8', '正在加载文本编码器。'),
                    ('Loading Q4', '正在加载主模型。'),
                    ('Loading VAE', '正在加载图像解码器。'),
                    ('COMFY_SESSION_READY:', '模型已就绪，正在生图；后续直接复用。'),
                ]:
                    if line.startswith(prefix): self.progress({'phase':'generating' if prefix=='COMFY_SESSION_READY:' else 'loading','message':message})
        if len(self.progress_buffer)>2*1024*1024: self.progress_buffer=''

    def generate(self, prompt, seed=42, steps=20):
        return self.generate_with_progress(prompt, seed, steps)

    def generate_with_progress(self, prompt, seed=42, steps=20, progress=None):
        with self.lock:
            started = time.monotonic()
            self.progress = progress
            if progress:
                progress({'phase':'generating' if self.session.transport is not None else 'session', 'message':'正在复用已加载的模型生图。' if self.session.transport is not None else '正在检查远端模型，首次使用将自动准备。'})
            try:
                output = self.session.execute(generation_code(prompt, seed, steps, self.model_id), timeout=7200)
                matches = [line[len('COMFY_RESULT:'):] for line in output.splitlines()
                           if line.startswith('COMFY_RESULT:')]
                if len(matches) != 1:
                    raise ValueError('Missing image result')
                result = json.loads(matches[0])
                raw = base64.b64decode(result['png'], validate=True)
                stats = dict(result['stats'], request_seconds=round(time.monotonic() - started, 2), backend='session')
                return raw, stats
            except Exception:
                # Connection URLs contain credentials. Never include raw transport errors.
                raise RuntimeError('常驻生图失败。请检查 Notebook 是否仍在运行及连接地址是否有效。本次请求不会自动改走批处理或重复提交。') from None
            finally:
                self.progress = None

    def close(self):
        # Disconnect the client while leaving the notebook's model in memory.
        self.session.close()
