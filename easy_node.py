import base64
import io
import json
import os
from pathlib import Path
import threading
import hashlib
import numpy as np
import torch
from PIL import Image

_backends = {}
_registry_lock = threading.Lock()

def shared_backend():
    path=Path(os.environ.get('EASY_CONFIG_PATH',str(Path.home()/'.kaggle-qwen-client/settings.json')))
    if path.is_file():
        config=json.loads(path.read_text(encoding='utf-8'))
        config.setdefault('mode','batch')
        key=(str(path),path.stat().st_mtime_ns)
    else:
        user=os.environ.get('KAGGLE_USERNAME','')
        token=os.environ.get('KAGGLE_API_TOKEN') or os.environ.get('KAGGLE_KEY','')
        if not user or not token:raise RuntimeError('先在 GUI 保存一次账号配置，或设置 KAGGLE_USERNAME 和 KAGGLE_API_TOKEN。之后节点可独立运行。')
        config={'mode':'account','user':user,'token':token}
        key=('environment',user,hashlib.sha256(token.encode()).hexdigest())
    # The independent worker owns the queue and Kaggle connection. The GUI
    # and its HTTP service do not need to run.
    with _registry_lock:
        backend=_backends.get(key)
        if backend is None:
            for previous in _backends.values():
                if callable(getattr(previous,'close',None)):previous.close()
            _backends.clear()
            from .studio_client import SharedBackend
            backend=SharedBackend(config)
            _backends[key]=backend
    return backend


def tensor_png(tensor,mode='RGB'):
    """First ComfyUI IMAGE ([B,H,W,C]) or MASK ([B,H,W]/[H,W]) as PNG bytes."""
    array=tensor.detach().cpu().float().numpy()
    if mode=='L':
        while array.ndim>2:array=array[0]
    else:
        array=array[0] if array.ndim==4 else array
        array=array[...,:3]
    buffer=io.BytesIO()
    Image.fromarray((np.clip(array,0,1)*255).round().astype(np.uint8)).save(buffer,format='PNG')
    return buffer.getvalue()


def png_tensor(raw):
    image=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'),dtype=np.float32)/255
    return torch.from_numpy(image).unsqueeze(0)


def default_steps(model_id,steps,use_model_defaults):
    if model_id=="sdxl-lightning":return 4
    if model_id=="z-image-turbo" and use_model_defaults:return 9
    return steps


class KaggleEasyImage:
    @classmethod
    def INPUT_TYPES(cls):
        from .model_registry import SIZES
        return {'required':{'prompt':('STRING',{'multiline':True,'default':''}),'seed':('INT',{'default':42,'min':0,'max':2147483647}),'steps':('INT',{'default':20,'min':1,'max':50})},
                'optional':{'model_id':(['qwen-image-2.1','z-image-turbo','sdxl-lightning'],),'use_model_defaults':('BOOLEAN',{'default':True}),
                            'size':(list(SIZES),),'negative_prompt':('STRING',{'multiline':True,'default':''})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='generate'
    CATEGORY='Kaggle'
    def generate(self,prompt,seed,steps,model_id="qwen-image-2.1",use_model_defaults=True,size='512x512',negative_prompt=''):
        backend=shared_backend()
        steps=default_steps(model_id,steps,use_model_defaults)
        options={k:v for k,v in (('size',size),('negative_prompt',negative_prompt)) if v not in ('','512x512',None)}
        raw,stats=backend.generate(prompt,seed,steps,model_id=model_id,**options)
        return (png_tensor(raw),)


class CCMediaImageEdit:
    """Image-to-image or inpainting on the shared local worker (preview)."""
    @classmethod
    def INPUT_TYPES(cls):
        from .model_registry import SIZES
        return {'required':{'prompt':('STRING',{'multiline':True,'default':''}),'image':('IMAGE',),
                            'model_id':(['z-image-turbo','sdxl-lightning'],),'task':(['img2img','inpaint'],),
                            'strength':('FLOAT',{'default':0.6,'min':0.05,'max':1.0,'step':0.05}),
                            'seed':('INT',{'default':42,'min':0,'max':2147483647}),'size':(list(SIZES),)},
                'optional':{'mask':('MASK',),'steps':('INT',{'default':9,'min':1,'max':50}),'use_model_defaults':('BOOLEAN',{'default':True})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='edit'
    CATEGORY='CC Media Studio'
    def edit(self,prompt,image,model_id,task,strength,seed,size,mask=None,steps=9,use_model_defaults=True):
        if task=='inpaint' and mask is None:raise RuntimeError('局部编辑需要连接 MASK（白色/1 为重绘区域）。')
        backend=shared_backend()
        raw,stats=backend.generate(prompt,seed,default_steps(model_id,steps,use_model_defaults),model_id=model_id,size=size,task=task,
            strength=strength,image=tensor_png(image),mask=tensor_png(mask,'L') if task=='inpaint' else None)
        return (png_tensor(raw),)
