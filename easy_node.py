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

class KaggleEasyImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'prompt':('STRING',{'multiline':True,'default':''}),'seed':('INT',{'default':42,'min':0,'max':2147483647}),'steps':('INT',{'default':20,'min':1,'max':50})},'optional':{'model_id':(['qwen-image-2.1','z-image-turbo','sdxl-lightning'],),'use_model_defaults':('BOOLEAN',{'default':True})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='generate'
    CATEGORY='Kaggle'
    def generate(self,prompt,seed,steps,model_id="qwen-image-2.1",use_model_defaults=True):
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
        mode=config.get('mode','batch')
        with _registry_lock:
            backend=_backends.get(key)
            if backend is None:
                for previous in _backends.values():
                    if callable(getattr(previous,'close',None)):previous.close()
                _backends.clear()
                from .studio_client import SharedBackend
                backend=SharedBackend(config)
                _backends[key]=backend
        if model_id=="sdxl-lightning":steps=4
        elif model_id=="z-image-turbo" and use_model_defaults:steps=9
        raw,stats=backend.generate(prompt,seed,steps,model_id=model_id)
        image=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'),dtype=np.float32)/255
        return (torch.from_numpy(image).unsqueeze(0),)
