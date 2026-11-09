import atexit
import base64
import io
import json
import os
import re
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .session_runtime import JupyterTransport, ModelSession
from .session_backend import bootstrap_code, generation_code

_sessions = {}
_registry_lock = threading.Lock()


class KaggleQwenSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'prompt': ('STRING', {'multiline': True, 'default': ''}),
            'dataset_handle': ('STRING', {'default': 'your-user/qwen-image-t4-assets'}),
            'lifetime_minutes': ('INT', {'default': 60, 'min': 1, 'max': 600}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2147483647}),
            'steps': ('INT', {'default': 20, 'min': 1, 'max': 50}),
        }}

    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('image', 'timing')
    FUNCTION = 'generate'
    CATEGORY = 'Kaggle'

    def generate(self, prompt, dataset_handle, lifetime_minutes, seed, steps):
        if not prompt.strip() or len(prompt) > 4000:
            raise ValueError('Prompt must contain 1–4000 characters')
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', dataset_handle):
            raise ValueError('Dataset handle must be user/dataset')
        url = os.environ.get('KAGGLE_JUPYTER_URL', '').strip()
        if not url:
            raise RuntimeError('Configure the official Kaggle Jupyter server URL in KAGGLE_JUPYTER_URL. A Kaggle API token alone does not supply this URL.')
        # Keep connection credentials out of workflow JSON and output text.
        key = (url, dataset_handle, int(lifetime_minutes))
        with _registry_lock:
            if key not in _sessions:
                # Avoid leaving old models allocated when connection/settings change.
                for old in _sessions.values():
                    old.close()
                _sessions.clear()
                bootstrap = bootstrap_code(dataset_handle)
                _sessions[key] = ModelSession(lambda: JupyterTransport(url), bootstrap, lifetime_minutes * 60)
            session = _sessions[key]
        code = generation_code(prompt, seed, steps)
        output = session.execute(code)
        results = [line[len('COMFY_RESULT:'):] for line in output.splitlines() if line.startswith('COMFY_RESULT:')]
        if len(results) != 1:
            raise RuntimeError('No complete image received from the remote session')
        data = json.loads(results[0])
        image = Image.open(io.BytesIO(base64.b64decode(data['png']))).convert('RGB')
        return (torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0),
                json.dumps(data['stats'], ensure_ascii=False))


@atexit.register
def close_sessions():
    for session in list(_sessions.values()):
        try:
            session.close()
        except Exception:
            pass
