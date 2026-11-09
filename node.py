import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class KaggleQwen:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'prompt': ('STRING', {'multiline': True, 'default': ''}),
            'kaggle_user': ('STRING', {'default': ''}),
            'kernel_slug': ('STRING', {'default': 'qwen-image-comfy-job'}),
            'dataset_slug': ('STRING', {'default': 'qwen-image-t4-assets'}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2147483647}),
            'steps': ('INT', {'default': 35, 'min': 1, 'max': 50}),
            'timeout_minutes': ('INT', {'default': 45, 'min': 5, 'max': 180}),
        }}

    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'generate'
    CATEGORY = 'Kaggle'

    def generate(self, prompt, kaggle_user, kernel_slug, dataset_slug, seed, steps, timeout_minutes):
        if not prompt.strip():
            raise ValueError('Prompt cannot be empty')
        for value in (kaggle_user, kernel_slug, dataset_slug):
            if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
                raise ValueError('Kaggle username and slugs may contain only letters, digits, hyphens and underscores')
        run_id = uuid.uuid4().hex
        image_name = 'result_' + run_id + '.png'
        notebook = json.loads((Path(__file__).parent / 'template.ipynb').read_text())
        cell = notebook['cells'][3]
        code = ''.join(cell['source'])
        code = re.sub(r'^PROMPT\s*=.*$', lambda _: 'PROMPT = ' + repr(prompt), code, count=1, flags=re.M)
        code = re.sub(r'^SEED\s*=.*$', 'SEED = ' + str(seed), code, count=1, flags=re.M)
        code = re.sub(r'^STEPS\s*=.*$', 'STEPS = ' + str(steps), code, count=1, flags=re.M)
        code = code.replace("'/kaggle/working/result.png'", repr('/kaggle/working/' + image_name))
        cell['source'] = code.splitlines(keepends=True)
        for c in notebook['cells']:
            if c['cell_type'] == 'code':
                c['execution_count'] = None
                c['outputs'] = []
        cell = notebook['cells'][2]
        code = ''.join(cell['source']).replace('faberbruntch/qwen-image-t4-assets', kaggle_user + '/' + dataset_slug)
        cell['source'] = code.splitlines(keepends=True)
        slug = kaggle_user + '/' + kernel_slug
        with tempfile.TemporaryDirectory(prefix='kaggle_comfy_') as tmp:
            path = Path(tmp)
            (path / 'job.ipynb').write_text(json.dumps(notebook, ensure_ascii=False), encoding='utf-8')
            meta = {'id': slug, 'title': kernel_slug, 'code_file': 'job.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [kaggle_user + '/' + dataset_slug], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
            (path / 'kernel-metadata.json').write_text(json.dumps(meta), encoding='utf-8')
            def run(*args):
                return subprocess.run(['kaggle', *args], capture_output=True, text=True, check=True, timeout=120)
            run('kernels', 'push', '-p', str(path))
            deadline = time.monotonic() + timeout_minutes * 60
            while time.monotonic() < deadline:
                status = run('kernels', 'status', slug).stdout.lower()
                if 'error' in status or 'cancel' in status:
                    raise RuntimeError('Kaggle job failed: ' + status)
                if 'complete' in status:
                    run('kernels', 'output', slug, '-p', str(path / 'output'))
                    result = path / 'output' / image_name
                    if result.exists():
                        image = np.asarray(Image.open(result).convert('RGB'), dtype=np.float32) / 255.0
                        return (torch.from_numpy(image).unsqueeze(0),)
                    raise RuntimeError('Kaggle job completed without the expected image. Check its output and logs: ' + slug)
                time.sleep(15)
            raise TimeoutError('Kaggle job timed out. Check its status before retrying.')


NODE_CLASS_MAPPINGS = {'KaggleQwen': KaggleQwen}
NODE_DISPLAY_NAME_MAPPINGS = {'KaggleQwen': 'Kaggle Qwen Image'}
