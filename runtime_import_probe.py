"""Inspect inference imports in child processes without changing installed packages."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time

CHECKS=[
    ('torch','import torch; print(torch.__version__, torch.cuda.device_count())'),
    ('diffusers','import diffusers; print(diffusers.__version__)'),
    ('transformers','import transformers; print(transformers.__version__)'),
    ('bitsandbytes','import bitsandbytes; print(bitsandbytes.__version__)'),
    ('gguf','import gguf; print("OK")'),
    ('qwen_pipeline','from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21, GGUFQuantizationConfig; print("OK")'),
    ('qwen_encoder','from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor; print("OK")'),
    ('gguf_utils','from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor; print("OK")'),
    ('combined_runtime','import torch, diffusers, kagglehub, numpy; from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor; from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig; from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor; print("OK")'),
]


def probe_imports(python=None,timeout=120,checks=None):
    python=python or sys.executable
    rows=[]
    for name,source in CHECKS if checks is None else checks:
        print('CHECK_BEGIN',name,flush=True)
        started=time.perf_counter()
        try:
            result=subprocess.run([python,'-X','faulthandler','-c',source],
                capture_output=True,text=True,timeout=timeout)
            row={'check':name,'returncode':result.returncode,
                 'stdout':result.stdout[-500:],'stderr':result.stderr[-2500:]}
        except subprocess.TimeoutExpired:
            row={'check':name,'timeout':True}
        row['seconds']=round(time.perf_counter()-started,2)
        rows.append(row)
        print('CHECK_END',json.dumps(row),flush=True)
    return rows


def main():
    versions={}
    for name in ['torch','diffusers','transformers','bitsandbytes','gguf','accelerate','numpy']:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    print('INSTALLED',json.dumps(versions),flush=True)
    report={'versions':versions,'checks':probe_imports()}
    output=Path('.benchmarks/import-diagnostics.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print('DIAG_COMPLETE',flush=True)


if __name__=='__main__':main()
