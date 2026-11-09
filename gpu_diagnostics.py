"""Run inside a Kaggle cell after loading a model. No credentials are collected."""

def inspect_gpu_runtime():
    import subprocess
    import torch
    gpus=[]
    for index in range(torch.cuda.device_count()):
        prop=torch.cuda.get_device_properties(index)
        gpus.append({'index':index,'name':prop.name,'memory_gib':round(prop.total_memory/2**30,2),
                     'compute_capability':list(torch.cuda.get_device_capability(index))})
    peers=[[bool(torch.cuda.can_device_access_peer(a,b)) if a!=b else True for b in range(len(gpus))] for a in range(len(gpus))]
    model=globals().get('transformer')
    quantized=bool(getattr(model,'is_quantized',False) or getattr(model,'quantization_method',None))
    cp=bool(getattr(model,'_cp_plan',None));tp=bool(getattr(model,'_tp_plan',None))
    try:topology=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):topology=None
    return {'gpus':gpus,'peer_access':peers,'topology':topology,
        'model_id':globals().get('_CC_MODEL_ID'),'transformer_quantized':quantized,
        'context_parallel_plan':cp,'tensor_parallel_plan':tp,
        'single_image_parallel_ready':False,
        'notes':['Default encoder/model split saves memory; it does not parallelize denoising.',
                 'A parallel plan alone does not validate GGUF, causal masks, KV cache or memory fit.',
                 'Separate image replicas improve throughput, not latency of one image.']}


if __name__=='__main__':
    import json
    print(json.dumps(inspect_gpu_runtime(),ensure_ascii=False,indent=2))
