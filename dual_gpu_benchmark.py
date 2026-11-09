"""Explicit, isolated Qwen CFG experiment; does not modify the app configuration."""
import argparse
import gc
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True,help='Prepared Qwen INT8/Q4 asset directory')
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/cc-dual-cfg'))
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--prompt',default='An adult orange-haired catgirl sitting beside a sunlit window, cream sweater, detailed anime illustration, gentle warm light')
    args=parser.parse_args(argv)
    if not 1<=args.runs<=10:parser.error('--runs must be 1–10')
    if not 1<=args.steps<=100:parser.error('--steps must be 1–100')

    import numpy as np
    import torch
    from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
    from diffusers import QwenImage21Pipeline,AutoencoderKLQwenImage21,FlowMatchEulerDiscreteScheduler
    from dual_gpu import ParallelCFG,load_replica

    if torch.cuda.device_count()!=2:raise RuntimeError('Select GPU T4 x2')
    assets=args.assets
    args.output.mkdir(parents=True,exist_ok=True)
    # GGUF uses mmap. Keep its checkpoint on the VM's local disk.
    checkpoint_dir=tempfile.TemporaryDirectory(prefix='cc-cfg-',dir='/kaggle/temp')
    local_checkpoint=Path(checkpoint_dir.name)/'transformer-q4.gguf'
    source=assets/'transformer-q4.gguf'
    processor=Qwen3VLProcessor.from_pretrained(assets/'processor',local_files_only=True)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(assets/'scheduler',local_files_only=True)
    sigmas=json.loads((assets/'model_index.json').read_text()).get('sample_sigmas')
    encoder=Qwen3VLForConditionalGeneration.from_pretrained(assets/'text_encoder',dtype=torch.float16,
        device_map={'':1},attn_implementation='sdpa',local_files_only=True)
    pipe=QwenImage21Pipeline(scheduler=scheduler,vae=None,text_encoder=encoder,
        processor=processor,transformer=None,sample_sigmas=sigmas)
    with torch.inference_mode():
        positive=pipe.encode_prompt(args.prompt,device=torch.device('cuda:1'))
        negative=pipe.encode_prompt(' ',device=torch.device('cuda:1'))
        positive=tuple(v.to('cuda:0') if v is not None else None for v in positive)
        negative=tuple(v.to('cuda:0') if v is not None else None for v in negative)
    pipe.register_modules(text_encoder=None)
    del encoder,pipe
    gc.collect();torch.cuda.empty_cache()
    print('ENCODER_RELEASED',flush=True)
    shutil.copyfile(source,local_checkpoint)
    primary=load_replica(assets,'cuda:0',checkpoint_path=local_checkpoint)
    vae=AutoencoderKLQwenImage21.from_pretrained(assets/'vae',dtype=torch.float16,local_files_only=True).to('cuda:0')
    pipe=QwenImage21Pipeline(scheduler=scheduler,vae=vae,text_encoder=None,
        processor=processor,transformer=primary,sample_sigmas=sigmas)

    @torch.inference_mode()
    def sample():
        torch.cuda.synchronize(0)
        start=time.perf_counter()
        image=pipe(prompt_embeds=positive[0],prompt_embeds_mask=positive[1],
            negative_prompt_embeds=negative[0],negative_prompt_embeds_mask=negative[1],
            width=512,height=512,num_inference_steps=args.steps,true_cfg_scale=5.0,use_kv_cache=True,
            sigmas=torch.linspace(1,0,args.steps+1)[:-1].tolist(),
            generator=torch.Generator(device='cuda:0').manual_seed(args.seed)).images[0]
        torch.cuda.synchronize(0)
        return image,round(time.perf_counter()-start,3)

    reference,cold_seconds=sample()
    reference.save(args.output/'baseline.png')
    baseline=[];baseline_matches=[]
    for _ in range(args.runs):
        image,seconds=sample()
        baseline.append(seconds)
        baseline_matches.append(bool(np.array_equal(np.asarray(reference),np.asarray(image))))
    report={'gpu':[torch.cuda.get_device_name(i) for i in range(2)],'steps':args.steps,'seed':args.seed,
        'resolution':[512,512],'runs':args.runs,'mode':'preencoded_encoder_released',
        'includes_encoder_transfer_or_reload':False,'production_enabled':False,
        'baseline_cold_seconds':cold_seconds,'baseline_seconds':baseline,
        'baseline_same_pixels':baseline_matches,
        'baseline_median_seconds':statistics.median(baseline)}
    replica=None
    original=primary.forward
    try:
        start=time.perf_counter()
        replica=load_replica(assets,'cuda:1',checkpoint_path=local_checkpoint)
        report['replica_load_seconds']=round(time.perf_counter()-start,3)
        parallel_times=[];errors=[];pairs=[]
        for repeat in range(args.runs):
            with ParallelCFG(primary,replica,negative) as parallel:
                image,seconds=sample()
            delta=np.abs(np.asarray(reference,dtype=np.int16)-np.asarray(image,dtype=np.int16))
            errors.append({'max':int(delta.max()),'mean':float(delta.mean())})
            pairs.append(parallel.calls);parallel_times.append(seconds)
            image.save(args.output/f'parallel-{repeat}.png')
        report.update(parallel_seconds=parallel_times,parallel_median_seconds=statistics.median(parallel_times),
            pixel_errors=errors,cfg_pairs=pairs)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error).split('\n')[0][:200])
    finally:
        primary.forward=original
        if replica is not None:replica.to('cpu')
        del replica
        gc.collect();torch.cuda.empty_cache()
        report['forward_restored']=primary.forward==original
    rollback,seconds=sample()
    rollback.save(args.output/'rollback.png')
    report.update(rollback_seconds=seconds,
        rollback_same_pixels=bool(np.array_equal(np.asarray(reference),np.asarray(rollback))))
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print('CC_DUAL_FINAL',json.dumps(report),flush=True)
    if not report['rollback_same_pixels']:raise RuntimeError('Rollback output differs from baseline')
    if 'error_type' in report:raise RuntimeError('Parallel experiment failed; original forward restored')
    return report


if __name__=='__main__':main()
